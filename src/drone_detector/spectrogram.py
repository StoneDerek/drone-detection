"""Explicit spectrogram tensors, a compact CNN, Grad-CAM and a non-neural SVM.

Experimental models share the existing causal recording/stream engine. Model
packages contain numeric NPZ arrays and JSON, with no executable serialization.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import signal
from scipy.special import expit
from scipy.sparse import csr_matrix

from .common import digest, file_hash, read_json


@dataclass(frozen=True)
class SpectrogramConfig:
    axis: str = 'linear'
    intensity: str = 'log_power'
    n_fft: int = 4096
    stft_hop: int = 256
    mel_bins: int = 64
    sample_rate: int = 16000
    window_seconds: float = 3.
    hop_seconds: float = .5
    f_min: float = 50.
    f_max: float = 7600.
    min_audio_seconds: float = .25
    silence_rms: float = 1e-5
    dynamic_range_db: float = 80.
    power_reference: str = 'context_peak'
    normalize_rms: bool | str = False
    gain_target_rms: float = .05
    gain_max_db: float = 20.

    def __post_init__(self):
        if self.axis not in {'linear', 'mel', 'linear_filterbank'} or self.intensity not in {'log_power', 'linear_power'}:
            raise ValueError('Unknown spectrogram axes/intensity')
        if not 16 <= self.n_fft <= self.window_samples or not 1 <= self.stft_hop <= self.n_fft:
            raise ValueError('Invalid analysis window or hop')
        if not 0 <= self.f_min < self.f_max <= self.sample_rate / 2 or self.mel_bins < 4:
            raise ValueError('Invalid spectral frequency range')
        if not 0 < self.hop_seconds <= self.window_seconds or self.dynamic_range_db <= 0:
            raise ValueError('Invalid context or dynamic range')
        if self.power_reference not in {'context_peak', 'unit_stft_power'} or not (isinstance(self.normalize_rms, bool) or self.normalize_rms == 'exact'):
            raise ValueError('Invalid spectral amplitude rule')
        if not 0 < self.gain_target_rms <= .5 or not 0 <= self.gain_max_db <= 40:
            raise ValueError('Invalid reference RMS or gain cap')

    @property
    def window_samples(self):
        return round(self.window_seconds * self.sample_rate)

    @property
    def hop_samples(self):
        return round(self.hop_seconds * self.sample_rate)

    def to_dict(self):
        return asdict(self)


@lru_cache(maxsize=32)
def axes(config):
    frequencies = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
    times = (config.n_fft / 2 + np.arange(1 + (config.window_samples - config.n_fft) // config.stft_hop)
             * config.stft_hop) / config.sample_rate
    if config.axis == 'linear':
        return frequencies[(frequencies >= config.f_min) & (frequencies <= config.f_max)], times, None
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)  # noqa: E731
    hz = lambda value: 700 * (10 ** (value / 2595) - 1)  # noqa: E731
    edges = (np.linspace(config.f_min, config.f_max, config.mel_bins + 2) if config.axis == 'linear_filterbank'
             else hz(np.linspace(mel(config.f_min), mel(config.f_max), config.mel_bins + 2)))
    bank = np.maximum(0, np.minimum(
        (frequencies[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None],
        (edges[2:, None] - frequencies[None]) / (edges[2:] - edges[1:-1])[:, None]))
    bank /= np.maximum(bank.sum(axis=1, keepdims=True), 1e-12)
    return edges[1:-1], times, csr_matrix(bank)


def spectrogram(samples, config):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not 0 < len(samples) <= config.window_samples or not np.isfinite(samples).all():
        raise ValueError('Expected finite nonempty mono context')
    if config.normalize_rms:
        from .audio import normalize_context_gain
        samples = normalize_context_gain(samples, config)
    wave = np.pad(samples, (0, config.window_samples - len(samples)))
    _, _, stft = signal.stft(wave, fs=config.sample_rate, nperseg=config.n_fft,
                            noverlap=config.n_fft - config.stft_hop, boundary=None, padded=False)
    power = np.abs(stft).astype(np.float64) ** 2
    _, _, bank = axes(config)
    if bank is not None:
        power = bank @ power
    else:
        f = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
        power = power[(f >= config.f_min) & (f <= config.f_max)]
    peak = float(power.max())
    if peak <= 1e-30:
        return np.zeros(power.shape, dtype=np.float32)
    relative = power / (peak if config.power_reference == 'context_peak' else 1.)
    if config.intensity == 'log_power':
        # The reference is explicit. A fixed value of 1 is unit STFT power,
        # not an acoustic sound-pressure calibration.
        relative = np.maximum(0., 1 + 10 * np.log10(np.maximum(relative, 1e-30)) / config.dynamic_range_db)
    return np.clip(relative, 0, 1).astype(np.float32)


def descriptors(images):
    """Non-neural per-frequency temporal means/std; retain narrow frequency bins."""
    return np.concatenate([images.mean(axis=-1), images.std(axis=-1)], axis=-1)


def tensorflow():
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    return tf


def cnn_spec(variant='compact'):
    """Named, fixed architectures; old variants retain their exact layout."""
    if variant not in {'compact', 'frequency_context', 'coverage', 'coverage_medium',
                       'frequency_aware', 'global_readout'}:
        raise ValueError('Unknown fixed CNN architecture')
    channels = [32, 64, 128] if variant in {'coverage_medium', 'frequency_aware', 'global_readout'} else [6, 8, 16]
    frequency_stride = 4 if variant in {'compact', 'frequency_context'} else 3
    layers = [{'filters': channels[0], 'kernel': [5, 5], 'stride': [2, 2], 'name': 'conv1'},
              {'filters': channels[1], 'kernel': [3, 3], 'stride': [frequency_stride, 2], 'name': 'conv2'},
              {'filters': channels[2], 'kernel': [3, 3], 'stride': [frequency_stride, 2],
               'name': 'conv3' if variant == 'frequency_context' else 'features'}]
    if variant == 'frequency_context':
        layers.append({'filters': 16, 'kernel': [5, 1], 'stride': [1, 1], 'name': 'features'})
    return layers


def cnn_weight_shapes(variant='compact'):
    shapes, channels = [], 1
    for layer in cnn_spec(variant):
        shapes.extend([(*layer['kernel'], channels, layer['filters']), (layer['filters'],)])
        channels = layer['filters']
    if variant == 'frequency_aware':
        return shapes + [(15*channels, 32), (32,), (32, 1), (1,)]
    if variant == 'global_readout':
        return shapes + [(channels, 473), (473,), (473, 1), (1,)]
    return shapes + [(channels, 1), (1,)]


def build_cnn(seed=42, variant='compact', dropout_rate=.1):
    if not np.isfinite(dropout_rate) or not 0 <= dropout_rate < 1:
        raise ValueError('Dropout rate must be finite and in [0, 1)')
    layers = cnn_spec(variant)
    tf = tensorflow()
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(256 if variant == 'frequency_aware' else None, None, 1), name='spectrogram')
    regularizer = tf.keras.regularizers.l2(1e-4)
    x = inputs
    for layer in layers:
        x = tf.keras.layers.Conv2D(layer['filters'], layer['kernel'], strides=layer['stride'],
            padding='same', activation='relu', kernel_regularizer=regularizer, name=layer['name'])(x)
    if variant == 'frequency_aware':
        x = tf.keras.layers.Lambda(lambda value: tf.reduce_mean(value, axis=2), name='pool')(x)
        x = tf.keras.layers.Flatten(name='frequency_positions')(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D(name='pool')(x)
    x = tf.keras.layers.Dropout(.1, name='dropout')(x)
    if variant in {'frequency_aware', 'global_readout'}:
        x = tf.keras.layers.Dense(32 if variant == 'frequency_aware' else 473, activation='relu', name='readout')(x)
    outputs = tf.keras.layers.Dense(1, name='logit')(x)
    network = tf.keras.Model(inputs, outputs, name='small_spectrogram_cnn')
    # Construct the original seeded layers before changing the rate. Creating
    # Dropout(0) directly changes Keras RNG consumption and later initial weights.
    network.get_layer('dropout').rate = dropout_rate
    return network


class SpectralModel:
    def __init__(self, metadata, arrays):
        self.metadata, self.arrays = metadata, arrays
        self.config = SpectrogramConfig(**metadata['features'])
        if metadata.get('architecture', {}).get('variant') == 'frequency_aware' and len(axes(self.config)[0]) != 256:
            raise ValueError('Frequency-preserving head requires the declared256-bin frontend')
        self.encoder = None
        self._net = self._forward = None
        if metadata['family'] == 'rbf_svm':
            self._support_norm = np.sum(arrays['support_vectors'] ** 2, axis=1)

    @property
    def model_id(self):
        return self.metadata['model_id']

    def network(self):
        if self.metadata['family'] != 'spectrogram_cnn':
            raise ValueError('Grad-CAM requires a convolutional model')
        if self._net is None:
            tf = tensorflow()
            architecture = self.metadata.get('architecture', {})
            self._net = build_cnn(variant=architecture.get('variant', 'compact'),
                                  dropout_rate=architecture.get('dropout', .1))
            self._net.set_weights([self.arrays[f'w{i}'] for i in range(len(self.metadata['weight_shapes']))])
            self._forward = tf.function(lambda x: self._net(x, training=False),
                input_signature=[tf.TensorSpec([None, None, None, 1], tf.float32)])
        return self._net

    def score_images(self, images):
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 3 or not np.isfinite(images).all():
            raise ValueError('Expected batch × frequency × time')
        if self.metadata['family'] == 'spectrogram_cnn':
            self.network()
            return expit(self._forward(images[..., None]).numpy().ravel())
        values = (descriptors(images).astype(np.float64) - self.arrays['mean']) / self.arrays['scale']
        distance = np.maximum(0, (values ** 2).sum(axis=1)[:, None] + self._support_norm[None]
                              - 2 * values @ self.arrays['support_vectors'].T)
        margin = np.exp(-self.metadata['gamma'] * distance) @ self.arrays['dual_coef'] + self.arrays['intercept'][0]
        return expit(margin)

    def score(self, samples):
        return float(self.score_images(spectrogram(samples, self.config)[None])[0])

    def warmup(self):
        if self.metadata['family'] == 'spectrogram_cnn':
            self.score(np.zeros(self.config.window_samples, dtype=np.float32))

    def batch_trace(self, audio, decision=None, batch_size=16):
        """Offline batching of independently causal contexts; no delivery claim."""
        from .stream import replay_audio
        config = self.config
        starts = list(range(0, max(0, len(audio) - config.window_samples + 1), config.hop_samples))
        if not starts and len(audio):
            starts = [0]
        elif starts and starts[-1] + config.window_samples < len(audio):
            starts.append(starts[-1] + config.hop_samples)
        scores, images = [], []
        for start in starts:
            wave = audio[start:start + config.window_samples]
            rms = np.sqrt(np.mean(wave.astype(np.float64) ** 2))
            if len(wave) < config.min_audio_seconds * config.sample_rate or rms < config.silence_rms:
                continue
            images.append(spectrogram(wave, config))
            if len(images) == batch_size:
                scores.extend(float(v) for v in self.score_images(np.stack(images)))
                images.clear()
        if images:
            scores.extend(float(v) for v in self.score_images(np.stack(images)))
        iterator = iter(scores)
        proxy = SimpleNamespace(config=config, metadata=self.metadata, model_id=self.model_id,
                                score=lambda wave: next(iterator))
        rows = replay_audio(proxy, audio, decision)
        if next(iterator, None) is not None:
            raise ValueError('Offline contexts disagree with streaming contexts')
        for row in rows:
            row['processing_mode'] = 'offline batched contexts; use stream command for paced causal delivery'
            row['compute_seconds'] = row['delivered_at_seconds'] = None
        return rows

    def gradcam(self, samples, target='drone', layer='features'):
        if target not in {'drone', 'non_drone'} or layer not in {'conv1', 'conv2', 'features'}:
            raise ValueError('Unknown explanation target or layer')
        tf = tensorflow()
        net = self.network()
        image = spectrogram(samples, self.config)
        probe = tf.keras.Model(net.inputs, [net.get_layer(layer).output, net.output])
        with tf.GradientTape() as tape:
            activations, logits = probe(tf.convert_to_tensor(image[None, ..., None]), training=False)
            objective = logits[0, 0] * (1 if target == 'drone' else -1)
        gradients = tape.gradient(objective, activations)
        if gradients is None:
            raise ValueError('No gradient path to selected layer')
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        coarse = tf.nn.relu(tf.reduce_sum(weights * activations, axis=-1))[0]
        maximum = float(tf.reduce_max(coarse))
        normalized = coarse / maximum if maximum > 1e-15 else tf.zeros_like(coarse)
        heatmap = tf.image.resize(normalized[..., None], image.shape, method='bilinear')[..., 0].numpy()
        return {'spectrogram': image, 'heatmap': heatmap, 'native_heatmap': normalized.numpy(),
                'coarse_shape': list(coarse.shape),
                'target': target, 'layer': layer, 'classifier_score': float(expit(logits.numpy()[0, 0])),
                'has_positive_attribution': maximum > 1e-15}


def load_spectral_model(path, metadata=None, require_frozen=False):
    path = Path(path)
    meta = metadata or read_json(path / 'model.json')
    if meta.get('schema_version') != 2 or meta.get('class_mapping') != {'0': 'non_drone', '1': 'drone'}:
        raise ValueError('Unknown spectral model schema/classes')
    if meta.get('family') not in {'spectrogram_cnn', 'rbf_svm'}:
        raise ValueError('Unknown spectral model family')
    if digest({k: v for k, v in meta.items() if k != 'model_id'}) != meta.get('model_id'):
        raise ValueError('Model metadata identity mismatch')
    if file_hash(path / 'parameters.npz') != meta.get('parameters_sha256'):
        raise ValueError('Model parameters hash mismatch')
    if require_frozen:
        decision = meta.get('decision', {})
        if meta.get('status') != 'frozen' or not 0 < decision.get('off_threshold', -1) <= decision.get('on_threshold', -1) <= 1:
            raise ValueError('Freeze valid decisions on development before inference')
        if any(not isinstance(decision.get(k), int) or decision[k] < 1 for k in ['on_hits', 'off_hits']):
            raise ValueError('Invalid confirmation counts')
    with np.load(path / 'parameters.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite model arrays')
    if meta['family'] == 'spectrogram_cnn':
        variant = meta.get('architecture', {}).get('variant', 'compact')
        shapes = cnn_weight_shapes(variant)
        expected_layers = [{k: v for k, v in layer.items() if k != 'name'} for layer in cnn_spec(variant)]
        declared_layers = meta.get('architecture', {}).get('convolutions')
        if declared_layers is not None and declared_layers != expected_layers:
            raise ValueError('Declared convolution layout disagrees with the architecture variant')
        if set(arrays) != {f'w{i}' for i in range(len(shapes))} or any(arrays[f'w{i}'].shape != shape for i, shape in enumerate(shapes)):
            raise ValueError('CNN parameter dimensions disagree with fixed architecture')
        if meta.get('weight_shapes') != [list(shape) for shape in shapes]:
            raise ValueError('CNN weight schema mismatch')
    else:
        config = SpectrogramConfig(**meta['features'])
        size = 2 * len(axes(config)[0])
        expected = {'mean', 'scale', 'support_vectors', 'dual_coef', 'intercept'}
        if set(arrays) != expected or arrays['mean'].shape != (size,) or arrays['scale'].shape != (size,):
            raise ValueError('SVM normalization dimension mismatch')
        if np.any(arrays['scale'] <= 0) or arrays['support_vectors'].ndim != 2 or arrays['support_vectors'].shape[1] != size:
            raise ValueError('Invalid SVM scaling/support dimensions')
        if arrays['dual_coef'].shape != (len(arrays['support_vectors']),) or arrays['intercept'].shape != (1,):
            raise ValueError('Invalid SVM coefficients')
        if not np.isfinite(meta.get('gamma', np.nan)) or meta['gamma'] <= 0:
            raise ValueError('Invalid RBF gamma')
    return SpectralModel(meta, arrays)
