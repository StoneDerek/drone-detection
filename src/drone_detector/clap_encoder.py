"""Frozen LAION CLAP audio embeddings with pinned, offline-only assets.

The caller supplies mono 16 kHz canonical contexts. This adapter explicitly
resamples to CLAP's native 48 kHz; upsampling restores no discarded high-frequency
information. The unfused checkpoint's deterministic short-input repeatpad policy
is retained. Inputs longer than 10 seconds are rejected to avoid random cropping.
"""

import importlib.metadata
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .common import file_hash, read_json

MODEL_NAME = 'laion/clap-htsat-unfused'
MODEL_REVISION = '8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a'
TRANSFORMERS_VERSION = '4.57.3'
ASSETS = {
    'config.json': (5390, '9efb9557bc804f2ca6e394486af2e45dfed0b18554909735a99c6220b84e4288'),
    'preprocessor_config.json': (541, '9739f58296aa6f9ac18008fd0150fb2649bc554985fbde86d0a4041c882ac753'),
    'README.md': (4454, '3542c5c4ce25e7868f2c9c97c16229d876bd8d02049e1e0f4d7675da0e5b5db6'),
    'pytorch_model.bin': (614525833, '1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb'),
}


def asset_url(name):
    if name not in ASSETS:
        raise ValueError('Unknown CLAP asset')
    return f'https://huggingface.co/{MODEL_NAME}/resolve/{MODEL_REVISION}/{name}'


def verify_assets(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'assets.json')
    if (manifest.get('schema_version') != 1 or manifest.get('model') != MODEL_NAME
            or manifest.get('revision') != MODEL_REVISION):
        raise ValueError('Unknown CLAP asset manifest')
    expected = {name: value[1] for name, value in ASSETS.items()}
    if manifest.get('sha256') != expected:
        raise ValueError('CLAP asset manifest differs from pinned identities')
    for name, (size, checksum) in ASSETS.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size or file_hash(path) != checksum:
            raise ValueError(f'CLAP asset missing or changed: {name}')
    return manifest


class CLAPEncoder:
    """L2-normalized 512-D projected audio embeddings, no text-model inference."""

    sample_rate = 16000
    native_sample_rate = 48000
    feature_dim = 512
    minimum_samples = 1
    maximum_samples = 160000

    def __init__(self, directory, device='cpu'):
        self.directory = Path(directory).resolve()
        verify_assets(self.directory)
        self.device = str(device)
        if self.device not in {'cpu', 'cuda'} and not (
                self.device.startswith('cuda:') and self.device[5:].isdigit()):
            raise ValueError('CLAP device must be cpu, cuda or cuda:<index>')
        self._model = self._extractor = None
        packages = {}
        for package in ['torch', 'transformers', 'numpy', 'scipy']:
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = 'not-installed'
        self.identity = {
            'family': 'clap-frozen', 'model': MODEL_NAME, 'revision': MODEL_REVISION,
            'adapter_sha256': file_hash(__file__),
            'asset_sha256': {name: value[1] for name, value in ASSETS.items()},
            'code': f'HuggingFace transformers=={TRANSFORMERS_VERSION}; no remote model code',
            'frontend': {
                'canonical_sample_rate': self.sample_rate,
                'native_sample_rate': self.native_sample_rate,
                'resampling': 'scipy.signal.resample_poly(up=3,down=1), amplitudes unchanged',
                'resampling_limit': '16kHz canonical input has no information above 8 kHz; upsampling restores none',
                'configuration_sha256': ASSETS['preprocessor_config.json'][1],
                'padding': 'released repeatpad to 480000 samples',
                'truncation': 'rand_trunc setting; inputs longer than 10s rejected before extractor so random branch never executes',
                'fusion': False, 'gain_normalization': 'none',
            },
            'pooling': 'released audio encoder pooler, projection, then L2 normalization',
            'feature_dim': self.feature_dim, 'precision': 'float32',
            'device': self.device, 'packages': packages,
            'pretraining_overlap': 'Public-source or AudioSet pretraining overlap not ruled out',
        }

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            import transformers
            from transformers import ClapAudioConfig, ClapAudioModelWithProjection, ClapFeatureExtractor
        except ImportError as error:
            raise RuntimeError('CLAP requires the separate torch/transformers runtime') from error
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(f'CLAP requires transformers=={TRANSFORMERS_VERSION}')
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested for CLAP but unavailable')
        verify_assets(self.directory)
        configuration = read_json(self.directory / 'config.json')['audio_config']
        if configuration.get('enable_fusion') or configuration.get('projection_dim') != self.feature_dim:
            raise ValueError('Expected unfused 512-D CLAP audio configuration')
        config = ClapAudioConfig.from_dict(configuration)
        with torch.random.fork_rng(devices=[]):
            model = ClapAudioModelWithProjection(config)
        weights = torch.load(self.directory / 'pytorch_model.bin', map_location='cpu', weights_only=True)
        if not isinstance(weights, dict):
            raise ValueError('Expected CLAP state dictionary')
        audio_weights = {key: value for key, value in weights.items()
                         if key.startswith(('audio_model.', 'audio_projection.'))}
        model.load_state_dict(audio_weights, strict=True)
        self._extractor = ClapFeatureExtractor.from_dict(
            read_json(self.directory / 'preprocessor_config.json'))
        self._model = model.eval().requires_grad_(False).to(self.device)
        return self._model

    def features(self, audio):
        audio = np.asarray(audio)
        if audio.ndim != 1 or not self.minimum_samples <= len(audio) <= self.maximum_samples:
            raise ValueError('CLAP requires nonempty mono 16 kHz input no longer than 10 seconds')
        if not np.isfinite(audio).all():
            raise ValueError('CLAP audio contains non-finite values')
        model = self._load()
        import torch
        wave = resample_poly(np.asarray(audio, dtype=np.float32), 3, 1)
        inputs = self._extractor(wave, sampling_rate=self.native_sample_rate,
                                 truncation='rand_trunc', padding='repeatpad',
                                 max_length=480000, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs).audio_embeds
            output = torch.nn.functional.normalize(output, dim=-1)
            vector = output[0].cpu().numpy().astype(np.float64)
        if vector.shape != (self.feature_dim,) or not np.isfinite(vector).all():
            raise ValueError('Invalid CLAP embedding output')
        return vector
