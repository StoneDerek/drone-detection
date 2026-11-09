#!/usr/bin/env python3
"""Test whether the training objective can reward class-independent noise."""

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', 'runs/matplotlib')

import numpy as np
from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, write_json
from drone_detector.model import load_model
from drone_detector.spectrogram import tensorflow


def main():
    root = Path('runs/cnn-foundations-v1')
    result = {'scope': 'Mathematical/software loss diagnostic plus frozen real-input checkpoint audit; not acoustic accuracy evidence',
        'script_sha256': file_hash(__file__), 'noise_witness': [], 'checkpoints': {}}
    softplus = lambda value: float(np.logaddexp(0, value))  # noqa: E731
    for n in [5, 8, 15, 60]:
        # One constant feature, dropout.1, bias1.5 and weight×feature=-2.25.
        # Kept:1.5-2.25/.9=-1; dropped:+1.5. Inputs/labels are unrelated.
        high = 1-.9**n
        positive = (1-high)*softplus(1)+high*softplus(-1.5)
        negative_mean = .9*softplus(-1)+.1*softplus(1.5)
        negative_max = (1-high)*softplus(-1)+high*softplus(1.5)
        result['noise_witness'].append({'contexts': n, 'same_identical_input_for_both_labels': True,
            'dropout': .1, 'logits_when_kept_or_dropped': [-1., 1.5],
            'old_asymmetric_balanced_loss': .5*(positive+negative_mean),
            'consistent_max_balanced_loss': .5*(positive+negative_max),
            'constant_uninformative_loss': softplus(0),
            'deterministic_inference_balanced_loss': .5*(softplus(.75)+softplus(-.75))})
    index = read_json('runs/spectral-lab-v1/cache/linear256-fft4096/index.json')
    records = [r for r in index['records'] if r['split'] == 'train']
    labels = np.asarray([r['label'] == 'drone' for r in records])
    tf = tensorflow()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    for name in ['compact8', 'medium-sampled-max-last', 'medium-full-smooth-last', 'medium-balanced-max-last']:
        model = load_model(root / 'trained' / name)
        net = model.network()
        probe = tf.keras.Model(net.inputs, net.get_layer('pool').output)
        forward = tf.function(lambda images: probe(images, training=False))
        bag_features = []
        for row in records:
            images = np.load(Path('runs/spectral-lab-v1/cache/linear256-fft4096') / row['cache'], mmap_mode='r', allow_pickle=False)[:8]
            bag_features.append(forward(np.asarray(images)[..., None]).numpy().reshape(len(images), -1))
        arrays = model.arrays

        def head(features):
            values = features @ arrays['w6'] + arrays['w7']
            if 'w8' in arrays:
                values = np.maximum(values, 0) @ arrays['w8'] + arrays['w9']
            return values.ravel()

        def losses(stochastic, rng):
            rows, symmetric = [], []
            for features, label in zip(bag_features, labels, strict=True):
                if stochastic:
                    features = features * (rng.random(features.shape) >= .1).astype(np.float32) / np.float32(.9)
                logits = head(features)
                rows.append(float(np.logaddexp(0, -logits.max()) if label else np.logaddexp(0, logits).mean()))
                symmetric.append(float(np.logaddexp(0, (-1 if label else 1)*logits.max())))
            rows, symmetric = np.asarray(rows), np.asarray(symmetric)
            return [(rows[labels].mean()+rows[~labels].mean())/2,
                    (symmetric[labels].mean()+symmetric[~labels].mean())/2]

        rng = np.random.default_rng(123)
        deterministic = losses(False, rng)
        monte_carlo = np.asarray([losses(True, rng) for _ in range(20)])
        # Replaying the head without dropout must agree with the actual model.
        first = records[0]
        images = np.load(Path('runs/spectral-lab-v1/cache/linear256-fft4096') / first['cache'], allow_pickle=False)[:8]
        logits = head(bag_features[0])
        expected = 1/(1+np.exp(-logits))
        error = float(np.max(abs(expected-model.score_images(images))))
        assert error < 1e-5
        audit = {'model_id': model.model_id, 'recordings': len(records), 'contexts_per_bag': 'same fixed first min8 contexts in both modes',
            'head_replay_score_error': error,
            'old_loss_inference': float(deterministic[0]),
            'old_loss_with_dropout_mean': float(monte_carlo[:, 0].mean()),
            'old_loss_with_dropout_std': float(monte_carlo[:, 0].std()),
            'consistent_max_loss_inference': float(deterministic[1]),
            'consistent_max_loss_dropout_mean': float(monte_carlo[:, 1].mean())}
        result['checkpoints'][name] = audit
        print(name, audit, flush=True)
    result['interpretation'] = 'The old positive-max/negative-mean objective has an input-independent noise loophole. Dropout supplies such noise inside the network. The fixed-checkpoint comparison isolates dropout mode on identical real inputs/weights, not optimization trajectory. This does not prove it is the only cause of field failures or explain the historical EfficientNet pipeline.'
    write_json('reports/cnn-foundations-v1/mil-objective-audit.json', result)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
