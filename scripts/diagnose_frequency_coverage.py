#!/usr/bin/env python3
"""Check architectural input coverage without retraining or changing models."""

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', 'runs/matplotlib')

import numpy as np
from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import file_hash, write_json
from drone_detector.model import load_model
from drone_detector.spectrogram import spectrogram, tensorflow


def main():
    source = Path('runs/real-v2/corpus/audio/batear-20260415_192501.wav')
    audio, _ = decode(source)
    tf = tensorflow()
    result = {'scope': 'Structural dependency diagnostic, not acoustic accuracy evidence',
        'script_sha256': file_hash(__file__), 'audio_sha256': file_hash(source),
        'context_seconds': [0, 3],
        'method': 'Replace every Conv2D by a linear convolution with positive unit kernels and retain its kernel shape, channels, strides and SAME padding. Gradient of the summed outputs marks all possible paths from input pixels, independent of trained ReLU activity or zero weights. Then replace all structurally unobserved rows in a real spectrogram and score it with the unchanged trained network.',
        'models': []}
    for name in ['linear-fft4096', 'linear256-fft4096', 'frequency-context-fft4096']:
        model = load_model(Path('artifacts/spectral-lab-v1/models') / name, True)
        image = spectrogram(audio[:48000], model.config)
        inputs = tf.zeros([1, *image.shape, 1], tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            values = inputs
            for layer in model.network().layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    kernel = tf.ones([*layer.kernel_size, int(values.shape[-1]), layer.filters])
                    values = tf.nn.conv2d(values, kernel, strides=[1, *layer.strides, 1], padding='SAME')
            objective = tf.reduce_sum(values)
        dependency = tape.gradient(objective, inputs).numpy()[0, :, :, 0]
        unobserved = np.flatnonzero(np.all(dependency == 0, axis=1))
        changed = image.copy()
        changed[unobserved] = 1 - changed[unobserved]
        scores = model.score_images(np.stack([image, changed]))
        item = {'name': name, 'model_id': model.model_id, 'shape': list(image.shape),
            'unobserved_rows': unobserved.tolist(), 'unobserved_count': len(unobserved),
            'unobserved_fraction': len(unobserved) / image.shape[0],
            'original_score': float(scores[0]), 'changed_score': float(scores[1]),
            'max_pixel_change': float(np.max(abs(image - changed))),
            'absolute_score_change': float(abs(scores[0] - scores[1]))}
        assert item['unobserved_count'] > 0 and item['max_pixel_change'] > .9
        assert item['absolute_score_change'] < 1e-7
        result['models'].append(item)
        print({key: value for key, value in item.items() if key != 'unobserved_rows'}, flush=True)
    result['interpretation_limit'] = 'These are missing paths from spectrogram tensor rows, not a measured percentage of waveform information or a proven explanation of field errors. Filterbanks and spectral leakage share physical tone energy across neighboring bins. Later larger receptive fields cannot restore rows already skipped by earlier layers.'
    write_json('reports/cnn-diagnosis-v1/frequency-coverage.json', result)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
