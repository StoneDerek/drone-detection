#!/usr/bin/env python3
"""One bounded follow-up: let high-resolution CNN filters span harmonic gaps."""

import tarfile
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.spectral_training import train_cnn
from drone_detector.spectrogram import SpectrogramConfig


def main():
    root = Path('runs/spectral-lab-v1')
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    output = root / 'followup-models' / 'frequency-context-fft4096'
    if output.exists():
        print('Preserving the existing frequency-context model.', flush=True)
        return
    plan_path = root / 'frequency-context-plan.json'
    if not plan_path.exists():
        plan = {'code': revision(), 'manifest_sha256': file_hash(manifest),
            'scope': 'Additional DEVELOPMENT follow-up after the original six-arm sweep. No final data opened; assessed default unchanged.',
            'hypothesis': 'At 3.90625Hz/bin, the initial CNN frequency receptive field is only 25 bins, about 98Hz. It cannot jointly see many adjacent harmonics. Add a frequency-only 5x1 convolution to extend it to 153 bins, about 598Hz, without changing temporal receptive field.',
            'controlled_changes': 'One additional 16-filter 5x1 ReLU convolution, stride1; 3077 versus1781 parameters. Same high-resolution linear log-power input, cached train/development windows, optimizer, eight epochs, seed42, weak bag objective and checkpoint selection.',
            'calibration': 'Reuse the explicit additional development negative-recording-budget policy; do not invent a fresh test claim.',
            'engineering_choices': 'Layer width/kernel and eight-epoch budget are bounded pilot choices; not user requirements or a validated harmonic detector.'}
        write_json(plan_path, plan)
        with tarfile.open(root / 'frequency-context-source.tar.gz', 'w:gz') as archive:
            for file in sorted(Path('src').rglob('*.py')):
                archive.add(file, arcname=str(file))
            archive.add('scripts/experiment_frequency_context.py')
    plan = read_json(plan_path)
    with threadpool_limits(limits=1):
        model = train_cnn(manifest, root / 'cache' / 'linear-fft4096', SpectrogramConfig(),
                          output, plan, variant='frequency_context')
    print('Frequency-context model trained:', model.model_id, flush=True)


if __name__ == '__main__':
    main()
