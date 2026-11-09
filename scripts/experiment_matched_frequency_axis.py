#!/usr/bin/env python3
"""Match tensor dimensions to distinguish frequency warping from bin count."""

import tarfile
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.spectral_training import prepare_cache, train_cnn
from drone_detector.spectrogram import SpectrogramConfig


def main():
    root = Path('runs/spectral-lab-v1')
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    output = root / 'followup-models' / 'linear256-fft4096'
    if output.exists():
        print('Preserving matched-frequency-axis model.', flush=True)
        return
    config = SpectrogramConfig(axis='linear_filterbank', mel_bins=256)
    plan_path = root / 'matched-frequency-axis-plan.json'
    if not plan_path.exists():
        plan = {'code': revision(), 'manifest_sha256': file_hash(manifest),
            'scope': 'Additional DEVELOPMENT control; same data, no final assessment opened, default unchanged.',
            'hypothesis': 'The full linear FFT has many more bins than the mel input. Match both inputs to 256x172 to test linear versus mel band placement independently of tensor dimensions.',
            'controlled_changes': '256 area-normalized triangular filters spaced uniformly in Hz versus uniformly in mel. Same 4096-sample Hann window, hop256, 50-7600Hz band, relative log power, 80dB floor, 1781-parameter CNN, seed42, Adam, eight epochs, bag sampling/objective/checkpoint policy.',
            'calibration': 'The same additional 5% negative-recording-budget development calibration; no new generalization claim.',
            'interpretation': 'Band placement changes local physical receptive fields. Tensor dimensions, initialization, model capacity and amplitude rule match the mel256 control.'}
        write_json(plan_path, plan)
        with tarfile.open(root / 'matched-frequency-axis-source.tar.gz', 'w:gz') as archive:
            for file in sorted(Path('src').rglob('*.py')):
                archive.add(file, arcname=str(file))
            archive.add('scripts/experiment_matched_frequency_axis.py')
    plan = read_json(plan_path)
    with threadpool_limits(limits=1):
        cache = root / 'cache' / 'linear256-fft4096'
        prepare_cache(manifest, config, cache)
        model = train_cnn(manifest, cache, config, output, plan)
    print('Matched-axis model trained:', model.model_id, flush=True)


if __name__ == '__main__':
    main()
