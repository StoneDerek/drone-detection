#!/usr/bin/env python3
"""Python references from existing train/development audio, kept out of deploy."""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, write_json
from drone_detector.data import load_manifest
from drone_detector.model import load_model
from drone_detector.spectrogram import spectrogram
from drone_detector.stream import replay_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=Path, default=Path('reports/spectral-lab-v1/catalog.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/cloudflare-deployment/parity'))
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    rows = load_manifest(manifest, True)
    choices = [next(r for r in rows if r['id'] == 'batear-20260415_192501')]
    choices += [next(r for r in rows if r['source'] == 'svanstrom' and r['label'] == 'drone')]
    choices += [next(r for r in rows if r['source'] == 'esc10' and r['split'] == 'development')]
    catalog = read_json(args.catalog)
    models = {r['name']: load_model(args.catalog.parent / r['bundle'], True) for r in catalog['models']}
    result = {'scope': 'Numerical software parity on three existing real TRAIN/DEVELOPMENT files, not an acoustic benchmark',
              'cases': []}
    for number, row in enumerate(choices):
        assert row['split'] != 'final'
        source = manifest.parent / row['path']
        assert file_hash(source) == row['sha256']
        wave, info = decode(source)
        canonical = output / f'case-{number}.bin'
        canonical.write_bytes(wave.astype('<f4').tobytes())
        case = {'id': row['id'], 'source': str(source), 'sha256': row['sha256'], 'info': info,
                'canonical': canonical.name, 'models': {}, 'gradcam': {}}
        for name, model in models.items():
            if number == 0:
                trace = model.batch_trace(wave) if hasattr(model, 'batch_trace') else replay_audio(model, wave)
                expected = [{k: t[k] for k in ['window_start_seconds', 'timestamp_seconds', 'score', 'reason', 'alarm', 'padded']} for t in trace]
            else:
                expected = [{'window_start_seconds': 0., 'score': model.score(wave[:48000])}]
            case['models'][name] = {'model_id': model.model_id, 'expected': expected}
            if number == 0 and hasattr(model, 'score_images'):
                image = spectrogram(wave[:48000], model.config)
                file = output / f'{name}-image.bin'
                file.write_bytes(image.astype('<f4').tobytes())
                case['models'][name]['image'] = file.name
        if number == 0:
            cam_models = ['linear256-fft4096'] + [name for name in models
                if name.startswith('coherent-') or name in {'coverage8', 'medium-balanced-frequency-head'}]
            for name in cam_models:
                model = models[name]
                for target in ['drone', 'non_drone']:
                    for layer in ['conv1', 'conv2', 'features']:
                        values = model.gradcam(wave[:48000], target, layer)
                        file = output / f'cam-{name}-{target}-{layer}.bin'
                        file.write_bytes(values['heatmap'].astype('<f4').tobytes())
                        case['gradcam'][name+':'+target+'-'+layer] = {'heatmap': file.name,
                            'model': name, 'target': target, 'layer': layer,
                            'score': values['classifier_score'], 'positive': values['has_positive_attribution'],
                            'coarse_shape': values['coarse_shape']}
        result['cases'].append(case)
        print('Reference fixture', row['id'], info['original_sample_rate'], flush=True)
    write_json(output / 'expected.json', result)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
