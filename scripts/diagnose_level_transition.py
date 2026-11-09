#!/usr/bin/env python3
"""Real-clip gain controls: uniform attenuation versus an internal level step."""

import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', 'runs/matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from threadpoolctl import threadpool_limits

from drone_detector.audio import FeatureConfig, decode, normalize_context_gain
from drone_detector.common import file_hash, read_json, write_json
from drone_detector.model import load_model
from drone_detector.stream import replay_audio


def rms(wave):
    return float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=Path, default=Path('reports/spectral-lab-v1/catalog.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/level-transition-v1'))
    parser.add_argument('--models', nargs='*')
    args = parser.parse_args()
    source = Path('runs/real-v2/corpus/audio/svanstrom-DRONE_001.wav')
    wave, info = decode(source)
    catalog = read_json(args.catalog)
    models = {item['name']: load_model(args.catalog.parent / item['bundle'], True)
              for item in catalog['models'] if not args.models or item['name'] in args.models}
    result = {'scope': 'Controlled diagnostic changes to one existing real training recording, not training data or acoustic accuracy evidence',
        'recording': 'svanstrom-DRONE_001', 'original_sha256': file_hash(source),
        'canonical_samples': len(wave), 'original_info': info,
        'screenshot_identification': 'Compare original trace scores with user screenshot before attributing it to this case.',
        'models': {}}
    block = 320
    levels = np.asarray([rms(wave[i:i+block]) for i in range(0, len(wave)-block+1, block)])
    loglevels = 20 * np.log10(np.maximum(levels, 1e-12))
    candidates = np.arange(50, len(levels)-25)
    change = np.asarray([np.median(loglevels[i:i+12])-np.median(loglevels[i-12:i]) for i in candidates])
    transition = int(candidates[np.argmin(change)]) * block
    before = wave[max(0, transition-32000):transition]
    after = wave[transition:min(len(wave), transition+32000)]
    n = min(len(before), len(after))
    before, after = before[-n:], after[:n]
    gain = rms(before) / max(rms(after), 1e-20)
    result['measured_transition'] = {'seconds': transition/16000,
        'median_level_change_db': float(change.min()), 'matched_excerpt_samples': n,
        'before_rms': rms(before), 'after_rms': rms(after), 'gain_to_match_after_to_before_rms': gain,
        'method': 'Largest drop in median 20ms-block RMS, using adjacent240ms blocks and excluding edges. Descriptive, not an independently annotated event.'}
    frequency, pre = signal.welch(before, fs=16000, nperseg=4096)
    _, post = signal.welch(after, fs=16000, nperseg=4096)
    result['band_level_changes_db'] = {}
    for lo, hi in [(0, 100), (100, 500), (500, 1500), (1500, 4000), (4000, 8000)]:
        chosen = (frequency >= lo) & (frequency < hi)
        result['band_level_changes_db'][f'{lo}-{hi}Hz'] = float(10*np.log10(max(float(post[chosen].sum()),1e-30)/max(float(pre[chosen].sum()),1e-30)))
    # Same exact real waveform in every gain control; only its amplitude rule changes.
    base = wave[64000:112000].copy()
    controls = {'unchanged_4_to_7s': base, 'uniform_minus12dB': base * .25,
                'uniform_minus20dB': base * .1}
    envelope = np.ones(48000, dtype=np.float32)
    envelope[24000:] = .25
    ramp = 160  # 10ms taper avoids introducing a hard click as a second factor.
    envelope[24000-ramp//2:24000+ramp//2] = .25 + .75*(1+np.cos(np.linspace(0, np.pi, ramp)))/2
    controls['internal_minus12dB_step_at1_5s'] = base * envelope
    reference = FeatureConfig(representation='yamnet_rms')
    controls['reference_rms_base'] = normalize_context_gain(base, reference)
    controls['reference_rms_uniform_minus12dB'] = normalize_context_gain(base * .25, reference)
    controls['reference_rms_internal_step'] = normalize_context_gain(base * envelope, reference)
    controls['real_before_matched_duration'] = before
    controls['real_after_matched_duration'] = after
    controls['real_after_reference_rms_matched'] = (after * gain).astype(np.float32)
    result['controls'] = {key: {'available_samples': len(value), 'rms': rms(value),
        'zero_padding_samples': 48000-len(value)} for key, value in controls.items()}
    for name, model in models.items():
        trace = model.batch_trace(wave) if hasattr(model, 'batch_trace') else replay_audio(model, wave)
        values = {key: model.score(value) for key, value in controls.items()}
        result['models'][name] = {'model_id': model.model_id, 'controls': values,
            'trace': [{key: row[key] for key in ['window_start_seconds', 'timestamp_seconds', 'score', 'alarm', 'rms', 'padded']} for row in trace],
            'uniform_minus12dB_score_change': values['uniform_minus12dB']-values['unchanged_4_to_7s'],
            'internal_step_score_change': values['internal_minus12dB_step_at1_5s']-values['unchanged_4_to_7s']}
        print(name, values, flush=True)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'diagnosis.json', result)
    fig, panels = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    panels[0].plot(np.arange(len(levels))*.02, loglevels)
    panels[0].axvline(transition/16000, color='red', linestyle='--')
    panels[0].set(xlabel='Audio time (s)', ylabel='20ms RMS (dBFS)', title='Actual recording level; dashed line is largest measured drop')
    shown = ['linear256-fft4096', 'svm-linear-fft4096', 'yamnet-default', 'relative-logistic']
    shown += [name for name in models if name.startswith('coherent-')]
    for name in shown:
        if name not in models:
            continue
        rows = result['models'][name]['trace']
        panels[1].plot([r['timestamp_seconds'] for r in rows], [r['score'] for r in rows], marker='.', label=name)
    panels[1].axvline(transition/16000, color='red', linestyle='--')
    panels[1].set(xlabel='Window END time (s); each score uses preceding3s', ylabel='Classifier score', ylim=(-.02, 1.02))
    panels[1].legend()
    fig.savefig(out / 'recording-level-and-scores.png', dpi=140)
    plt.close(fig)
    print('Measured transition', result['measured_transition'], flush=True)
    print('Band level changes', result['band_level_changes_db'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
