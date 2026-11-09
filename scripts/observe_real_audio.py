#!/usr/bin/env python3
"""Specific real cases for visual inspection, not a new accuracy benchmark."""

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', 'runs/matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import file_hash, write_json
from drone_detector.data import load_manifest
from drone_detector.model import load_model
from drone_detector.spectrogram import axes, spectrogram


def main():
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    records = load_manifest(manifest, True)
    selected = [next(r for r in records if r['id'] == identifier) for identifier in [
        'svanstrom-DRONE_001', 'svanstrom-HELICOPTER_001', 'svanstrom-BACKGROUND_001',
        'batear-20260415_192501', 'batear-20230701_054200']]
    selected += [next(r for r in records if r['source'] == 'd3set-ground' and r['split'] == 'train')]
    selected += [next(r for r in records if r['source'] == 'd3set-noise' and r['split'] == 'train' and r['negative_kind'] == 'Insect')]
    out = Path('reports/real-observations-v1')
    local_audio = Path('runs/real-observations-v1')
    out.mkdir(parents=True, exist_ok=True)
    local_audio.mkdir(parents=True, exist_ok=True)
    public = Path('web/public/examples')
    public.mkdir(parents=True, exist_ok=True)
    models = {name: load_model(path, True) for name, path in {
        'cnn': 'artifacts/spectral-lab-v1/models/linear256-fft4096',
        'svm': 'artifacts/spectral-lab-v1/models/svm-linear-fft4096',
        'yamnet': 'artifacts/real-v2/candidate'}.items()}
    result = {'scope': 'Seven deliberately small qualitative case studies, NOT independent detector trials. Visual inspection does not establish audible drone timing.',
        'selection': 'First numbered Svanstrom drone/helicopter/background; previously used Batear192501; first Batear background; first D3 ground training file middle10s; first D3 Insect training file first10s. Selection precedes spectral inspection and does not use prediction quality.',
        'cases': []}
    overview, grid = plt.subplots(4, 2, figsize=(16, 15), constrained_layout=True)
    for number, row in enumerate(selected):
        assert row['split'] in {'train', 'development'}
        source = manifest.parent / row['path']
        assert file_hash(source) == row['sha256']
        entire, info = decode(source)
        start = max(0, (len(entire) - 160000) // 2) if row['source'] == 'd3set-ground' else 0
        wave = entire[start:start + 160000]
        sf.write(local_audio / f"{row['id']}.wav", wave, 16000, subtype='FLOAT')
        frequencies, times, z = signal.stft(wave, fs=16000, nperseg=4096, noverlap=4096-256,
                                           boundary=None, padded=False)
        db = 20 * np.log10(np.maximum(np.abs(z), 1e-15))
        relative = db - db.max()
        freq_mask = (frequencies >= 50) & (frequencies <= 3000)
        scores = {name: model.score(wave[:48000]) for name, model in models.items()}
        image = spectrogram(wave[:48000], models['cnn'].config)
        cam = models['cnn'].gradcam(wave[:48000], 'drone', 'conv1')
        figure, panels = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        for panel, values, limits, title in [
            (panels[0, 0], db, (-90, -15), 'Fixed STFT reference (dB; not calibrated SPL)'),
            (panels[0, 1], relative, (-60, 0), 'Peak-relative view (60 dB display range)')]:
            plot = panel.pcolormesh(times, frequencies[freq_mask], values[freq_mask],
                                   shading='auto', cmap='magma', vmin=limits[0], vmax=limits[1], rasterized=True)
            panel.set(xlabel='Time in excerpt (s)', ylabel='Frequency (Hz)', title=title)
            figure.colorbar(plot, ax=panel, label='dB')
        f, t, _ = axes(models['cnn'].config)
        panels[1, 0].pcolormesh(t, f, image, shading='auto', cmap='magma', vmin=0, vmax=1, rasterized=True)
        panels[1, 0].set(xlabel='Time in first context (s)', ylabel='Frequency (Hz)',
                         title='Actual CNN input: 256 bands × 172 frames', ylim=(50, 3000))
        panels[1, 1].pcolormesh(t, f, image, shading='auto', cmap='gray', vmin=0, vmax=1, rasterized=True)
        if cam['has_positive_attribution']:
            panels[1, 1].pcolormesh(t, f, cam['heatmap'], shading='auto', cmap='autumn', alpha=.6, vmin=0, vmax=1, rasterized=True)
        panels[1, 1].set(xlabel='Time in first context (s)', ylabel='Frequency (Hz)', ylim=(50, 3000),
            title='Drone-logit Grad-CAM conv1' + ('' if cam['has_positive_attribution'] else ' — zero positive map'))
        figure.suptitle(f"{row['id']} · publisher: {row['label']} · excerpt {start/16000:.2f}–{(start+len(wave))/16000:.2f}s\nFirst-context scores: CNN {scores['cnn']:.3f}, SVM {scores['svm']:.3f}, YAMNet {scores['yamnet']:.3f}", fontsize=12)
        figure.savefig(out / f"{row['id']}.png", dpi=125)
        plt.close(figure)
        panel = grid.flat[number]
        panel.pcolormesh(times, frequencies[freq_mask], relative[freq_mask], shading='auto', cmap='magma', vmin=-60, vmax=0, rasterized=True)
        panel.set(xlabel='Excerpt time (s)', ylabel='Hz', title=f"{row['id']} · {row['label']}")
        average_power = (np.abs(z) ** 2).mean(axis=1)
        peaks, _ = signal.find_peaks(10 * np.log10(average_power + 1e-30), prominence=6, distance=4)
        peaks = [int(p) for p in peaks if 50 <= frequencies[p] <= 3000]
        strongest = sorted(peaks, key=lambda p: average_power[p], reverse=True)[:12]
        case = {'id': row['id'], 'publisher_label': row['label'], 'source': row['source'],
            'split': row['split'], 'annotation_status': row['annotation_status'],
            'source_file_sha256': row['sha256'], 'excerpt_start_seconds': start/16000,
            'excerpt_duration_seconds': len(wave)/16000, 'original_info': info,
            'license': row['license'], 'provenance': row['source_recording'],
            'first_context_scores': scores, 'cnn_gradcam_drone_conv1_positive': cam['has_positive_attribution'],
            'strongest_mean_spectral_peaks_hz': [float(frequencies[p]) for p in strongest],
            'peak_rule': 'Mean power, 50–3000Hz, local prominence6dB and minimum4 FFT bins; descriptive only, not rotor identification',
            'figure': f"{row['id']}.png"}
        if row['license'] in {'CC0-1.0', 'MIT'}:
            sf.write(public / f"{row['id']}.wav", wave, 16000, subtype='PCM_16')
            case['public_example'] = f"examples/{row['id']}.wav"
        result['cases'].append(case)
        print('Observed case prepared', row['id'], scores, flush=True)
    grid.flat[-1].axis('off')
    overview.suptitle('Real recording examples · linear frequency, relative log magnitude\nPublisher classes are weak labels; source/year/microphone vary. No listening-based verification is claimed.', fontsize=14)
    overview.savefig(out / 'overview.png', dpi=135)
    plt.close(overview)
    write_json(out / 'cases.json', result)
    write_json(public / 'catalog.json', {'scope': result['scope'],
        'examples': [{k: c[k] for k in ['id', 'publisher_label', 'public_example', 'license', 'provenance', 'excerpt_start_seconds', 'excerpt_duration_seconds']}
                     for c in result['cases'] if 'public_example' in c]})


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
