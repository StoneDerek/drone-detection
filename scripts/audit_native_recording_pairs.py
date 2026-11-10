#!/usr/bin/env python3
"""Inspect the shortest point/array pair in each archive, without model scores.

This is deterministic data-quality sampling. Reports describe these complete
members and the fixed middle eight seconds, not the full corpus's quality.
"""

import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ddl_audio import RangeReader


def native_facts(payload):
    with sf.SoundFile(io.BytesIO(payload)) as stream:
        subtype = stream.subtype
        rate = stream.samplerate
        samples = stream.read(dtype='float64', always_2d=True)
    if len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError('Empty/nonfinite native audio; do not silently normalize it')
    return {'frames': len(samples), 'channels': samples.shape[1], 'sample_rate': rate,
            'subtype': subtype, 'duration_seconds': len(samples) / rate,
            'peak': float(np.max(np.abs(samples))),
            'rms': float(np.sqrt(np.mean(samples ** 2))),
            'zero_fraction': float(np.mean(samples == 0)),
            'fraction_abs_above_one': float(np.mean(np.abs(samples) > 1)),
            'nonfinite_samples': 0}, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headers', type=Path, nargs='+', required=True)
    parser.add_argument('--remote-config', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    headers = [json.loads(line) for path in args.headers for line in path.read_text().splitlines()]
    urls = json.loads(args.remote_config.read_text())
    by_archive = {}
    for row in headers:
        by_archive.setdefault(row['archive'], []).append(row)
    pairs = []
    for name, members in sorted(by_archive.items()):
        point = min((row for row in members if row['view'] == 'Mic1'), key=lambda row: (row['bytes'], row['path']))
        array = next(row for row in members if row['measurement_key'] == point['measurement_key'] and row['view'] == 'Arr1')
        receipt = json.loads((args.receipts / (name + '.complete.json')).read_text())
        reader = RangeReader(urls[name], receipt['bytes'], block_size=4 * 1024 * 1024,
                             cache_dir=args.cache / (name + '.quality-blocks-4194304'))
        values = []
        with zipfile.ZipFile(reader) as archive:
            for member in [point, array]:
                info = archive.getinfo(member['path'])
                if info.file_size != member['bytes'] or info.CRC != member['crc32'] or info.file_size > 32 * 1024 * 1024:
                    raise ValueError('Unexpected selected member extent/integrity metadata')
                payload = archive.read(info)  # zipfile verifies the full member CRC.
                facts, samples = native_facts(payload)
                stem = Path(member['path']).stem
                (args.output / (stem + '.wav')).write_bytes(payload)
                facts.update(path=member['path'], archive=name, view=member['view'],
                             input_kind=member['input_kind'], sha256=hashlib.sha256(payload).hexdigest(),
                             crc32=info.CRC, whole_member_crc_verified=True)
                values.append((facts, samples))
                print('Full member checked:', member['path'], facts['duration_seconds'], 'seconds', flush=True)
        point_facts, array_facts = values[0][0], values[1][0]
        if point_facts['sample_rate'] != array_facts['sample_rate'] or point_facts['frames'] != array_facts['frames']:
            raise ValueError('Pair alignment must be reviewed; do not silently resize it')
        # Fixed display interval chosen without scores/listening: middle eight s.
        rate = point_facts['sample_rate']
        frames = min(8 * rate, point_facts['frames'])
        start = (point_facts['frames'] - frames) // 2
        plot_pair(values, start, frames, args.output / (point['measurement_key'] + '.png'))
        pairs.append({'measurement_key': point['measurement_key'],
                      'selection_rule': 'Smallest complete point-microphone file in each archive, then matched Arr1; no scores or semantic selection',
                      'archive_sha256_reference': receipt['sha256'],
                      'display_start_seconds': start / rate, 'display_duration_seconds': frames / rate,
                      'native_members': [point_facts, array_facts],
                      'point_to_array_rms_ratio': point_facts['rms'] / array_facts['rms'] if array_facts['rms'] else None})
    report = {'scope': 'Two complete native pairs only; no resampling/gain/mean/clipping or class scores. No semantic listening or audible-label verification.',
              'amplitude_reference': 'Numeric sample units; calibration across recorders not established. Above-one counts do not establish ADC clipping.',
              'pairs': pairs}
    (args.output / 'native-pairs.json').write_text(json.dumps(report, indent=2) + '\n')


def plot_pair(values, start, frames, output):
    cache = Path(tempfile.gettempdir()) / 'drone-native-plot-cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import signal

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for column, (facts, samples) in enumerate(values):
        rate = facts['sample_rate']
        clip = samples[start:start + frames, 0]
        times = np.arange(len(clip)) / rate + start / rate
        axes[0, column].plot(times, clip, linewidth=.4)
        axes[0, column].set(title=facts['path'], ylabel='Native numeric amplitude', xlabel='Recording time (s)')
        frequencies, offsets, spectrum = signal.spectrogram(clip, rate, window='hann',
            nperseg=8192, noverlap=8192 - 256, detrend=False, scaling='density', mode='psd')
        selected = frequencies <= 8000
        image = axes[1, column].pcolormesh(offsets + start / rate, frequencies[selected],
            10 * np.log10(np.maximum(spectrum[selected], 1e-20)), vmin=-140, vmax=-30,
            shading='auto', cmap='magma')
        axes[1, column].set(ylabel='Frequency (Hz)', xlabel='Recording time (s)')
    amplitude = max(float(np.max(np.abs(samples[start:start + frames]))) for _, samples in values)
    for axis in axes[0]:
        axis.set_ylim(-amplitude * 1.05, amplitude * 1.05)
    fig.colorbar(image, ax=list(axes[1]), label='PSD (dB re 1 numeric unit²/Hz)')
    fig.suptitle('Point microphone vs 24-microphone array mean; fixed middle 8 s\n8192-point FFT, 256-sample hop; shared numeric scales, calibration unknown', fontsize=12)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
