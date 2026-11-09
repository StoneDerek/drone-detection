#!/usr/bin/env python3
"""Independent numerical frontend, stereo and cache checks on existing train/dev."""

from collections import Counter
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, write_json
from drone_detector.data import load_manifest
from drone_detector.spectrogram import SpectrogramConfig, axes, spectrogram


@lru_cache(maxsize=16)
def reference_bank(config):
    hz = np.arange(config.n_fft//2+1)*config.sample_rate/config.n_fft
    if config.axis == 'linear':
        return (hz >= config.f_min) & (hz <= config.f_max)
    if config.axis == 'linear_filterbank':
        edges = np.linspace(config.f_min, config.f_max, config.mel_bins+2)
    else:
        low, high = 2595*np.log10(1+config.f_min/700), 2595*np.log10(1+config.f_max/700)
        edges = 700*(10**(np.linspace(low, high, config.mel_bins+2)/2595)-1)
    weights = []
    for low, center, high in zip(edges[:-2], edges[1:-1], edges[2:], strict=True):
        row = np.where(hz < center, (hz-low)/(center-low), (high-hz)/(high-center))
        row = np.maximum(0, row)
        weights.append(row/max(row.sum(), 1e-12))
    return csr_matrix(np.stack(weights))


def independent_image(wave, config):
    """Direct float64 FFT, independently built windows/bands; no production STFT."""
    wave = np.pad(np.asarray(wave, dtype=np.float64), (0, config.window_samples-len(wave)))
    frames = np.lib.stride_tricks.sliding_window_view(wave, config.n_fft)[::config.stft_hop]
    window = np.hanning(config.n_fft+1)[:-1]
    fft = np.fft.rfft(frames*window, axis=1)/window.sum()
    power = (fft.real**2+fft.imag**2).T
    bank = reference_bank(config)
    power = power[bank] if config.axis == 'linear' else bank @ power
    if power.max() <= 1e-30:
        return np.zeros(power.shape, dtype=np.float32)
    relative = power/power.max()
    if config.intensity == 'log_power':
        relative = 1+10*np.log10(np.maximum(relative, 1e-30))/config.dynamic_range_db
    return np.clip(relative, 0, 1).astype(np.float32)


def main():
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    cache = Path('runs/spectral-lab-v1/cache/linear256-fft4096')
    index = read_json(cache/'index.json')
    rows = {row['id']: row for row in load_manifest(manifest, True) if row['split'] in {'train', 'development'}}
    assert len(index['records']) == len(rows) == 615
    config = SpectrogramConfig(**index['identity']['features'])
    output, contexts = [], 0
    for number, cached in enumerate(index['records']):
        row = rows[cached['id']]
        source = manifest.parent/row['path']
        assert file_hash(source) == row['sha256']
        info = sf.info(source)
        original, rate = sf.read(source, dtype='float64', always_2d=True)
        channel_energy = np.einsum('ij,ij->j', original, original)/len(original)
        mono64 = original.mean(axis=1)
        mono_energy = float(np.mean(mono64**2))
        expected_ratio = mono_energy/max(float(channel_energy.mean()), 1e-30)
        canonical, _ = decode(source)
        reference_wave = mono64
        if rate != 16000:
            divisor = gcd(rate, 16000)
            reference_wave = signal.resample_poly(reference_wave, 16000//divisor, rate//divisor)
        decode_error = float(np.max(abs(canonical-reference_wave)))
        assert len(canonical) == len(reference_wave) == int(np.ceil(len(original)*16000/rate))
        assert decode_error < 2e-6
        assert file_hash(cache/cached['cache']) == cached['cache_sha256']
        images = np.load(cache/cached['cache'], mmap_mode='r', allow_pickle=False)
        assert images.shape == (len(cached['starts']), 256, 172)
        cache_error = independent_error = 0.
        for i, start in enumerate(cached['starts']):
            wave = canonical[start:start+48000]
            cache_error = max(cache_error, float(np.max(abs(spectrogram(wave, config)-images[i]))))
            if i in {0, len(images)//2, len(images)-1}:
                independent_error = max(independent_error, float(np.max(abs(independent_image(wave, config)-images[i]))))
        assert cache_error == 0.
        assert independent_error < 2e-3
        output.append({'id': row['id'], 'source': row['source'], 'split': row['split'], 'label': row['label'],
            'channels': info.channels, 'source_rate': rate, 'subtype': info.subtype,
            'canonical_samples': len(canonical), 'source_frames': len(original),
            'channel_rms': np.sqrt(channel_energy).tolist(),
            'mono_power_to_mean_channel_power': expected_ratio,
            'channel_max_difference': float(np.max(abs(original[:,0]-original[:,-1]))) if info.channels > 1 else None,
            'decode_float64_reference_max_error': decode_error,
            'cache_max_error': cache_error, 'independent_fft_max_error': independent_error,
            'contexts': len(images), 'waveform_mean': float(canonical.mean()),
            'zero_fraction': float(np.mean(canonical == 0)),
            'cached_floor_fraction': float(np.mean(images == 0)),
            'cached_ceiling_fraction': float(np.mean(images == 1))})
        contexts += len(images)
        if (number+1) % 100 == 0:
            print('Frontend audited', number+1, flush=True)
    # Known tones are software probes, never training/performance examples.
    t = np.arange(48000)/16000
    tone = (.1*np.sin(2*np.pi*1000*t)).astype(np.float32)
    probes = []
    for axis in ['linear', 'linear_filterbank', 'mel']:
        for fft in [1024, 4096]:
            c = SpectrogramConfig(axis=axis, n_fft=fft, mel_bins=256)
            image, reference = spectrogram(tone, c), independent_image(tone, c)
            frequencies, times, _ = axes(c)
            peak = float(frequencies[np.argmax(image.mean(axis=1))])
            assert abs(peak-1000) < 60
            assert image.shape == (len(frequencies), len(times))
            assert float(np.max(abs(reference-image))) < 2e-3
            probes.append({'axis': axis, 'fft': fft, 'shape': list(image.shape), 'peak_frequency_hz': peak,
                           'independent_max_error': float(np.max(abs(reference-image)))})
    result = {'passed': True, 'scope': '615 existing TRAIN/DEVELOPMENT files only; independent float64 DSP controls, no acoustic accuracy claim',
        'manifest_sha256': file_hash(manifest), 'contexts_reextracted': contexts,
        'formats': dict(Counter(f"{r['channels']}ch/{r['source_rate']}Hz/{r['subtype']}" for r in output)),
        'maximum_cache_error': max(row['cache_max_error'] for row in output),
        'maximum_independent_fft_error': max(row['independent_fft_max_error'] for row in output),
        'stereo_records': sum(row['channels'] > 1 for row in output),
        'minimum_stereo_power_ratio': min(row['mono_power_to_mean_channel_power'] for row in output if row['channels'] > 1),
        'known_tone_software_probes': probes, 'recordings': output}
    write_json('reports/training-audit-v1/frontend.json', result)
    print({k:v for k,v in result.items() if k not in {'recordings','known_tone_software_probes'}}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
