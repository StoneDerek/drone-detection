#!/usr/bin/env python3
"""Check frequency-selective stereo cancellation on existing real train/dev."""

from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf
from threadpoolctl import threadpool_limits

from drone_detector.common import read_json, write_json
from drone_detector.data import load_manifest


def main():
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    starts = {row['id']:row['starts'] for row in read_json('runs/spectral-lab-v1/cache/linear256-fft4096/index.json')['records']}
    rows = [row for row in load_manifest(manifest,True) if row['split'] in {'train','development'} and row['channels']==2]
    results = []
    for row in rows:
        raw, rate = sf.read(manifest.parent/row['path'], dtype='float64',always_2d=True)
        if rate != 16000:
            divisor = gcd(rate,16000)
            raw = signal.resample_poly(raw,16000//divisor,rate//divisor,axis=0)
        s = starts[row['id']]
        windows = []
        for start in sorted({s[0],s[len(s)//2],s[-1]}):
            excerpt = raw[start:start+48000]
            f, channels = signal.welch(excerpt.T,16000,nperseg=4096)
            _, mono = signal.welch(excerpt.mean(axis=1),16000,nperseg=4096)
            reference = channels.mean(axis=0)
            bands = {}
            for low,high in [(50,200),(200,1000),(1000,3000),(3000,7600)]:
                chosen = (f>=low)&(f<high)
                denominator = float(reference[chosen].sum())
                ratio = float(mono[chosen].sum()/denominator) if denominator>1e-25 else None
                bands[f'{low}-{high}Hz'] = 10*np.log10(max(ratio,1e-30)) if ratio is not None else None
            chosen = (f>=50)&(f<=7600)
            notch = mono < .1*reference
            windows.append({'start_sample':start,'mono_power_to_mean_channel_power_db':bands,
                'energy_fraction_in_bins_attenuated_over10db':float(reference[chosen&notch].sum()/max(reference[chosen].sum(),1e-30))})
        results.append({'id':row['id'],'source':row['source'],'label':row['label'],'contexts':windows})
    write_json('reports/training-audit-v1/channel-mix.json',{
        'scope':'First/middle/last cached context in all101 stereo train/dev files; frequency-power diagnostic, no new labels or accuracy claim',
        'files':len(results),'contexts':sum(len(row['contexts']) for row in results),
        'worst_band_mean_channel_attenuation_db':min(value for row in results for context in row['contexts'] for value in context['mono_power_to_mean_channel_power_db'].values() if value is not None),
        'maximum_energy_fraction_in_deep_notches':max(context['energy_fraction_in_bins_attenuated_over10db'] for row in results for context in row['contexts']),
        'recordings':results,'limitation':'Power ratio uses mean single-channel power as reference, not the better microphone or a calibrated acoustic signal. Narrow spectral notches can exist even with acceptable whole-file RMS.'})
    print('Frequency-selective stereo audit complete',len(results),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
