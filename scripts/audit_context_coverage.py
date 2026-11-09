#!/usr/bin/env python3
"""Quantify existing cache/replay coverage and padded-tail alarm effects."""

import gzip
from pathlib import Path

import numpy as np

from drone_detector.common import read_json, write_json
from drone_detector.data import checked_audio, load_manifest
from drone_detector.model import load_model
from drone_detector.spectrogram import SpectrogramConfig, spectrogram
from drone_detector.weak import recording_results, weak_metrics


def main():
    records = [row for row in load_manifest('runs/real-v2/corpus/manifest-v2.jsonl',True) if row['split']=='development']
    uniform = {row['id']:row for row in read_json('runs/spectral-lab-v1/cache/linear256-fft4096/index.json')['records'] if row['split']=='development'}
    dense = read_json('runs/cnn-foundations-v1/dense-development/index.json')
    config = SpectrogramConfig(**dense['identity']['features'])
    by_id = {row['id']:row for row in records}
    verified_contexts = 0
    coverage = []
    for row in dense['records']:
        audio = checked_audio(by_id[row['id']],'runs/real-v2/corpus/manifest-v2.jsonl')
        trace = row['template']
        images = np.load(Path('runs/cnn-foundations-v1/dense-development')/row['cache'],mmap_mode='r',allow_pickle=False)
        assert len(images) == len(row['positions'])
        for index,position in enumerate(row['positions']):
            item = trace[position]
            start = round(item['window_start_seconds']*16000)
            wave = audio[start:start+48000]
            assert item['timestamp_seconds'] == (start+len(wave))/16000
            assert item['padded'] == (len(wave)<48000)
            fresh = spectrogram(wave,config)
            assert np.isfinite(images[index]).all() and np.array_equal(fresh,images[index])
            verified_contexts += 1
        full = [item for item in trace if item['reason']=='scored' and not item['padded']]
        tail = [item for item in trace if item['padded']]
        last_end = max((item['timestamp_seconds'] for item in full),default=0.)
        coverage.append({'id':row['id'],'label':uniform[row['id']]['label'],
            'source':uniform[row['id']]['source'],'uniform_contexts':len(uniform[row['id']]['starts']),
            'dense_full_scored_contexts':len(full),'padded_contexts':len(tail),
            'tail_new_audio_seconds':tail[0]['timestamp_seconds']-last_end if tail else None})
    names = ['coherent-global-seed42','coherent-frequency-seed42','coherent-frequency-seed7']
    outcomes = {}
    for name in names:
        model = load_model(Path('artifacts/cnn-foundations-v1/frozen')/name,True)
        with gzip.open(Path('runs/cnn-foundations-v1/traces')/(name+'.jsonl.gz'),'rt') as stream:
            traces = {row['recording_id']:row['trace'] for row in (read_json_line(line) for line in stream)}
        no_tails = {key:[item for item in trace if not item['padded']] for key,trace in traces.items()}
        original = recording_results(records,traces,model.metadata['decision'])
        truncated = recording_results(records,no_tails,model.metadata['decision'])
        changes = [{'id':a['id'],'label':a['label'],'with_tail_alarm':a['predicted'],'without_tail_alarm':b['predicted']}
                   for a,b in zip(original,truncated,strict=True) if a['predicted']!=b['predicted']]
        assert weak_metrics(original)['confusion'] == model.metadata['selection']['chosen']['metrics']['confusion']
        outcomes[name] = {'model_id':model.model_id,'with_tails':weak_metrics(original),
            'without_tails_fixed_original_decision':weak_metrics(truncated),'changed_recordings':changes}
    write_json('reports/training-audit-v1/context-coverage.json',{
        'passed':True,'scope':'Existing DEVELOPMENT traces only; fixed weights/decisions, no new assessment or threshold selection',
        'uniform_contexts':sum(row['uniform_contexts'] for row in coverage),
        'dense_full_contexts':sum(row['dense_full_scored_contexts'] for row in coverage),
        'dense_scored_contexts_freshly_verified':verified_contexts,
        'dense_tensor_max_error':0.,'all_dense_scored_timestamps_and_padding_verified':True,
        'padded_files':sum(row['padded_contexts']>0 for row in coverage),
        'files_with_less_than_1ms_new_audio_in_tail':sum(row['tail_new_audio_seconds'] is not None and row['tail_new_audio_seconds']<.001 for row in coverage),
        'recordings':coverage,'model_outcomes':outcomes,
        'interpretation':'Training/validation use at most60 uniform full windows; product replay uses all dense windows and may append a padded tail. This is a disclosed distribution mismatch, not an off-by-one feature/label indexing error. Tail-removal comparisons retain the original calibrated decision and are diagnostic.'})
    print({name:row['changed_recordings'] for name,row in outcomes.items()},flush=True)


def read_json_line(line):
    import json
    return json.loads(line)


if __name__ == '__main__':
    main()
