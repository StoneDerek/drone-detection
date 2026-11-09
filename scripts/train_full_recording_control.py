#!/usr/bin/env python3
"""One declared same-input control for the sampled-recording target discrepancy."""

import importlib.util
import tarfile
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.foundation_training import train_foundation_cnn
from drone_detector.spectrogram import SpectrogramConfig, tensorflow


def main():
    root, report = Path('runs/training-audit-v1'), Path('reports/training-audit-v1')
    assert read_json(report/'gradients-and-checkpoints.json')['passed']
    plan_path = report/'full-recording-plan.json'
    if plan_path.exists():
        raise ValueError('Preserve this declared control; use a new run identity for changes')
    plan = {'code':revision(),'manifest_sha256':file_hash('runs/real-v2/corpus/manifest-v2.jsonl'),
        'audit_plan_sha256':file_hash(report/'plan.json'),
        'sampling_diagnosis_sha256':file_hash(report/'sampled-objective.json'),
        'gradient_oracle_sha256':file_hash(report/'gradients-and-checkpoints.json'),
        'name':'full-recording-frequency-seed42','variant':'frequency_aware','seed':42,
        'epochs':24,'snapshot_epochs':[8],
        'unchanged':'Same405 train/210 development files, cached256x172 frontend,154689-parameter CNN, same initial weights,51 balanced4+4 recording batches/epoch, Adam.001 per-tensor clipnorm1,L2.0001,dropout0.',
        'change':'At every update, scan every cached context in each selected recording (maximum60), select all exact tied maxima with inference mode, then recompute only those contexts with gradients. With dropout0/no batch normalization, this equals the full cached-recording symmetric max-BCE gradient. No sampled sub-bag inherits the full label.',
        'selection':'Minimum deterministic balanced full-cache max-BCE on DEVELOPMENT every2 epochs. Preserve best-at8, best-at24 and last24.',
        'limits':'The fixed cache still caps60 uniform contexts/recording; it is not every dense product context. Labels are still weak. One seed, no fresh assessment or default promotion.'}
    write_json(plan_path,plan)
    with tarfile.open(root/'full-recording-implementation.tar.gz','w:gz') as archive:
        for p in [*sorted(Path('src').rglob('*.py')),Path(__file__),Path('scripts/evaluate_cnn_foundations.py')]:
            archive.add(p,arcname=str(p))
    tf = tensorflow()
    tf.config.experimental.enable_op_determinism()
    name = plan['name']
    model = train_foundation_cnn('runs/real-v2/corpus/manifest-v2.jsonl',
        'runs/spectral-lab-v1/cache/linear256-fft4096',SpectrogramConfig(axis='linear_filterbank',mel_bins=256),
        root/'trained'/name,plan,variant='frequency_aware',balanced_batches=True,
        consistent_loss=True,dropout_rate=0.,full_recording_bags=True)
    write_json(report/(name+'-training.json'),{'model_id':model.model_id,'architecture':model.metadata['architecture'],
        'training':{k:v for k,v in model.metadata['training'].items() if k!='records'}})
    spec = importlib.util.spec_from_file_location('foundation_evaluation','scripts/evaluate_cnn_foundations.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.REPORT = root, report
    index = module.prepare_dense()
    for suffix in ['', '-at8', '-last']:
        module.evaluate(name+suffix,index)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
