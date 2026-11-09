#!/usr/bin/env python3
"""Article-guided controlled ablation; development evidence, never a fresh test."""

from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import read_json, file_hash, revision, write_json
from drone_detector.model import load_model
from drone_detector.weak import train_weak, tune_weak


def main():
    root = Path('runs/context-gain-v1')
    root.mkdir(parents=True, exist_ok=True)
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    baseline = load_model('artifacts/real-v2/candidate', True)
    if file_hash(manifest) != baseline.metadata['training']['manifest_sha256']:
        raise ValueError('Ablation must use the identical versioned development protocol')
    plan = {'code': revision(), 'manifest_sha256': file_hash(manifest),
        'hypothesis': 'Available-context RMS gain control reduces microphone/loudness sensitivity of frozen audio embeddings',
        'control': 'Frozen YAMNet v1; identical MI head method, C1, seed42, train/development files, context/hop, threshold grid and sensitivity-v2 selection',
        'only_feature_change': 'Per-context RMS target .05, gain cap +20dB, peak cap .98, before zero padding; original silence guard unchanged',
        'engineering_choices': 'Gain target/caps and advisory thresholds are provisional choices, not claims from the article or user requirements',
        'reference': 'User-pasted Counter-UAS 101 – Acoustic Drone Detection (Jan 2026); secondary overview used for hypotheses, quantitative claims unverified',
        'evaluation_scope': 'DEVELOPMENT ONLY. Prior v2 assessment informed the failure hypotheses and is consumed for any future model optimization. No fresh assessment is scored here.',
        'promotion': 'The independently assessed v2 default is preserved; this model is an opt-in experiment regardless of development outcome'}
    if (root / 'plan.json').exists() and read_json(root / 'plan.json')['manifest_sha256'] != file_hash(manifest):
        raise ValueError('Do not overwrite an experiment with different data')
    if not (root / 'plan.json').exists():
        write_json(root / 'plan.json', plan)
    with threadpool_limits(limits=1):
        if not (root / 'trained').exists():
            train_weak(manifest, root / 'trained', 'yamnet_rms', seed=42, c=1,
                       encoder_source='runs/real-v1/encoders/yamnet')
        if not (root / 'frozen').exists():
            tune_weak(root / 'trained', manifest, root / 'frozen', policy='sensitivity-v2')
    candidate = load_model(root / 'frozen', True)
    report = {'scope': plan['evaluation_scope'], 'plan': plan,
        'baseline': {'model_id': baseline.model_id, 'development': baseline.metadata['selection']['chosen']},
        'candidate': {'model_id': candidate.model_id, 'development': candidate.metadata['selection']['chosen'],
                      'input_gain': candidate.metadata['input_gain'],
                      'training_seconds': candidate.metadata['training']['seconds'],
                      'development_seconds': candidate.metadata['selection']['runtime_seconds']},
        'quality_advisory_coverage': candidate.metadata['selection']['quality_advisory_coverage'],
        'default_changed': False, 'fresh_reliability_claim': False}
    write_json(root / 'results.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
