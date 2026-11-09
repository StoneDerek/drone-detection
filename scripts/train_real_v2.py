#!/usr/bin/env python3
"""Train and freeze v2 on development before any fresh assessment is opened."""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.data import load_manifest
from drone_detector.model import load_model
from drone_detector.weak import evaluate_weak, train_weak, tune_weak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assess', action='store_true', help='Run the one-time frozen final comparison')
    args = parser.parse_args()
    root = Path('runs/real-v2')
    manifest = root / 'corpus/manifest-v2.jsonl'
    load_manifest(manifest, True)
    plan_path = root / 'experiment-plan.json'
    if not plan_path.exists():
        write_json(plan_path, {
            'manifest_sha256': file_hash(manifest), 'code': revision(), 'seed': 42,
            'old_assessment_consumed': True,
            'hypotheses': {
                'relative': 'Broader real positives plus sensitivity calibration improve the cheap baseline',
                'yamnet': 'Frozen audio pretraining transfers across microphones better than spectral summaries'},
            'controlled': 'Same real data, MI learning, seed42, C1, context3s/hop0.5s, train-only scaling; representation differs',
            'calibration': 'sensitivity-v2: maximize development recall under <=5% negative-file FPR; prefer lowest feasible threshold, then FPR and two-hit confirmation',
            'candidate_selection': 'Feasible FPR<=5%, then highest recall, highest max-score ROC AUC, lowest FPR; ties prefer YAMNet. Development only.',
            'assessment': 'One frozen batch, both arms. Fresh field days and negative source IDs; no final-driven optimization.',
            'claims': 'Weak publisher recording labels. No independently verified temporal event metric or field-reliability claim.'})
    plan = read_json(plan_path)
    if plan['manifest_sha256'] != file_hash(manifest):
        raise ValueError('Corpus changed after the experiment plan')
    ledger, models = [], {}
    with threadpool_limits(limits=1):
        for name, representation in [('relative', 'relative_mel'), ('yamnet', 'yamnet')]:
            trained = root / 'models' / (name + '-trained')
            frozen = root / 'models' / name
            encoder = Path('runs/real-v1/encoders/yamnet') if name == 'yamnet' else None
            if not trained.exists():
                train_weak(manifest, trained, representation, encoder_source=encoder)
            if not frozen.exists():
                tune_weak(trained, manifest, frozen, policy='sensitivity-v2')
            model = load_model(frozen, True)
            if model.metadata['training']['manifest_sha256'] != file_hash(manifest):
                raise ValueError('Existing artifact uses a different corpus')
            models[name] = model
            metrics = model.metadata['selection']['chosen']['metrics']
            ledger.append({'arm': name, 'model_id': model.model_id,
                'training': model.metadata['training'], 'decision': model.metadata['decision'],
                'development': metrics, 'code': model.metadata['code']})
            write_json(root / 'experiment-ledger.json', ledger)
            print(name, metrics, flush=True)

        def rank(name):
            m = models[name].metadata['selection']['chosen']['metrics']
            fpr, recall = m['negative_recording_false_positive_fraction'], m['recording_recall']
            return (fpr > .05, -recall if fpr <= .05 else fpr,
                    -m['roc_auc_max_window_score'], fpr, name != 'yamnet')

        selected = min(models, key=rank)
        selection = {'selected': selected, 'manifest_sha256': file_hash(manifest),
            'models': {name: model.model_id for name, model in models.items()},
            'code': revision(), 'final_touched_at_selection': False,
            'selection_rule': plan['candidate_selection']}
        frozen_path = root / 'frozen-selection.json'
        if frozen_path.exists() and read_json(frozen_path)['models'] != selection['models']:
            raise ValueError('Refusing to replace the frozen candidate')
        if not frozen_path.exists():
            write_json(frozen_path, selection)
        demo_path = root / 'fixed-demo.json'
        if not demo_path.exists():
            final_rows = [r for r in load_manifest(manifest, True) if r['split'] == 'final']
            ids = []
            for source, label, subtype in [
                ('ddl-real', 'drone', 'MINI'), ('ddl-real', 'drone', 'PRO4'),
                ('batear-new-day', 'drone', 'mavic-pro'),
                ('batear-new-day', 'drone', 'mini-4-pro'),
                ('batear-new-day', 'non_drone', None),
                ('esc50-fresh-negatives', 'non_drone', 'engine'),
                ('esc50-fresh-negatives', 'non_drone', 'airplane'),
                ('esc50-fresh-negatives', 'non_drone', 'vacuum_cleaner'),
                ('esc50-fresh-negatives', 'non_drone', 'breathing')]:
                row = next((r for r in final_rows if r['source'] == source and r['label'] == label
                            and (subtype is None or r.get('drone_type') == subtype or r.get('negative_kind') == subtype)
                            and r['duration_seconds'] <= 600), None)
                if row:
                    ids.append(row['id'])
            write_json(demo_path, {'recording_ids': ids, 'manifest_sha256': file_hash(manifest),
                'reason': 'First matching source/class/type by manifest order, fixed before fresh assessment. Includes real positives, ordinary negatives and mechanical hard negatives.'})
        if args.assess:
            report = evaluate_weak([root / 'models/relative', root / 'models/yamnet'],
                                   manifest, 'final', root / 'final')
            report['selected'] = selected
            write_json(root / 'results.json', report)
            for arm in report['results']:
                print(arm['model_id'], arm['summary'], flush=True)
        else:
            print(f'Frozen candidate: {selected}. Fresh assessment remains unscored.', flush=True)


if __name__ == '__main__':
    main()
