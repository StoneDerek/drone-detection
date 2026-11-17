#!/usr/bin/env python3
"""Compare one fully completed adapted arm with its matched frozen control."""

import argparse
import datetime
import hashlib
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from drone_detector.adaptation import batches_by_length, cohort_signature, partition_summary
from drone_detector.common import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from drone_detector.optimization_audit import weighted_loss
from drone_detector.paired_comparison import source_class_pairs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_beats_adaptation import verified_result
from run_confuser_study import broad_cutoff
from run_corpus_study import logits


def matrix(path, rows):
    metadata = read_json(path.with_suffix('.json'))
    if (file_hash(path) != metadata['sha256']
            or metadata['cohort_signature'] != cohort_signature(rows)):
        raise ValueError('Feature matrix identity changed')
    with np.load(path, allow_pickle=False) as saved:
        if saved['ids'].tolist() != [r['id'] for r in rows]:
            raise ValueError('Feature matrix row order changed')
        x = saved['x'].astype(np.float64)
    if x.shape != (len(rows), 768) or not np.isfinite(x).all():
        raise ValueError('Incomplete or nonfinite feature matrix')
    return x, metadata


def joint_head(path, expected_sha):
    import torch

    torch.set_num_threads(1)
    if read_json(path.with_suffix(path.suffix + '.json'))['sha256'] != expected_sha:
        raise ValueError('Selected encoder receipt changed')
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
        if h.hexdigest() != expected_sha:
            raise ValueError('Selected encoder bytes changed')
        stream.seek(0)
        selected = torch.load(stream, map_location='cpu', weights_only=True)
    head = {key: selected[key].cpu().numpy().astype(np.float64) for key in ['mean', 'scale']}
    head['coef'] = selected['training_head']['weight'].cpu().numpy()[0].astype(np.float64)
    head['intercept'] = selected['training_head']['bias'].cpu().numpy().astype(np.float64)
    return head, selected['selection_loss'], selected['epoch']


def verify_confusions(cells, reference, family):
    for source in {r['source'] for r in cells}:
        observed = {key: sum(r[family + '_confusion'][key] for r in cells if r['source'] == source)
                    for key in ['tp', 'fn', 'fp', 'tn']}
        if observed != reference['validation']['by_source'][source]['confusion']:
            raise ValueError('Reconstructed decisions differ from the completed evaluation receipt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('work/beats-adaptation-cpu2-v1'))
    parser.add_argument('--fold', choices=['audrok-dresden', 'wsu-air3'], required=True)
    parser.add_argument('--positive-mix', choices=['current', 'broader'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private-output', type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.work / 'manifest.jsonl')
    plan = read_json(args.work / 'plan.json')
    plan_sha = file_hash(args.work / 'plan.json')
    if cohort_signature(rows) != plan['recipe']['cohort_signature']:
        raise ValueError('Study cohort changed')
    spec = plan['folds'][args.fold]
    train_ids = spec[args.positive_mix + '_positive_train']
    lookup = {r['id']: i for i, r in enumerate(rows)}
    indices = {key: [lookup[i] for i in ids] for key, ids in
               [('train', train_ids), ('selection', spec['selection']), ('validation', spec['validation'])]}
    training = [rows[i] for i in indices['train']]
    recipe = plan['recipe']['fine_tuning']
    expected_steps = len(batches_by_length(training, recipe['batch_size'], recipe['seed'], 0)) * recipe['epochs']
    results = {family: verified_result(args.work, args.fold, args.positive_mix, family,
                                     plan_sha, expected_steps, len(training), digest(train_ids), recipe)
               for family in ['frozen', 'fine-tuned']}
    if any(r is None for r in results.values()):
        raise ValueError('Both complete evaluation receipts are required; no partial-arm result')
    frozen, adapted = results['frozen'], results['fine-tuned']
    folder = args.work / args.fold / args.positive_mix
    x0, meta0 = matrix(args.work / 'base-embeddings.npz', rows)
    x1, meta1 = matrix(folder / 'fine-tuned/embeddings.npz', rows)
    if (meta0['encoder'] != frozen['model']['encoder']
            or meta1['adapted_parameters_sha256'] != adapted['model']['selected_encoder_sha256']):
        raise ValueError('Classifier and encoder matrix identities differ')
    heads = {}
    for family in ['frozen', 'fine-tuned']:
        with np.load(folder / family / 'head.npz', allow_pickle=False) as saved:
            heads[family] = {key: saved[key].copy() for key in saved.files}
    joint, joint_reference, selected_epoch = joint_head(
        folder / 'fine-tuned/best.pt', adapted['model']['selected_encoder_sha256'])
    if selected_epoch != adapted['training']['selected_epoch']:
        raise ValueError('Selected joint head belongs to another epoch')
    stage_losses, selection_scores = {}, {}
    for name, x, head in [('frozen', x0, heads['frozen']), ('selected_joint_head', x1, joint),
                          ('refitted_C1_head', x1, heads['fine-tuned'])]:
        stage_losses[name] = {}
        for part in ['train', 'selection']:
            selected_rows = [rows[i] for i in indices[part]]
            scores = logits(x[indices[part]], head)
            loss, _ = weighted_loss(selected_rows, scores)
            stage_losses[name][part + '_weighted_logloss'] = loss
            if part == 'selection':
                selection_scores[name] = scores
    joint_error = abs(stage_losses['selected_joint_head']['selection_weighted_logloss'] - joint_reference)
    if joint_error > 1e-4:
        raise ValueError('Cached selected encoder/joint head does not reproduce its selection receipt')
    selection_rows = [rows[i] for i in indices['selection']]
    for family, stage in [('frozen', 'frozen'), ('fine-tuned', 'refitted_C1_head')]:
        cutoff, components = broad_cutoff(selection_scores[stage], selection_rows)
        if cutoff != results[family]['model']['cutoff_logit'] or components != results[family]['model']['component_cutoffs']:
            raise ValueError('Locked decision rule differs from its completed model')
    validation = [rows[i] for i in indices['validation']]
    s0 = logits(x0[indices['validation']], heads['frozen'])
    s1 = logits(x1[indices['validation']], heads['fine-tuned'])
    c0, c1 = frozen['model']['cutoff_logit'], adapted['model']['cutoff_logit']
    cells, p0, p1 = source_class_pairs(validation, s0, s1, c0, c1)
    verify_confusions(cells, frozen, 'frozen')
    verify_confusions(cells, adapted, 'adapted')
    ledger = [dict(record_id_hash=digest(r['id']), group_id_hash=digest(r['group_id']),
                   source=r['source'], label=r['label'], frozen_logit=float(s0[i]),
                   adapted_logit=float(s1[i]), frozen_prediction=bool(p0[i]), adapted_prediction=bool(p1[i]))
              for i, r in enumerate(validation)]
    ledger_path = args.private_output / 'paired-predictions.jsonl'
    write_jsonl(ledger_path, ledger)
    root = Path(__file__).resolve().parents[1]
    summary = dict(
        captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        held_source=args.fold, positive_mix=args.positive_mix, corpus_rows=len(rows),
        compared_validation_rows=len(validation), selected_epoch=selected_epoch,
        optimizer_steps=adapted['training']['optimizer_steps'],
        training=partition_summary(rows, train_ids),
        stage_losses=stage_losses, joint_selection_receipt_absolute_error=joint_error,
        source_class_pairs=cells, frozen_cutoff_logit=c0, adapted_cutoff_logit=c1,
        frozen_evaluation=frozen, adapted_evaluation=adapted,
        private_paired_ledger_sha256=file_hash(ledger_path),
        plan_sha256=plan_sha, manifest_sha256=file_hash(args.work / 'manifest.jsonl'),
        base_embeddings_sha256=meta0['sha256'], adapted_embeddings_sha256=meta1['sha256'],
        code_sha256={name: file_hash(root / name) for name in [
            'scripts/compare_completed_beats_arm.py', 'src/drone_detector/paired_comparison.py',
            'src/drone_detector/beats_training.py', 'scripts/train_beats_adaptation.py']},
        scope='Complete matched frozen/adapted known-corpus development comparison. Both heads and per-DEV-source cutoffs were fixed before held-source scoring. No threshold tuning, new fit, encoder inference, bootstrap or independent-field claim.',
        limitations=[
            'Publisher labels are weak and crop audibility has not been independently reviewed.',
            'Correlated excerpts/capture groups prevent treating clip counts as independent field trials.',
            'FSD false-positive clip rates do not establish false alarms/hour.',
            'This single completed arm does not settle the four-arm study or identify the cause of transfer failures.',
        ],
    )
    write_json(args.output / 'paired-comparison.json', summary)
    captured = datetime.datetime.fromisoformat(summary['captured_at_utc']).astimezone(ZoneInfo('America/New_York'))
    lines = [
        '# Completed BEATs arm: paired comparison', '',
        f'Captured {captured.isoformat()} (America/New_York). Held source: **{args.fold}**; positive mix: **{args.positive_mix}**.', '',
        f'All three epochs completed: **{summary["optimizer_steps"]:,} steps**, selected epoch **{selected_epoch}**. '
        f'The final C1 head is refitted and its DEV cutoff locked. All **{len(validation):,}** prescribed validation rows are compared; no reduced cohort.', '',
        '| Source / publisher label | Clips | Frozen errors | Adapted errors | Corrected | New errors | Both wrong |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for cell in cells:
        lines.append(f'| {cell["source"]} / {cell["label"]} | {cell["clips"]:,} | '
                     f'{cell["frozen_errors"]:,} | {cell["adapted_errors"]:,} | '
                     f'{cell["corrected_by_adaptation"]:,} | {cell["new_errors_after_adaptation"]:,} | {cell["both_wrong"]:,} |')
    lines += ['', 'Positive-label errors are missed drone clips; negative-label errors are false-positive clips. '
              'Each classifier uses its own cutoff from the same prescribed DEV rule. The paired counts are descriptive, with no independent-clip significance claim.', '',
              '## What the final head refit changed', '',
              '| Classifier stage | TRAIN weighted log-loss | Selection-DEV weighted log-loss |',
              '|---|---:|---:|']
    for name, values in stage_losses.items():
        lines.append(f'| {name} | {values["train_weighted_logloss"]:.6f} | {values["selection_weighted_logloss"]:.6f} |')
    lines += ['', f'The selected encoder and saved joint head reproduce their recorded DEV loss within {joint_error:.3g}. '
              'This separates the interim joint-head diagnostic from the final refitted classifier. '
              'Log-loss measures confidence as well as classification errors.', '',
              '## Evidence and limits', '',
              '[paired-comparison.json](paired-comparison.json) retains complete source-level evaluation receipts, '
              'paired counts, cutoffs and input/model/code hashes. Reconstructed confusion counts and DEV cutoff '
              'calculations exactly match the locked completed artifacts. The complete per-record paired ledger remains private.', '']
    lines += [f'- {limit}' for limit in summary['limitations']]
    lines += ['', 'The remaining study arms continue under the original recipe and cumulative 44-hour budget. '
              'Public FPV candidates remain outside this experiment.']
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    print({'output': str(args.output), 'compared_rows': len(validation), 'paired_counts': cells,
           'stage_losses': stage_losses, 'joint_reference_error': joint_error})


if __name__ == '__main__':
    main()
