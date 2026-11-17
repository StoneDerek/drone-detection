#!/usr/bin/env python3
"""Audit complete schedules and an immutable read of an active recovery file."""

import argparse
import datetime
import hashlib
from pathlib import Path

import numpy as np

from drone_detector.adaptation import assert_partitions, cohort_signature
from drone_detector.common import file_hash, read_json, read_jsonl, write_json
from drone_detector.optimization_audit import schedule_audit, trace_audit, weighted_loss


def checkpoint_snapshot(path, plan, train_ids, initial_head, root):
    import torch

    torch.set_num_threads(1)
    receipt = read_json(path.with_suffix(path.suffix + '.json'))
    # Keep one file descriptor: atomic checkpoint replacement cannot switch the
    # bytes between checksum verification and weights-only loading.
    with path.open('rb') as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
        if checksum.hexdigest() != receipt['sha256']:
            raise ValueError('Checkpoint changed during snapshot acquisition; retry between saves')
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
    identity = state['identity']
    from drone_detector.common import digest

    if (identity['plan_sha256'] != plan['sha256']
            or identity['train_ids_sha256'] != digest(train_ids)
            or identity['recipe'] != plan['recipe']['fine_tuning']
            or identity['held_source'] != 'audrok-dresden'
            or identity['positive_mix'] != 'current'
            or identity['training_implementation_sha256'] != file_hash(root / 'src/drone_detector/beats_training.py')
            or identity['driver_sha256'] != file_hash(root / 'scripts/train_beats_adaptation.py')):
        raise ValueError('Checkpoint does not belong to the first unchanged study arm')
    for key in ['mean', 'scale']:
        if not np.array_equal(state['model'][key].numpy(), initial_head[key].astype(np.float32)):
            raise ValueError('Joint training scaler differs from its paired frozen head')
    groups = state['optimizer']['param_groups']
    recipe = plan['recipe']['fine_tuning']
    if (len(groups) != 2 or groups[0]['lr'] != recipe['encoder_lr']
            or groups[1]['lr'] != recipe['head_lr']
            or groups[0]['weight_decay'] != recipe['encoder_weight_decay']
            or groups[1]['weight_decay'] != 0):
        raise ValueError('Optimizer recipe changed')
    tensors = list(state['model'].values())
    tensors += [v for item in state['optimizer']['state'].values() for v in item.values()
                if isinstance(v, torch.Tensor)]
    if not all(bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError('Nonfinite model or optimizer checkpoint tensor')
    head_norm = float(torch.linalg.vector_norm(state['model']['head.weight'].double()))
    witness = torch.tensor([-15., -20., 15., 20.], requires_grad=True)
    labels = torch.tensor([0., 0., 1., 1.])
    losses = torch.nn.functional.binary_cross_entropy_with_logits(witness, labels, reduction='none')
    losses.sum().backward()
    numerical = dict(
        scope='Numerical float32 BCE witness, not acoustic examples or measured training logits.',
        logits=witness.detach().tolist(), labels=labels.tolist(), reported_loss=losses.detach().tolist(),
        gradient=witness.grad.tolist(),
    )
    return dict(
        sha256=checksum.hexdigest(), epoch_zero_based=state['epoch'], next_batch_position=state['position'],
        optimizer_steps=state['steps'], history=state['history'],
        initial_reference_loss=state['initial_reference'], best_loss=state['best_loss'],
        model_and_optimizer_tensors_checked_finite=len(tensors),
        frozen_scaler_exactly_preserved=True, optimizer_recipe_verified=True,
        head_weight_L2_norm=head_norm, head_L2_penalty=head_norm ** 2 / (2 * 9959),
        head_regularization_gradient_L2_norm=head_norm / 9959,
        numerical_zero_loss_witness=numerical, torch_version=str(torch.__version__),
        training_implementation_sha256=identity['training_implementation_sha256'],
        driver_sha256=identity['driver_sha256'],
        scope='Read-only recovery checkpoint snapshot. No training forward pass, optimization step, checkpoint promotion or live-process state change.',
    )


def partition_loss(rows, x, head):
    scores = ((x - head['mean']) / head['scale']) @ head['coef'] + float(head['intercept'][0])
    loss, terms = weighted_loss(rows, scores)
    return dict(rows=len(rows), weighted_logloss=loss), terms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('work/beats-adaptation-cpu2-v1'))
    parser.add_argument('--output', type=Path, default=Path('reports/beats-adaptation-v1/optimization-audit'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = read_jsonl(args.work / 'manifest.jsonl')
    plan = read_json(args.work / 'plan.json')
    if cohort_signature(rows) != plan['recipe']['cohort_signature']:
        raise ValueError('Cohort changed')
    plan['sha256'] = file_hash(args.work / 'plan.json')
    metadata = read_json(args.work / 'base-embeddings.json')
    if (metadata['sha256'] != file_hash(args.work / 'base-embeddings.npz')
            or metadata['cohort_signature'] != cohort_signature(rows)):
        raise ValueError('Base embedding identity changed')
    with np.load(args.work / 'base-embeddings.npz', allow_pickle=False) as saved:
        if saved['ids'].tolist() != [r['id'] for r in rows]:
            raise ValueError('Embedding row identity changed')
        x = saved['x'].copy()
    lookup = {r['id']: i for i, r in enumerate(rows)}
    controls, first_head = {}, None
    for fold, spec in plan['folds'].items():
        controls[fold] = {}
        for mix in ['current', 'broader']:
            ids = spec[mix + '_positive_train']
            assert_partitions(rows, ids, spec['selection'], spec['validation'])
            folder = args.work / fold / mix / 'frozen'
            model = read_json(folder / 'model.json')
            if (model['parameters_sha256'] != file_hash(folder / 'head.npz')
                    or model['encoder'] != metadata['encoder']):
                raise ValueError('Paired frozen head/runtime identity changed')
            with np.load(folder / 'head.npz', allow_pickle=False) as saved:
                head = {key: saved[key].copy() for key in saved.files}
            train_indices, dev_indices = [lookup[i] for i in ids], [lookup[i] for i in spec['selection']]
            training, selection = [rows[i] for i in train_indices], [rows[i] for i in dev_indices]
            train_result, terms = partition_loss(training, x[train_indices], head)
            dev_result, _ = partition_loss(selection, x[dev_indices], head)
            controls[fold][mix] = dict(
                frozen_TRAIN=train_result, frozen_selection_DEV=dev_result,
                schedules=schedule_audit(training, plan['recipe']['fine_tuning'], terms),
                head_sha256=model['parameters_sha256'],
            )
            if fold == 'audrok-dresden' and mix == 'current':
                first_head = head
    assert first_head is not None
    first = controls['audrok-dresden']['current']
    batch_count = first['schedules']['epochs'][0]['batches']
    checkpoint = checkpoint_snapshot(
        args.work / 'audrok-dresden/current/fine-tuned/latest.pt', plan,
        plan['folds']['audrok-dresden']['current_positive_train'], first_head, root,
    )
    if checkpoint['optimizer_steps'] != checkpoint['epoch_zero_based'] * batch_count + checkpoint['next_batch_position']:
        raise ValueError('Checkpoint counters do not match complete exact-length schedules')
    initial_error = abs(first['frozen_selection_DEV']['weighted_logloss'] - checkpoint['initial_reference_loss'])
    if initial_error > 1e-5:
        raise ValueError('Cached frozen-head DEV calculation disagrees with native initial reference')
    trace = trace_audit((args.work / 'four-full-fine-tuning-arms.log').read_text(), batch_count)
    completed = [r for r in trace['completed_epochs'] if r['epoch'] <= checkpoint['epoch_zero_based']]
    if checkpoint['history'] != completed or trace['replayed_logged_batches_with_different_loss_or_norm']:
        raise ValueError('Checkpoint history or deterministic replay differs from the trace')
    result = dict(
        captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='All four complete planned TRAIN schedules, frozen TRAIN/selection diagnostics, and first-arm optimization telemetry. No held-source/FSD score calculation, new acoustic benchmark or recipe change.',
        corpus_rows=len(rows), manifest_sha256=file_hash(args.work / 'manifest.jsonl'),
        plan_sha256=plan['sha256'], controls=controls,
        checkpoint=checkpoint, first_arm_trace=trace,
        initial_frozen_DEV_vs_native_reference_absolute_error=initial_error,
        base_embeddings_sha256=metadata['sha256'],
        code_sha256={name: file_hash(root / name) for name in [
            'src/drone_detector/adaptation.py', 'src/drone_detector/beats_training.py',
            'scripts/train_beats_adaptation.py', 'src/drone_detector/optimization_audit.py',
            'scripts/audit_beats_optimization.py']},
        interpretation_limits=[
            'Fixed-parameter objective checks do not certify a successful learning rate, robustness or optimal batching.',
            'Sparse training-loss logs do not measure full-epoch TRAIN loss or accuracy.',
            'Worse joint-head DEV loss does not determine performance after the prescribed final C1 head refit.',
            'Source-specific shortcut learning, inadequate diversity and feature adaptation remain alternative explanations.',
        ],
    )
    write_json(args.output / 'audit.json', result)
    print({'output': str(args.output / 'audit.json'), 'checkpoint_steps': checkpoint['optimizer_steps'],
           'completed_epochs': checkpoint['epoch_zero_based'], 'initial_reference_error': initial_error,
           'first_arm_frozen_TRAIN_logloss': first['frozen_TRAIN']['weighted_logloss'],
           'first_arm_frozen_DEV_logloss': first['frozen_selection_DEV']['weighted_logloss']})


if __name__ == '__main__':
    main()
