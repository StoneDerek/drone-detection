"""Read-only checks of BEATs schedules and sampled optimization telemetry.

These checks use the complete TRAIN/selection schedules, not new audio scores.
They cannot establish generalization or infer every batch's loss from sparse logs.
"""

import collections
import json

import numpy as np

from .adaptation import batches_by_length, group_weights


def weighted_loss(rows, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError('Expected one finite logit per selected row')
    weights = group_weights(rows)
    signs = np.asarray([1 if r['label'] == 'drone' else -1 for r in rows])
    terms = np.logaddexp(0, -signs * scores)
    return float(np.dot(weights, terms) / weights.sum()), terms


def schedule_audit(rows, recipe, terms):
    weights = group_weights(rows)
    if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 9959):
        raise ValueError('Invalid global sample-weight mass')
    terms = np.asarray(terms, dtype=np.float64)
    if terms.shape != weights.shape or not np.isfinite(terms).all() or np.any(terms < 0):
        raise ValueError('Expected finite nonnegative fixed-parameter losses')
    source_cells = collections.defaultdict(list)
    for i, row in enumerate(rows):
        source_cells[row['label'], row['source']].append(i)
    for label in ['drone', 'non_drone']:
        cells_for_label = [indices for (cls, _), indices in source_cells.items() if cls == label]
        for indices in cells_for_label:
            expected = 9959 / (2 * len(cells_for_label))
            group_indices = collections.defaultdict(list)
            for i in indices:
                group_indices[rows[i]['group_id']].append(i)
            if (not np.isclose(weights[indices].sum(), expected)
                    or any(not np.isclose(weights[group].sum(), expected / len(group_indices))
                           for group in group_indices.values())):
                raise ValueError('Class/source/group weight allocation differs from the protocol')
    cells = []
    for (label, source), indices in sorted(source_cells.items()):
        cells.append(dict(label=label, source=source, clips=len(indices),
                          capture_groups=len({rows[i]['group_id'] for i in indices}),
                          weight_mass=float(weights[indices].sum()),
                          fraction_of_total_weight=float(weights[indices].sum() / weights.sum())))
    result = []
    for epoch in range(recipe['epochs']):
        batches = batches_by_length(rows, recipe['batch_size'], recipe['seed'], epoch)
        visits = [i for batch in batches for i in batch]
        if sorted(visits) != list(range(len(rows))):
            raise ValueError('Schedule repeats or omits a selected crop')
        if any(len({round(rows[i]['duration_seconds'] * 16000) for i in b}) != 1 for b in batches):
            raise ValueError('Mixed waveform lengths in an unpadded batch')
        normalizer = float(weights.sum()) / len(batches)
        masses = np.array([weights[b].sum() for b in batches])
        batch_objectives = [float(np.dot(weights[b], terms[b]) / normalizer) for b in batches]
        full = float(np.dot(weights, terms) / weights.sum())
        difference = abs(float(np.mean(batch_objectives)) - full)
        if difference > 1e-12 * max(1, abs(full)):
            raise ValueError('Batch normalization changes the fixed-parameter global objective')
        result.append(dict(
            epoch=epoch + 1, rows_visited=len(visits), unique_rows=len(set(visits)),
            exact_once_coverage=True, batches=len(batches),
            batch_size_counts=dict(collections.Counter(map(len, batches))),
            mixed_class_batches=sum(len({rows[i]['label'] for i in b}) == 2 for b in batches),
            positive_only_batches=sum(all(rows[i]['label'] == 'drone' for i in b) for b in batches),
            negative_only_batches=sum(all(rows[i]['label'] == 'non_drone' for i in b) for b in batches),
            global_weight_mass=float(weights.sum()), constant_batch_normalizer=normalizer,
            batch_weight_mass_quantiles=dict(zip(['min', 'p01', 'p50', 'p99', 'max'],
                                                np.quantile(masses, [0, .01, .5, .99, 1]).tolist(), strict=True)),
            mean_batch_objective_equals_global_loss_absolute_error=difference,
        ))
    return dict(
        rows=len(rows), positive_clips=sum(r['label'] == 'drone' for r in rows),
        equal_class_source_group_weighting_verified=True,
        source_class_masses=cells, epochs=result,
        normalization_scope='Equality is checked at fixed parameters. Parameters change during SGD; gradient clipping/AdamW do not make the optimizer trajectory equivalent to full-batch optimization.',
        coverage_scope='Complete deterministic schedules reconstructed from the pinned cohort/recipe. Runtime logs are sparse, not per-example visit instrumentation.',
    )


def trace_audit(text, batches_per_epoch):
    """Inspect the first arm only; stop before any following arm's telemetry."""
    records, completed, replayed, mismatched = {}, {}, 0, 0
    initial = None
    for line in text.splitlines():
        if line.startswith('Completed full adaptation arm '):
            break
        if line.startswith('Initial selection weighted log-loss '):
            initial = float(line.rsplit(' ', 1)[1])
        if line.startswith('Completed full epoch '):
            row = json.loads(line.removeprefix('Completed full epoch '))
            previous = completed.get(row['epoch'])
            if previous is not None and previous != row:
                raise ValueError('Conflicting completed-epoch receipts')
            completed[row['epoch']] = row
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or 'batch_weighted_loss_sum' not in row:
            continue
        if row['steps'] != (row['epoch'] - 1) * batches_per_epoch + row['batch']:
            raise ValueError('Logged step does not match the reconstructed epoch schedule')
        for field in ['batch_weighted_loss_sum', 'gradient_norm']:
            if not np.isfinite(row[field]) or row[field] < 0:
                raise ValueError('Invalid optimization telemetry')
        key = row['epoch'], row['batch'], row['steps']
        if key in records:
            replayed += 1
            if any(records[key][field] != row[field] for field in ['batch_weighted_loss_sum', 'gradient_norm']):
                mismatched += 1
        records[key] = row
    epochs = []
    for epoch in sorted({r['epoch'] for r in records.values()}):
        rows = [r for r in records.values() if r['epoch'] == epoch]
        norms = np.array([r['gradient_norm'] for r in rows])
        epochs.append(dict(
            epoch=epoch, unique_logged_batches=len(rows), planned_batches=batches_per_epoch,
            logged_batches_with_zero_data_loss=sum(r['batch_weighted_loss_sum'] == 0 for r in rows),
            logged_batches_with_preclip_norm_above_one=int(np.sum(norms > 1)),
            preclip_gradient_norm_min_median_max=np.quantile(norms, [0, .5, 1]).tolist(),
        ))
    return dict(
        initial_selection_weighted_logloss=initial, completed_epochs=[completed[e] for e in sorted(completed)],
        logged_epoch_summaries=epochs, replayed_logged_batches=replayed,
        replayed_logged_batches_with_different_loss_or_norm=mismatched,
        scope='First-arm TRAIN telemetry and completed selection-DEV loss only. Every-25-step samples, deduplicated for replay; not epoch-average loss or a complete TRAIN accuracy measurement.',
        loss_scope='batch_weighted_loss_sum omits the constant batch normalizer and the head L2 penalty. A displayed zero can coexist with nonzero gradients; neither every training example nor the complete objective is established zero.',
    )
