"""Numerical/reporting fixtures; these are not drone-performance trials."""

import json

import numpy as np
import pytest

from drone_detector.optimization_audit import schedule_audit, trace_audit, weighted_loss


def test_unequal_buckets_preserve_full_weighted_objective_without_resampling():
    rows = [dict(label=label, source=source, group_id=group, duration_seconds=duration)
            for label, source, group, duration in [
                ('drone', 'a', 'one', 3), ('drone', 'a', 'one', 3),
                ('drone', 'b', 'two', 1), ('non_drone', 'c', 'three', 3),
                ('non_drone', 'c', 'four', 2),
            ]]
    loss, terms = weighted_loss(rows, np.array([3, -1, 2, -4, 1]))
    assert loss > 0
    audit = schedule_audit(rows, dict(batch_size=2, seed=42, epochs=3), terms)
    assert [e['rows_visited'] for e in audit['epochs']] == [5, 5, 5]
    assert all(e['mean_batch_objective_equals_global_loss_absolute_error'] < 1e-12 for e in audit['epochs'])
    assert all(e['batch_size_counts'] == {1: 3, 2: 1} for e in audit['epochs'])
    for label in ['drone', 'non_drone']:
        assert sum(c['fraction_of_total_weight'] for c in audit['source_class_masses'] if c['label'] == label) == pytest.approx(.5)
    with pytest.raises(ValueError, match='finite logit'):
        weighted_loss(rows, np.array([np.nan] * 5))


def test_trace_deduplicates_replay_without_treating_samples_as_epoch_accuracy():
    row = dict(epoch=1, batch=25, steps=25, batch_weighted_loss_sum=0., gradient_norm=.01)
    lines = [json.dumps(row), 'Resuming exact training batch 0 0 0', json.dumps(row),
             'Completed full epoch ' + json.dumps(dict(epoch=1, steps=100, selection_weighted_logloss=2.)),
             'Completed full adaptation arm audrok-dresden current {}',
             json.dumps(dict(row, steps=99999))]
    result = trace_audit('\n'.join(lines), 100)
    assert result['replayed_logged_batches'] == 1
    assert result['replayed_logged_batches_with_different_loss_or_norm'] == 0
    assert result['logged_epoch_summaries'][0]['unique_logged_batches'] == 1
    assert result['logged_epoch_summaries'][0]['planned_batches'] == 100
    assert result['logged_epoch_summaries'][0]['logged_batches_with_zero_data_loss'] == 1
    different = trace_audit(json.dumps(row) + '\n' + json.dumps(dict(row, gradient_norm=.02)), 100)
    assert different['replayed_logged_batches_with_different_loss_or_norm'] == 1


def test_trace_rejects_inconsistent_epoch_counters_and_nonfinite_norms():
    row = dict(epoch=2, batch=25, steps=125, batch_weighted_loss_sum=0., gradient_norm=.01)
    with pytest.raises(ValueError, match='step'):
        trace_audit(json.dumps(dict(row, steps=25)), 100)
    with pytest.raises(ValueError, match='telemetry'):
        trace_audit(json.dumps(dict(row, gradient_norm=float('nan'))), 100)


def test_audit_rejects_a_schedule_that_repeats_one_crop_and_omits_another(monkeypatch):
    from drone_detector import optimization_audit

    rows = [dict(label=label, source='same', group_id=label, duration_seconds=3)
            for label in ['drone', 'non_drone']]
    monkeypatch.setattr(optimization_audit, 'batches_by_length', lambda *args: [[0], [0]])
    with pytest.raises(ValueError, match='repeats or omits'):
        schedule_audit(rows, dict(batch_size=1, seed=42, epochs=1), np.ones(2))
