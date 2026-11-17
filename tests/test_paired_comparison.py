"""Software decision fixtures, not acoustic research results."""

import numpy as np
import pytest

from drone_detector.paired_comparison import paired_outcomes, source_class_pairs


def test_paired_cells_account_for_both_error_directions_and_unchanged_cases():
    result = paired_outcomes([1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 1, 1])
    assert result == dict(
        clips=4, both_correct=1, both_wrong=1, corrected_by_adaptation=1,
        new_errors_after_adaptation=1, frozen_errors=2, adapted_errors=2,
        frozen_confusion=dict(tp=1, fn=1, fp=1, tn=1),
        adapted_confusion=dict(tp=2, fn=0, fp=2, tn=0),
    )


def test_thresholds_are_separate_and_ties_use_the_locked_greater_equal_rule():
    rows = [dict(source='s', label=label, group_id='same-capture')
            for label in ['drone', 'drone', 'non_drone']]
    cells, before, after = source_class_pairs(rows, [2, 1, 2], [9, 10, 9], 2, 10)
    np.testing.assert_array_equal(before, [True, False, True])
    np.testing.assert_array_equal(after, [False, True, False])
    assert sum(r['clips'] for r in cells) == 3
    assert all(r['declared_provenance_groups'] == 1 for r in cells)
    assert cells[0]['corrected_by_adaptation'] == 1
    assert cells[0]['new_errors_after_adaptation'] == 1
    assert cells[1]['corrected_by_adaptation'] == 1


@pytest.mark.parametrize('truth,before,after', [([1], [1, 0], [1]), ([2], [1], [1]),
                                             ([1], [np.nan], [1]), ([[1]], [1], [1])])
def test_invalid_vectors_cannot_silently_drop_or_relabel_comparisons(truth, before, after):
    with pytest.raises(ValueError, match='binary'):
        paired_outcomes(truth, before, after)


def test_nonfinite_logits_or_thresholds_are_rejected():
    rows = [dict(source='s', label='drone', group_id='capture')]
    with pytest.raises(ValueError, match='Invalid'):
        source_class_pairs(rows, [np.nan], [1], 0, 0)
    with pytest.raises(ValueError, match='Invalid'):
        source_class_pairs(rows, [1], [1], float('inf'), 0)
