import pytest

from drone_detector.spectral_calibration import activation_statistic, choose_budget_trial, exact_budget_trials, freeze_calibration_metadata


def test_calibration_freeze_requires_development_selection():
    with pytest.raises(ValueError, match='development'):
        freeze_calibration_metadata({'model_id': 'software-fixture'}, {}, {}, {'split': 'final'})


def trace(values):
    return [{'score': value, 'timestamp_seconds': i + 3., 'alarm': False}
            for i, value in enumerate(values)]


def test_confirmation_statistic_requires_adjacent_scores():
    rows = trace([.1, .9, .1])
    assert activation_statistic(rows, 1) == .9
    assert activation_statistic(rows, 2) == .1
    assert activation_statistic(trace([.7]), 2) == -float('inf')


def test_exact_development_threshold_resolves_coarse_grid_failure():
    records, scores = [], {}
    for i in range(20):
        identity = f'n{i}'
        records.append({'id': identity, 'label': 'non_drone', 'source': 'controlled',
                        'group_id': identity, 'duration_seconds': 5.})
        scores[identity] = trace([.9 if i == 0 else .4] * 3)
    for i in range(2):
        identity = f'p{i}'
        records.append({'id': identity, 'label': 'drone', 'source': 'controlled',
                        'group_id': identity, 'duration_seconds': 5.})
        scores[identity] = trace([.45 + i * .01] * 3)
    chosen = choose_budget_trial(exact_budget_trials(records, scores))
    assert .4 < chosen['decision']['on_threshold'] < .45
    assert chosen['metrics']['confusion'] == {'tp': 2, 'fp': 1, 'tn': 19, 'fn': 0}
    assert chosen['metrics']['negative_recording_false_positive_fraction'] == pytest.approx(.05)
