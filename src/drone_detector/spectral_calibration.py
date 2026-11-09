"""Explicit additional development calibration for clustered uncalibrated scores."""

import numpy as np

from .common import digest
from .weak import recording_results, weak_metrics


def freeze_calibration_metadata(parent, decision, contract, selection):
    if selection.get('split') != 'development':
        raise ValueError('Calibration must use development data')
    return {**parent, 'status': 'frozen', 'parent_model_id': parent['model_id'],
            'decision': decision, 'contract': contract, 'contract_sha256': digest(contract),
            'selection': selection}


def activation_statistic(trace, on_hits):
    scores = [row['score'] for row in trace if row['score'] is not None]
    if len(scores) < on_hits:
        return -np.inf
    return max(min(scores[start:start + on_hits]) for start in range(len(scores) - on_hits + 1))


def exact_budget_trials(records, scores, false_positive_budget=.05):
    """Smallest feasible threshold for each confirmation count on DEVELOPMENT."""
    trials = []
    negatives = [r for r in records if r['label'] == 'non_drone']
    if not negatives or not any(r['label'] == 'drone' for r in records):
        raise ValueError('Both development classes are required')
    allowed = int(np.floor(false_positive_budget * len(negatives)))
    for hits in [1, 2]:
        statistic = sorted((activation_statistic(scores[r['id']], hits) for r in negatives), reverse=True)
        cutoff = statistic[allowed] if allowed < len(statistic) else -np.inf
        # Leave the same 1e-5 tolerance used for batch/sequential score parity;
        # an infinitesimal float64 step is too small for float32 CNN arithmetic.
        threshold = float(cutoff + 1e-5) if np.isfinite(cutoff) else 1e-5
        threshold = min(1., max(float(np.finfo(float).eps), threshold))
        decision = {'on_threshold': threshold, 'off_threshold': .75 * threshold,
                    'on_hits': hits, 'off_hits': 2}
        metrics = weak_metrics(recording_results(records, scores, decision))
        trials.append({'decision': decision, 'metrics': metrics})
    return trials


def choose_budget_trial(trials, false_positive_budget=.05):
    def rank(trial):
        fpr = trial['metrics']['negative_recording_false_positive_fraction']
        recall = trial['metrics']['recording_recall']
        return (fpr > false_positive_budget, -recall if fpr <= false_positive_budget else fpr,
                trial['decision']['on_threshold'] if fpr <= false_positive_budget else -recall,
                fpr, -trial['decision']['on_hits'])
    return min(trials, key=rank)
