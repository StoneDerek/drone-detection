"""Descriptive paired decisions at already locked classifier thresholds."""

import numpy as np


def paired_outcomes(truth, frozen, adapted):
    values = [np.asarray(v) for v in [truth, frozen, adapted]]
    if (any(v.ndim != 1 for v in values)
            or len({len(v) for v in values}) != 1
            or any(not np.isin(v, [0, 1]).all() for v in values)):
        raise ValueError('Expected equal-length binary truth and decision vectors')
    y, before, after = [v.astype(bool) for v in values]
    right_before, right_after = before == y, after == y
    result = dict(
        clips=len(y), both_correct=int(np.sum(right_before & right_after)),
        both_wrong=int(np.sum(~right_before & ~right_after)),
        corrected_by_adaptation=int(np.sum(~right_before & right_after)),
        new_errors_after_adaptation=int(np.sum(right_before & ~right_after)),
        frozen_errors=int(np.sum(~right_before)), adapted_errors=int(np.sum(~right_after)),
    )
    for name, predictions in [('frozen', before), ('adapted', after)]:
        result[name + '_confusion'] = dict(
            tp=int(np.sum(predictions & y)), fn=int(np.sum(~predictions & y)),
            fp=int(np.sum(predictions & ~y)), tn=int(np.sum(~predictions & ~y)),
        )
    return result


def source_class_pairs(rows, frozen_scores, adapted_scores, frozen_cutoff, adapted_cutoff):
    scores = [np.asarray(s, dtype=np.float64) for s in [frozen_scores, adapted_scores]]
    if (any(s.shape != (len(rows),) or not np.isfinite(s).all() for s in scores)
            or not np.isfinite([frozen_cutoff, adapted_cutoff]).all()
            or any(r['label'] not in {'drone', 'non_drone'} for r in rows)):
        raise ValueError('Invalid fixed-threshold comparison inputs')
    frozen, adapted = scores[0] >= frozen_cutoff, scores[1] >= adapted_cutoff
    y = np.asarray([r['label'] == 'drone' for r in rows])
    cells = []
    for source, label in sorted({(r['source'], r['label']) for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['source'] == source and r['label'] == label]
        cells.append(dict(
            source=source, label=label,
            declared_provenance_groups=len({rows[i]['group_id'] for i in indices}),
            **paired_outcomes(y[indices], frozen[indices], adapted[indices]),
        ))
    return cells, frozen, adapted
