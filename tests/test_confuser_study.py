import importlib.util
from pathlib import Path

import numpy as np


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_confuser_study.py"
    spec = importlib.util.spec_from_file_location("confuser_study", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_global_cutoff_honors_each_source_with_ties():
    study = module()
    rows = (
        [dict(source="ambient", label="non_drone") for _ in range(200)]
        + [dict(source="tools", label="non_drone") for _ in range(100)]
        + [dict(source="positives", label="drone")]
    )
    scores = np.array([0.0] * 198 + [1.0] * 2 + [2.0] * 98 + [3.0] * 2 + [100.0])
    cutoff, values = study.broad_cutoff(scores, rows)
    assert cutoff > 3.0
    assert cutoff == max(values.values())
    for source in ["ambient", "tools"]:
        ix = [i for i, r in enumerate(rows) if r["source"] == source]
        assert np.mean(scores[ix] >= cutoff) <= 0.01
    scores[-1] = -100
    assert study.broad_cutoff(scores, rows) == (cutoff, values)
