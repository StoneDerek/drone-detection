import copy

import pytest

from drone_detector.adaptation import (
    assert_partitions,
    batches_by_length,
    group_weights,
    make_folds,
)


def corpus():
    def row(identifier, source, label, split, group=None):
        return dict(
            id=identifier,
            source=source,
            label=label,
            split=split,
            group_id=group or identifier,
            sha256=identifier,
            decoded_sha256=identifier,
            duration_seconds=3.0,
        )

    return [
        row("oldpos", "poland", "drone", "train"),
        row("oldneg", "urban", "non_drone", "train"),
        row("selectionpos", "poland", "drone", "development"),
        row("selectionneg", "urban", "non_drone", "development"),
        row("wsupos", "wsu-air3", "drone", "final", "wsu-date"),
        row("wsuneg", "wsu-air3", "non_drone", "final", "wsu-date"),
        row("audrokpos", "audrok-dresden", "drone", "final"),
        row("fsdneg", "fsd50k-eval", "non_drone", "final"),
    ]


def test_whole_source_holdout_and_matched_negative_training():
    rows = corpus()
    folds = make_folds(rows)
    for held_source, spec in folds.items():
        current, broader = set(spec["current_positive_train"]), set(spec["broader_positive_train"])
        assert current < broader
        negatives = {r["id"] for r in rows if r["label"] == "non_drone"}
        assert current & negatives == broader & negatives
        for r in rows:
            if r["source"] in {held_source, "fsd50k-eval"}:
                assert r["id"] in spec["validation"]
                assert r["id"] not in current | broader | set(spec["selection"])
    assert "wsuneg" in folds["audrok-dresden"]["current_positive_train"]
    assert "wsupos" not in folds["audrok-dresden"]["current_positive_train"]
    assert "wsupos" in folds["audrok-dresden"]["broader_positive_train"]


def test_cross_label_capture_may_not_leak_into_validation():
    rows = corpus()
    rows[2]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="leakage"):
        make_folds(rows)
    rows = corpus()
    rows[-1]["decoded_sha256"] = rows[0]["decoded_sha256"]
    with pytest.raises(ValueError, match="leakage"):
        assert_partitions(rows, ["oldpos"], ["selectionneg"], ["fsdneg"])


def test_exact_length_batches_cover_every_example_without_padding():
    rows = [dict(duration_seconds=n / 16000) for n in [48000, 4000, 48000, 7000, 4000]]
    a = batches_by_length(rows, batch_size=2, seed=42, epoch=1)
    assert a == batches_by_length(rows, batch_size=2, seed=42, epoch=1)
    assert sorted(i for batch in a for i in batch) == list(range(len(rows)))
    assert all(len({rows[i]["duration_seconds"] for i in batch}) == 1 for batch in a)


def test_balancing_respects_classes_sources_groups_and_fixed_mass():
    rows = corpus()[:2]
    rows += [dict(id="anotherpos", source="berne", label="drone", group_id="berne")]
    duplicate = copy.deepcopy(rows[-1])
    duplicate["id"] = "another-frame-same-capture"
    rows.append(duplicate)
    mass = group_weights(rows, total_mass=100)
    assert mass.sum() == pytest.approx(100)
    assert mass[[0, 2, 3]].sum() == pytest.approx(50)
    assert mass[0] == pytest.approx(25)
    assert mass[2] + mass[3] == pytest.approx(25)
