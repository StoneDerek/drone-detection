"""Software checks for provenance claims; fixtures are not drone evaluations."""

import numpy as np

from drone_detector.fpv_lineage import declared_lineage, exact_subsequence, summarize, union_length


def test_missing_original_does_not_become_verified_through_a_device_name():
    entries = [
        {"path": "Fpv/labeled_data/dayA/phone/drone/drone1.wav"},
        {"path": "Fpv/labeled_data/ProcessedAudioSplits/drone/test/drone_phone_drone2_test.wav"},
    ]
    rows = declared_lineage(entries, [])
    assert rows[0]["status"] == "original_parent_absent"
    assert rows[0]["candidate_parents"] == []
    assert rows[0]["candidate_capture_groups"] == ["Fpv/labeled_data/dayA"]


def test_alias_collision_copy_suffix_and_shared_capture_are_kept_explicit():
    entries = [
        {"path": "Fpv/labeled_data/dayA/phone/drone/drone1.wav"},
        {"path": "Fpv/labeled_data/dayB/phone/drone/drone1.wav"},
        {"path": "Fpv/labeled_data/ProcessedAudioSplits/drone/train/drone_phone_drone1_train.wav"},
        {"path": "Fpv/labeled_data/ProcessedAudioSplits/drone/test/drone_phone_drone1_test(1).wav"},
    ]
    rows = declared_lineage(entries, [])
    assert rows[0]["status"] == "ambiguous_parent"
    assert len(rows[0]["candidate_capture_groups"]) == 2
    assert rows[1]["status"] == "unresolved_copy_suffix"
    assert rows[1]["copy_tag"] == "(1)"
    counts = summarize(rows)
    assert counts["declared_source_aliases_in_multiple_partitions"] == 1
    assert counts["candidate_capture_groups_in_multiple_partitions"] == 2


def test_overlapping_windows_are_not_added_as_independent_time():
    assert union_length([(0, 1000), (100, 1100), (1000, 2000), (3000, 3500)]) == 2500
    assert union_length([]) == 0


def test_waveform_matching_requires_a_unique_exact_native_subsequence():
    source = np.arange(30, dtype=np.float32).reshape(15, 2)
    assert exact_subsequence(source, source[4:8].copy()) == {
        "status": "unique_exact_match", "start_frame": 4, "frames": 4,
    }
    repeated = np.zeros((20, 1), dtype=np.float32)
    assert exact_subsequence(repeated, repeated[:3])["status"] == "ambiguous_repeated_waveform"
    assert exact_subsequence(source, source[4:8] + .1)["status"] == "no_exact_match"
    assert exact_subsequence(source, source[:4].astype(np.float64))["status"] == "incompatible_sample_dtype"
