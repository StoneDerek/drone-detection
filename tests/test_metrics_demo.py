import copy
import json

import pytest

from drone_detector.common import file_hash, read_json, write_json
from drone_detector.demo import build_demo
from drone_detector.evaluation import record_metrics, summarize


def trace(alarms):
    return [{"timestamp_seconds": t, "delivered_at_seconds": t + .01, "alarm": active}
            for t, active in alarms]


def reference():
    return {"id": "r", "source": "s", "group_id": "g", "origin": "synthetic",
            "duration_seconds": 30, "events": [[5, 10], [18, 22]], "ambiguous": [[25, 27]]}


def test_event_matching_exposure_delay_and_ambiguity(prepared):
    contract = read_json(prepared["contract"])
    record = reference()
    rows = trace([(3, False), (6, True), (8, False), (15, True), (16, False),
                  (19, True), (23, False), (25.5, True), (28, False)])
    result = record_metrics(record, rows, contract)
    assert result["event_count"] == 2 and result["matched_events"] == 2
    assert result["false_alarm_count"] == 1
    assert result["negative_seconds"] == 17  # 30 - (6+5) collars - 2 ambiguous
    assert result["delays_seconds"] == pytest.approx([1.01, 1.01])
    assert result["recording_error"]


def test_latched_alarm_before_event_does_not_get_free_recall(prepared):
    contract = read_json(prepared["contract"])
    result = record_metrics(reference(), trace([(3, True), (29, True)]), contract)
    assert result["matched_events"] == 0 and result["false_alarm_count"] == 1
    assert result["negative_alarm_seconds"] > 0


def test_one_alarm_matches_one_event_only(prepared):
    contract = read_json(prepared["contract"])
    result = record_metrics(reference(), trace([(6, True), (29, True)]), contract)
    assert result["matched_events"] == 1
    assert result["missed_event_indices"] == [1]


def test_uncertainty_not_zero_risk_and_clusters_not_windows(prepared):
    contract = read_json(prepared["contract"])
    r = reference()
    row = record_metrics(r, trace([(6, True), (11, False), (19, True), (23, False)]), contract)
    result = summarize([row])
    assert result["independent_groups"] == 1 and result["events"] == 2
    assert result["false_alarms_per_hour"] == 0
    assert result["uncertainty"]["fa_per_hour_conservative_upper_95"] > 0
    assert result["uncertainty"]["recall_conservative_lower_95"] < .9
    copy_row = copy.deepcopy(row)
    copy_row["id"] = "another-excerpt"
    assert summarize([row, copy_row])["independent_groups"] == 1


def test_zero_recordings_rejected():
    with pytest.raises(ValueError, match="Zero"):
        summarize([])


def test_demo_embeds_disclosed_audio_scores_and_identity(prepared, tmp_path):
    wanted = {"recording_ids": [r["id"] for r in prepared["rows"] if r["split"] == "final"][:3],
              "reason": "fixed unit-test set", "manifest_sha256": file_hash(prepared["manifest"])}
    selection = tmp_path / "selection.json"
    write_json(selection, wanted)
    output = tmp_path / "demo.html"
    build_demo(prepared["bundle"], prepared["manifest"], selection, output)
    page = output.read_text()
    assert "not microphone capture" in page
    assert "data:audio/wav;base64," in page
    assert prepared["model"].model_id in page
    payload = page.split('id="payload">', 1)[1].split("</script>", 1)[0]
    parsed = json.loads(payload)
    assert len(parsed["records"]) == 3
    assert all(r["trace"] for r in parsed["records"])


def test_demo_selection_binding(prepared, tmp_path):
    selection = tmp_path / "wrong.json"
    write_json(selection, {"recording_ids": [], "manifest_sha256": "wrong", "reason": "test"})
    with pytest.raises(ValueError, match="manifest mismatch"):
        build_demo(prepared["bundle"], prepared["manifest"], selection, tmp_path / "wrong.html")
