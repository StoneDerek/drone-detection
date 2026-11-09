from drone_detector.cli import main
from drone_detector.common import read_json, read_jsonl


def test_cli_train_tune_development_and_infer(prepared, tmp_path, capsys):
    manifest = str(prepared["manifest"])
    raw, frozen = tmp_path / "raw", tmp_path / "frozen"
    assert main(["train", "--manifest", manifest, "--output", str(raw),
                 "--representation", "relative_mel"]) == 0
    assert main(["tune", "--bundle", str(raw), "--manifest", manifest,
                 "--contract", str(prepared["contract"]), "--output", str(frozen)]) == 0
    assert main(["evaluate", "--bundle", str(frozen), "--manifest", manifest,
                 "--split", "development", "--output", str(tmp_path / "dev")]) == 0
    result = read_json(tmp_path / "dev/comparison.json")
    assert "DEVELOPMENT ONLY" in result["results"][0]["assessment"]["status"]
    row = next(r for r in prepared["rows"] if r["split"] == "development")
    audio = prepared["manifest"].parent / row["path"]
    output = tmp_path / "trace.jsonl"
    assert main(["infer", "--bundle", str(frozen), "--audio", str(audio),
                 "--output", str(output)]) == 0
    assert len(read_jsonl(output)) == 19
    assert main(["infer", "--bundle", str(frozen), "--audio", str(audio),
                 "--output", str(output)]) == 2  # Preserve existing evidence.
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_cli_missing_data_fails_instead_of_empty_success(tmp_path, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    assert main(["train", "--manifest", str(empty), "--output", str(tmp_path / "raw")]) == 2
    assert "no audio" in capsys.readouterr().err
