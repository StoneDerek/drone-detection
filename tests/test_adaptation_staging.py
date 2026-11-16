import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from drone_detector.common import file_hash, read_jsonl, write_jsonl
from drone_detector.data import components


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_local_staging_preserves_bytes_and_refuses_conflicting_cache(tmp_path):
    p = tmp_path / "source.wav"
    sf.write(p, np.ones(4000, dtype=np.float32) * 0.1, 16000, subtype="FLOAT")
    rows = [
        dict(
            schema_version=1,
            id="one",
            path=str(p),
            source="fixture",
            source_version="1",
            label_evidence="synthetic staging contract",
            label="drone",
            origin="synthetic",
            annotation_status="unknown",
            events=[],
            ambiguous=[],
            duration_seconds=0.25,
            original_id="one",
            session_id="one",
            duplicate_review_required=False,
            sha256=file_hash(p),
            split="train",
        )
    ]
    rows[0]["group_id"] = components(rows)[0]
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    stage = script("stage_adaptation_audio.py").stage
    output = stage(manifest, tmp_path / "local")
    staged = read_jsonl(output)[0]
    assert Path(staged["path"]).read_bytes() == p.read_bytes()
    assert staged["group_id"] == rows[0]["group_id"]
    Path(staged["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="staged input differs"):
        stage(manifest, tmp_path / "local")


def test_original_archive_verification_uses_an_independent_view(tmp_path):
    source = tmp_path / "original/source"
    source.mkdir(parents=True)
    raw = b"original archive bytes"
    p = source / "archive.zip"
    p.write_bytes(raw)
    item = dict(
        source_id="source",
        filename=p.name,
        bytes=len(raw),
        md5=hashlib.md5(raw).hexdigest(),
        checksum_origin="publisher",
    )
    verify = script("verify_adaptation_archives.py").link_verified
    result = verify(item, tmp_path / "original", tmp_path / "view")
    assert result["publisher_checksum_verified"]
    assert p.read_bytes() == raw
    assert (tmp_path / "view/source/archive.zip").is_symlink()
    p.write_bytes(b"altered archive bytes!")
    with pytest.raises(ValueError):
        verify(item, tmp_path / "original", tmp_path / "view")
