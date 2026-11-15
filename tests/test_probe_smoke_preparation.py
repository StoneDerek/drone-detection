"""Verify that the Colab software fixture retains its published weak-label scope."""

import importlib.util
from pathlib import Path

import pytest

from drone_detector.data import load_manifest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / 'scripts/prepare_frozen_probe_smoke.py'
    spec = importlib.util.spec_from_file_location('prepare_probe_smoke', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_retained_examples_preserve_grouping_weak_labels_and_repeatability(tmp_path):
    prepare = module()
    manifest = prepare.prepare_smoke(tmp_path)
    first = manifest.read_bytes()
    rows = load_manifest(manifest, require_split=True)
    assert len(rows) == 5
    training = [r for r in rows if r['split'] == 'train']
    development = [r for r in rows if r['split'] == 'development']
    assert len(training) == 3 and len(development) == 2
    assert len({r['group_id'] for r in training}) == 1
    assert not {r['group_id'] for r in training} & {r['group_id'] for r in development}
    assert all(r['annotation_status'] == 'weak' and not r['events'] for r in rows)
    assert all(r['derivative']['parent_audio_sha256'] != r['sha256'] for r in rows)
    assert prepare.prepare_smoke(tmp_path).read_bytes() == first


def test_changed_browser_audio_identity_is_rejected_before_preparation(tmp_path, monkeypatch):
    prepare = module()
    monkeypatch.setitem(prepare.EXAMPLE_SHA256, 'svanstrom-DRONE_001', '0' * 64)
    with pytest.raises(ValueError, match='fixture changed'):
        prepare.prepare_smoke(tmp_path)
    assert not (tmp_path / 'manifest.jsonl').exists()
