from pathlib import Path

import pytest
from threadpoolctl import threadpool_limits

from drone_detector.data import audit, load_manifest, split_manifest
from drone_detector.evaluation import tune
from drone_detector.model import train
from drone_detector.synthetic import generate_corpus


@pytest.fixture(scope="session", autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled")
    annotations = generate_corpus(root / "corpus", sessions_per_source=5)
    inventory, manifest = root / "inventory.jsonl", root / "manifest.jsonl"
    audit(root / "corpus", inventory, annotations)
    split_manifest(inventory, manifest, final_sources=["synthetic-generator-2"])
    raw, frozen = root / "raw", root / "frozen"
    train(manifest, raw, "relative_mel", seed=42)
    contract = Path(__file__).resolve().parents[1] / "contracts/evaluation-v1.json"
    model = tune(raw, manifest, contract, frozen)
    return {"root": root, "manifest": manifest, "model": model, "raw": raw,
            "bundle": frozen, "rows": load_manifest(manifest, True), "contract": contract}
