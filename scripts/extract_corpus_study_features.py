#!/usr/bin/env python3
"""Incremental features for all prepared study records; final data stays gated."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from drone_detector.common import digest, file_hash, read_json, read_jsonl, write_json


def extractor(family, assets, threads, device="cpu"):
    if family == "spectral":
        from drone_detector.audio import FeatureConfig, features

        cfg = FeatureConfig(representation="relative_harmonic")
        identity = {
            "family": "relative_harmonic",
            "frontend": cfg.to_dict(),
            "implementation_sha256": file_hash(
                Path(__file__).resolve().parents[1] / "src/drone_detector/audio.py"
            ),
        }
        return lambda audio: features(audio, cfg), identity
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(threads)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    if family == "beats":
        from drone_detector.beats_encoder import BEATsEncoder

        encoder = BEATsEncoder(assets, device=device)
    else:
        from drone_detector.clap_encoder import CLAPEncoder

        encoder = CLAPEncoder(assets, device=device)
    encoder.identity["torch_intraop_threads"] = threads
    if device == "cuda":
        encoder.identity["cuda_math"] = {"TF32": False, "deterministic_algorithms": True}
    return encoder.features, encoder.identity


def feature_record(row, output, compute, identity, implementation_sha256=None):
    specification = {
        "audio_sha256": row["sha256"],
        "encoder": identity,
        "sample_rate": 16000,
        "input": "prospective central3s crop, actual shorter signal passed to native frontend; no external gain/padding",
        "implementation_sha256": implementation_sha256 or file_hash(__file__),
    }
    key = digest(specification)
    path = output / (key + ".npz")
    receipt = path.with_suffix(".json")
    if path.exists() and receipt.exists():
        meta = read_json(receipt)
        if meta["specification"] != specification or file_hash(path) != meta["sha256"]:
            raise ValueError("Corrupt feature cache " + row["id"])
        with np.load(path, allow_pickle=False) as z:
            vector = z["vector"].copy()
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Bad cached vector")
        return key, True
    if file_hash(row["path"]) != row["sha256"]:
        raise ValueError("Audio changed " + row["id"])
    audio, rate = sf.read(row["path"], dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("Invalid canonical audio")
    vector = np.asarray(compute(audio), dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Nonfinite/invalid embedding")
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as f:
        np.savez_compressed(f, vector=vector)
    temporary.replace(path)
    temporary = receipt.with_suffix(".partial.json")
    write_json(
        temporary,
        {
            "specification": specification,
            "sha256": file_hash(path),
            "feature_dimension": len(vector),
        },
    )
    temporary.replace(receipt)
    return key, False


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    snapshot = output / "source-snapshots" / (source_sha + ".py.txt")
    snapshot.parent.mkdir(exist_ok=True)
    if snapshot.exists() and snapshot.read_bytes() != source:
        raise ValueError("Extractor source snapshot changed")
    snapshot.write_bytes(source)
    compute, identity = extractor(args.family, args.assets, args.threads, args.device)
    identity_path = output / "encoder.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Encoder changed")
    write_json(identity_path, identity)
    index_path = output / "index.json"
    index = read_json(index_path) if index_path.exists() else {}
    seen = set(index)
    started = time.monotonic()
    done = 0
    computed = 0
    reused_payloads = 0
    completion_seen = False
    while True:
        changed = False
        for part in sorted((args.prepared / "parts").glob("*.json")):
            for row in read_json(part)["records"]:
                if row["id"] in seen or row["split"] not in args.splits:
                    continue
                key, reused = feature_record(row, output, compute, identity, source_sha)
                computed += int(not reused)
                reused_payloads += int(reused)
                index[row["id"]] = {
                    "key": key,
                    "split": row["split"],
                    "source": row["source"],
                    "audio_sha256": row["sha256"],
                }
                seen.add(row["id"])
                done += 1
                changed = True
                if done % 50 == 0:
                    temporary = index_path.with_suffix(".partial.json")
                    write_json(temporary, index)
                    temporary.replace(index_path)
                    print(
                        json.dumps(
                            {
                                "family": args.family,
                                "completed": len(index),
                                "new_this_process": done,
                                "elapsed_seconds": time.monotonic() - started,
                                "last_source": row["source"],
                            }
                        ),
                        flush=True,
                    )
        if changed:
            temporary = index_path.with_suffix(".partial.json")
            write_json(temporary, index)
            temporary.replace(index_path)
        if not args.wait:
            break
        if (args.prepared / "preparation-complete.json").exists():
            # A long-running first pass may have snapshotted the part list before
            # preparation added its last containers. Scan again after the seal.
            if completion_seen:
                break
            completion_seen = True
            continue
        if not changed:
            time.sleep(10)
    if file_hash(__file__) != source_sha:
        raise ValueError("Extractor source changed during execution")
    if (args.prepared / "preparation-complete.json").exists():
        required = {
            r["id"]
            for r in read_jsonl(args.prepared / "manifest.jsonl")
            if r["split"] in args.splits
        }
        if required - seen:
            raise ValueError("Prepared parts did not cover every accepted requested record")
    print(
        json.dumps(
            {
                "family": args.family,
                "completed": len(index),
                "new_embeddings": computed,
                "reused_payloads": reused_payloads,
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--family", choices=["spectral", "beats", "clap"], required=True)
    p.add_argument("--assets", type=Path)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "development", "final"],
        default=["train", "development"],
    )
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()
    if "final" in args.splits and not (args.prepared / "final-evaluation-authorized.json").exists():
        p.error("Freeze every model/threshold before authorizing final feature extraction")
    run(args)


if __name__ == "__main__":
    main()
