import argparse
import json
import sys
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

from .common import write_json, write_jsonl
from .data import audit, split_manifest
from .demo import build_demo
from .evaluation import evaluate, tune
from .experiments import run_synthetic
from .model import load_model, train
from .stream import StreamDetector, infer_file
from .audio import decode
from .synthetic import generate_corpus


def parser():
    root = argparse.ArgumentParser(description="Experimental acoustic detector; authentic reliability unmeasured")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("audit", help="Inventory audio without guessing labels")
    p.add_argument("--root", required=True)
    p.add_argument("--annotations")
    p.add_argument("--output", required=True)
    p = commands.add_parser("split", help="Group recordings before any window generation")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--final-source", action="append", default=[])
    p = commands.add_parser("train")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--representation", choices=["absolute_mel", "relative_mel", "relative_harmonic"], default="absolute_mel")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gain-augment", action="store_true")
    p = commands.add_parser("tune", help="Choose threshold and alarm logic on development ONLY")
    p.add_argument("--bundle", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--bundle", nargs="+", required=True, help="All frozen comparison arms in one final batch")
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["development", "final"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-final-reuse", action="store_true", help="Explicitly downgrade a reused final set to development evidence")
    p = commands.add_parser("infer")
    p.add_argument("--bundle", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("stream", help="Replay a recording in canonical-rate blocks; NOT microphone capture")
    p.add_argument("--bundle", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--realtime", action="store_true", help="Pace replay at wall-clock rate and print scores as they arrive")
    p = commands.add_parser("demo", help="Self-contained HTML with audio, scores and alarm state")
    p.add_argument("--bundle", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("synthetic")
    p.add_argument("--output", required=True)
    p.add_argument("--sessions", type=int, default=10)
    p = commands.add_parser("experiment", help="Run disclosed controlled synthetic experiments, never a real reliability benchmark")
    p.add_argument("--output", required=True)
    p.add_argument("--contract", default="contracts/evaluation-v1.json")
    p.add_argument("--sessions", type=int, default=10)
    p = commands.add_parser("train-weak", help="Train multiple-instance presence detector from publisher clip labels")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--representation", choices=["absolute_mel", "relative_mel", "relative_harmonic", "yamnet", "yamnet_rms"], default="relative_mel")
    p.add_argument("--encoder")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--regularization", type=float, default=1.0)
    p = commands.add_parser("tune-weak", help="Development-only recording-label threshold selection")
    p.add_argument("--bundle", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument('--policy', choices=['v1', 'sensitivity-v2'], default='v1')
    p = commands.add_parser("evaluate-weak", help="Recording-level assessment; never invents event timing truth")
    p.add_argument("--bundle", nargs="+", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["development", "final"], required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("serve", help="Local audio upload, score timeline and recorded replay product")
    p.add_argument("--bundle", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--manifest")
    p.add_argument("--selection")
    p = commands.add_parser('lab', help='Compare trained models and inspect genuine CNN Grad-CAM')
    p.add_argument('--catalog', required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    return root


def paced_stream(model, path):
    initialization_started = time.perf_counter()
    if hasattr(model, 'warmup'):
        model.warmup()
    elif model.encoder is not None:
        import numpy as np
        model.score(np.zeros(model.config.window_samples, dtype=np.float32))
    initialization_seconds = time.perf_counter() - initialization_started
    audio, info = decode(path, model.config.sample_rate)
    stream = StreamDetector(model)
    started = time.perf_counter()
    rows = []
    for start in range(0, len(audio), 1600):
        block = audio[start:start + 1600]
        deadline = started + (start + len(block)) / model.config.sample_rate
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)  # At most 100 ms per delivery, not a blocking long wait.
        for row in stream.push(block):
            row["wall_delivery_seconds"] = time.perf_counter() - started
            print(json.dumps(row), flush=True)
            rows.append(row)
    for row in stream.flush():
        row["wall_delivery_seconds"] = time.perf_counter() - started
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows, {**info, "mode": "wall-clock recorded replay, not live capture",
                  "elapsed_seconds": time.perf_counter() - started, "model_id": model.model_id,
                  "model_initialization_seconds_before_playback": initialization_seconds}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        with threadpool_limits(limits=1):
            if args.command == "audit":
                result = audit(args.root, args.output, args.annotations)
            elif args.command == "split":
                result = split_manifest(args.manifest, args.output, args.seed, args.final_source)
            elif args.command == "train":
                model = train(args.manifest, args.output, args.representation, args.seed, args.gain_augment)
                result = {"model_id": model.model_id, "status": "trained; tune before inference"}
            elif args.command == "tune":
                model = tune(args.bundle, args.manifest, args.contract, args.output)
                result = {"model_id": model.model_id, "decision": model.metadata["decision"]}
            elif args.command == "evaluate":
                report = evaluate(args.bundle, args.manifest, args.split, args.output, args.allow_final_reuse)
                result = {"split": report["split"], "final_reused": report["final_reused"],
                          "results": [{"model_id": r["model_id"], "summary": r["summary"],
                                       "assessment": r["assessment"]} for r in report["results"]]}
            elif args.command in {"infer", "stream"}:
                if Path(args.output).exists():
                    raise ValueError("Refusing to overwrite inference trace")
                model = load_model(args.bundle, require_frozen=True)
                if args.command == "stream" and args.realtime:
                    rows, timing = paced_stream(model, args.audio)
                else:
                    rows, timing = infer_file(model, args.audio)
                write_jsonl(args.output, rows)
                write_json(Path(args.output).with_suffix(".runtime.json"), timing)
                result = {"windows": len(rows), "timing": timing}
            elif args.command == "demo":
                result = build_demo(args.bundle, args.manifest, args.selection, args.output)
            elif args.command == "synthetic":
                result = {"annotations": str(generate_corpus(args.output, args.sessions))}
            elif args.command == "serve":
                from .web import serve
                serve(args.bundle, args.host, args.port, args.manifest, args.selection)
                return 0
            elif args.command == 'lab':
                from .lab import serve_lab
                serve_lab(args.catalog, args.host, args.port)
                return 0
            elif args.command == "train-weak":
                from .weak import train_weak
                model = train_weak(args.manifest, args.output, args.representation,
                                   args.seed, args.regularization, args.encoder)
                result = {"model_id": model.model_id, "status": "weak-trained; tune on development"}
            elif args.command == "tune-weak":
                from .weak import tune_weak
                model = tune_weak(args.bundle, args.manifest, args.output, args.policy)
                result = {"model_id": model.model_id, "decision": model.metadata["decision"],
                          "development": model.metadata["selection"]["chosen"]["metrics"]}
            elif args.command == "evaluate-weak":
                from .weak import evaluate_weak
                report = evaluate_weak(args.bundle, args.manifest, args.split, args.output)
                result = {"split": report["split"], "results": [r["summary"] for r in report["results"]]}
            else:
                report = run_synthetic(args.output, args.contract, args.sessions)
                result = {"selected": report["selected"], "scope": report["scope"],
                          "claim": report["claim"], "seconds": report["seconds"]}
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
