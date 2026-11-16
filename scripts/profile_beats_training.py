#!/usr/bin/env python3
"""Measure full-backbone training cost; no detector performance is estimated."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from drone_detector.beats_training import build_model, configure, optimizer_for, train_step
from drone_detector.common import file_hash, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    configure(args.device, threads=args.threads)
    recipe = json.loads(args.plan.read_text())["recipe"]["fine_tuning"]
    model, identity = build_model(args.assets, args.head, args.device)
    optimizer = optimizer_for(model, recipe)
    rng = np.random.default_rng(42)
    x = torch.tensor(
        rng.normal(0, 0.01, (args.batch_size, 48000)), dtype=torch.float32, device=args.device
    )
    y = torch.tensor(
        [i % 2 for i in range(args.batch_size)], dtype=torch.float32, device=args.device
    )
    mass = torch.ones(args.batch_size, device=args.device)
    before = model.encoder.patch_embedding.weight.detach().clone()
    seconds = []
    for _ in range(3):
        start = time.monotonic()
        loss, grad = train_step(model, optimizer, x, y, mass, args.batch_size)
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.monotonic() - start)
        print("Full encoder training step", seconds[-1], "seconds", flush=True)
    changed = float(torch.max(torch.abs(before - model.encoder.patch_embedding.weight)).detach())
    if changed == 0:
        raise ValueError("Encoder did not update")
    result = dict(
        scope="Synthetic training-mechanism and throughput check only; not a drone evaluation",
        native_input_seconds=3,
        batch_size=args.batch_size,
        device=args.device,
        threads=args.threads,
        step_seconds=seconds,
        encoder_patch_max_abs_update=changed,
        seconds_per_example=float(np.median(seconds[1:]) / args.batch_size),
        trainable_encoder_parameters=sum(
            p.numel() for p in model.encoder.parameters() if p.requires_grad
        ),
        peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        plan_sha256=file_hash(args.plan),
        base_encoder=identity,
    )
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
