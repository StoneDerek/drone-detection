"""Small JSON/hash helpers. No executable model serialization."""

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row).decode() + "\n" for row in rows))


def revision():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    files = sorted((root / "src").rglob("*.py"))
    return {
        "git_head": result.stdout.strip() if result.returncode == 0 else None,
        "source_sha256": digest({str(p.relative_to(root)): file_hash(p) for p in files}),
    }


def hardware():
    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {"cpu": cpu, "platform": platform.platform(), "python": platform.python_version()}


def versions():
    return {p: version(p) for p in ["numpy", "scipy", "scikit-learn", "soundfile"]}


def union_intervals(intervals, limit=None):
    result = []
    for start, end in sorted(intervals):
        if limit is not None:
            start, end = max(0.0, start), min(limit, end)
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([float(start), float(end)])
    return result


def overlap(start, end, intervals):
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value)
