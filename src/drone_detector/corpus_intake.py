"""Resumable, checksummed acquisition of original corpus distribution files.

This preserves acquisition state only. It does not assign labels or partitions.
Use one active writer per output directory, including across notebook runtimes.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request
import uuid


def validate_item(item: dict) -> None:
    for key in ("source_id", "filename"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", item[key]):
            raise ValueError(f"Unsafe corpus {key}")
    if type(item["bytes"]) is not int or item["bytes"] <= 0:
        raise ValueError("Invalid corpus file size")
    if not item["url"].startswith("https://"):
        raise ValueError("Corpus URLs must use HTTPS")
    if not (item.get("md5") or item.get("sha256")):
        raise ValueError("A pinned distribution checksum is required")
    for key, length in (("md5", 32), ("sha256", 64)):
        if item.get(key) is not None and not re.fullmatch(rf"[0-9a-f]{{{length}}}", item[key]):
            raise ValueError(f"Invalid corpus {key}")


def identity(item: dict) -> str:
    validate_item(item)
    payload = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    payload = json.dumps(data, sort_keys=True, indent=2, allow_nan=False).encode()
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
        os.replace(temporary, path)
        if path.read_bytes() != payload:
            raise ValueError("Acquisition state read-back failed")
    finally:
        temporary.unlink(missing_ok=True)


def verify(path: Path, item: dict) -> dict:
    if path.stat().st_size != item["bytes"]:
        raise ValueError("Source size differs from the pinned distribution")
    hashes = {key: hashlib.new(key) for key in ("md5", "sha256")}
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            for checksum in hashes.values():
                checksum.update(chunk)
    observed = {key: checksum.hexdigest() for key, checksum in hashes.items()}
    for key in hashes:
        if item.get(key) is not None and item[key] != observed[key]:
            raise ValueError(f"Source {key} differs from the pinned distribution; preserve for review")
    return observed


def acquire(item: dict, output: Path) -> dict:
    """Serialize writers on this runtime; retain the cross-runtime one-writer rule."""
    validate_item(item)
    lock_directory = Path(tempfile.gettempdir()) / "drone-corpus-intake-locks"
    lock_directory.mkdir(exist_ok=True)
    destination = (Path(output) / item["source_id"] / item["filename"]).resolve()
    lock_name = hashlib.sha256(str(destination).encode()).hexdigest() + ".lock"
    # Keep the lock on the runtime's local filesystem, rather than assuming that
    # a Drive mount implements POSIX locking across machines. Do not unlink it:
    # concurrent processes must continue to address the same lock inode.
    with (lock_directory / lock_name).open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another downloader is active for this source on this runtime") from error
        return _acquire(item, output)


def _acquire(item: dict, output: Path) -> dict:
    """Resume a partial file, then verify stored bytes before publishing completion.

    Final checksums detect a modified prefix or changed server object, even when
    no HTTP validator is provided. A range-ignoring server restarts that file.
    A partial file is never treated as accepted data or a verified source.
    """
    item_identity = identity(item)
    directory = Path(output) / item["source_id"]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / item["filename"]
    partial = target.with_name(target.name + ".part")
    journal = target.with_name(target.name + ".download.json")
    pending = target.with_name(target.name + ".pending.json")
    complete = target.with_name(target.name + ".complete.json")
    states = {}
    for name, path in (("journal", journal), ("pending", pending), ("complete", complete)):
        if path.exists():
            state = json.loads(path.read_text())
            if state.get("item_identity") != item_identity:
                raise ValueError("Corpus intake item changed; use a new intake directory")
            states[name] = state

    if target.exists():
        previous = states.get("complete", states.get("pending"))
        if previous is None:
            raise ValueError("Existing source has no recognized completion state; preserve for review")
        observed = verify(target, item)
        if any(observed[key] != previous.get(key) for key in observed):
            raise ValueError("Stored source differs from its completion receipt")
        write_json(complete, previous)
        pending.unlink(missing_ok=True)
        journal.unlink(missing_ok=True)
        return {**previous, "reused": True, "resumed_bytes": 0}

    if "complete" in states:
        raise ValueError("Completed source is missing; preserve its receipt for review")
    if partial.exists() and "journal" not in states:
        raise ValueError("Partial source has no recognized journal; preserve for review")
    start = partial.stat().st_size if partial.exists() else 0
    if start > item["bytes"]:
        raise ValueError("Partial source exceeds pinned size; preserve for review")
    if "journal" not in states:
        write_json(journal, {"item_identity": item_identity, "filename": item["filename"],
                             "bytes": item["bytes"], "status": "unverified partial download"})
    resumed_bytes = start
    if start < item["bytes"]:
        headers = {"Accept-Encoding": "identity"}
        if start:
            headers["Range"] = f"bytes={start}-{item['bytes'] - 1}"
        request = urllib.request.Request(item["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            if start and response.status == 206:
                expected = f"bytes {start}-{item['bytes'] - 1}/{item['bytes']}"
                if response.headers.get("Content-Range") != expected:
                    raise ValueError("Server returned the wrong resume range")
            elif response.status == 200:
                if response.headers.get("Content-Range"):
                    raise ValueError("Ambiguous full-file response")
                if start:
                    print(f"{item['filename']}: server ignored Range; restarting this file", flush=True)
                start = 0
                resumed_bytes = 0
            else:
                raise ValueError("Unexpected download response; source was not modified")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != item["bytes"] - start:
                raise ValueError("Response length differs from the pinned distribution")
            print(f"{item['source_id']}/{item['filename']}: {start}/{item['bytes']} bytes", flush=True)
            count = start
            next_progress = start + 256 * 1024 * 1024
            with partial.open("ab" if start else "wb") as stream:
                while True:
                    chunk = response.read(min(1024 * 1024, item["bytes"] - count + 1))
                    if not chunk:
                        break
                    if count + len(chunk) > item["bytes"]:
                        raise ValueError("Response exceeds the pinned distribution size")
                    stream.write(chunk)
                    stream.flush()
                    count += len(chunk)
                    if count >= next_progress:
                        print(f"{item['filename']}: {count / 1e9:.2f} GB", flush=True)
                        next_progress = count + 256 * 1024 * 1024
            if count != item["bytes"]:
                raise ValueError("Interrupted or truncated transfer; unverified partial retained")

    observed = verify(partial, item)
    result = {"item_identity": item_identity, "source_id": item["source_id"],
              "filename": item["filename"], "url": item["url"], "bytes": item["bytes"],
              **observed, "pinned_checksum_verified": True,
              "publisher_checksum_verified": item.get("checksum_origin") == "publisher",
              "status": "verified distribution file; labels and study split unassigned"}
    write_json(pending, result)
    if target.exists():
        raise ValueError("Another writer created this source; use one active runtime")
    os.replace(partial, target)
    write_json(complete, result)
    pending.unlink(missing_ok=True)
    journal.unlink(missing_ok=True)
    return {**result, "reused": False, "resumed_bytes": resumed_bytes}


def select_items(manifest: dict, profile: str) -> list[dict]:
    items = [item for item in manifest["items"] if profile in item["profiles"]]
    if not items:
        raise ValueError("Unknown or empty corpus profile")
    paths = set()
    for item in items:
        validate_item(item)
        path = (item["source_id"], item["filename"])
        if path in paths:
            raise ValueError("Duplicate corpus output path")
        paths.add(path)
    return items
