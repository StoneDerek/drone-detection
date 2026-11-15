import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from unittest.mock import patch

import pytest

from drone_detector import corpus_intake as intake


ROOT = Path(__file__).resolve().parents[1]


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


def item(payload=b"original distribution"):
    return {"source_id": "source", "filename": "original.zip", "bytes": len(payload),
            "url": "https://example.invalid/original.zip", "profiles": ["core"],
            "md5": hashlib.md5(payload).hexdigest(), "sha256": None,
            "checksum_origin": "publisher"}


def stage_prefix(output, source, payload):
    directory = output / source["source_id"]
    directory.mkdir()
    partial = directory / (source["filename"] + ".part")
    partial.write_bytes(payload)
    intake.write_json(directory / (source["filename"] + ".download.json"),
                      {"item_identity": intake.identity(source)})
    return partial


def test_real_manifest_has_unique_pinned_outputs_and_complete_selection():
    manifest = json.loads((ROOT / "data/corpus-intake-v3.json").read_text())
    rows = intake.select_items(manifest, "core")
    assert len(rows) == 222
    assert sum(row["bytes"] for row in rows) == 39_010_710_391
    assert {r["source_id"] for r in rows} == {
        "dnc", "dronenoise-edzell-v3", "idmt-berne-2022", "neaptide-v1", "poland-uav-v1",
        "tut2017-development", "tut2017-evaluation", "urbansound8k"}
    assert all(r["md5"] and r["checksum_origin"] == "publisher" for r in rows)
    assert {id(r) for r in rows} == {
        id(r) for p in ("field", "background") for r in intake.select_items(manifest, p)}


def test_resume_rejects_wrong_range_before_reading_or_changing_prefix(tmp_path):
    source = item()
    partial = stage_prefix(tmp_path, source, b"orig")

    class NoBody(Response):
        def read(self, *_):
            pytest.fail("Invalid resume headers must be rejected before reading a body")

    with patch.object(intake.urllib.request, "urlopen",
                      return_value=NoBody(b"", 206, {"Content-Range": "bytes 0-20/21"})):
        with pytest.raises(ValueError, match="wrong resume range"):
            intake.acquire(source, tmp_path)
    assert partial.read_bytes() == b"orig"


def test_ignored_range_restarts_and_complete_pending_state_recovers(tmp_path):
    payload = b"original distribution"
    source = item(payload)
    stage_prefix(tmp_path, source, payload[:4])
    with patch.object(intake.urllib.request, "urlopen", return_value=Response(payload)) as fetch:
        result = intake.acquire(source, tmp_path)
        assert not result["reused"] and result["resumed_bytes"] == 0
        assert fetch.call_args[0][0].get_header("Range") == "bytes=4-20"
        directory = tmp_path / "source"
        (directory / "original.zip.complete.json").rename(directory / "original.zip.pending.json")
        reused = intake.acquire(source, tmp_path)
        assert reused["reused"] and fetch.call_count == 1
        assert reused["publisher_checksum_verified"]


def test_truncated_transfer_stays_unverified_and_corrupt_prefix_cannot_complete(tmp_path):
    source = item()
    with patch.object(intake.urllib.request, "urlopen", return_value=Response(b"orig")):
        with pytest.raises(ValueError, match="truncated"):
            intake.acquire(source, tmp_path)
    partial = tmp_path / "source/original.zip.part"
    assert partial.read_bytes() == b"orig"
    partial.write_bytes(b"BAD!")
    with patch.object(intake.urllib.request, "urlopen", return_value=Response(
            b"inal distribution", 206, {"Content-Range": "bytes 4-20/21"})):
        with pytest.raises(ValueError, match="md5 differs"):
            intake.acquire(source, tmp_path)
    assert not (tmp_path / "source/original.zip").exists()
    assert not list(tmp_path.rglob("*.complete.json"))


def test_changed_item_or_unrecognized_local_file_is_preserved(tmp_path):
    source = item()
    partial = stage_prefix(tmp_path, source, b"orig")
    with pytest.raises(ValueError, match="item changed"):
        intake.acquire({**source, "url": "https://example.invalid/new"}, tmp_path)
    assert partial.read_bytes() == b"orig"
    partial.rename(partial.with_name("original.zip"))
    with pytest.raises(ValueError, match="no recognized completion"):
        intake.acquire(source, tmp_path)


def test_sigkill_then_new_process_resumes_http_bytes_and_reuses_completion(tmp_path):
    payload = bytes(range(256)) * 32768
    source = item(payload)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            value = self.headers.get("Range")
            requests.append(value)
            start = int(value.split("=")[1].split("-")[0]) if value else 0
            self.send_response(206 if value else 200)
            self.send_header("Content-Length", str(len(payload) - start))
            if value:
                self.send_header("Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}")
            self.end_headers()
            try:
                for offset in range(start, len(payload), 65536):
                    self.wfile.write(payload[offset:offset + 65536])
                    self.wfile.flush()
                    if not value:
                        time.sleep(.01)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Route the test source through an actual local HTTP server. Production still
    # accepts HTTPS sources only; this test adapter runs only in the child process.
    program = """
import json, sys, urllib.request
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from drone_detector import corpus_intake as intake
open_http = urllib.request.urlopen
def local(request, **kwargs):
    request = urllib.request.Request(sys.argv[4], headers=dict(request.header_items()))
    return open_http(request, **kwargs)
intake.urllib.request.urlopen = local
print(json.dumps(intake.acquire(json.loads(sys.argv[2]), Path(sys.argv[3]))), flush=True)
"""
    command = [sys.executable, "-c", program, str(ROOT / "src"), json.dumps(source),
               str(tmp_path), f"http://127.0.0.1:{server.server_port}/original.zip"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        partial = tmp_path / "source/original.zip.part"
        deadline = time.monotonic() + 10
        while not (partial.exists() and partial.stat().st_size >= 1024 * 1024):
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("Child did not reach a recoverable partial transfer")
            time.sleep(.01)
        contender = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert contender.returncode != 0 and "Another downloader is active" in contender.stderr
        assert len(requests) == 1
        process.kill()
        process.communicate(timeout=10)
        offset = partial.stat().st_size
        assert 0 < offset < len(payload)
        second = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        assert json.loads(second.stdout.splitlines()[-1])["resumed_bytes"] == offset
        assert requests[1] == f"bytes={offset}-{len(payload)-1}"
        assert (tmp_path / "source/original.zip").read_bytes() == payload
        third = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        assert json.loads(third.stdout.splitlines()[-1])["reused"]
        assert len(requests) == 2
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        server.shutdown()
        server.server_close()
