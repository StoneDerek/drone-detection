#!/usr/bin/env python3
"""Read a public Drive corpus listing and acquire every checksum-pinned source.

This does not mount Drive, need account credentials, assign labels, or treat a
partial folder listing as a corpus. Native publisher checksums remain the
acceptance criterion. Public HTML is an access route, not a stable Drive API.
"""

import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from drone_detector.common import read_json, write_json
from drone_detector.corpus_intake import acquire

ROOT = Path(__file__).resolve().parents[1]


class FolderParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.current = None
        self.title_depth = None
        self.rows = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "div":
            self.depth += 1
            if "flip-entry" in attrs.get("class", "").split() and attrs.get("id", "").startswith(
                "entry-"
            ):
                self.current = {
                    "id": attrs["id"][6:],
                    "name": "",
                    "folder": False,
                    "depth": self.depth,
                }
            if self.current and "flip-entry-title" in attrs.get("class", "").split():
                self.title_depth = self.depth
        if tag == "a" and self.current and "/drive/folders/" in attrs.get("href", ""):
            self.current["folder"] = True

    def handle_data(self, data):
        if self.current and self.title_depth is not None:
            self.current["name"] += data

    def handle_endtag(self, tag):
        if tag != "div":
            return
        if self.title_depth == self.depth:
            self.title_depth = None
        if self.current and self.current["depth"] == self.depth:
            self.current.pop("depth")
            self.rows.append(self.current)
            self.current = None
        self.depth -= 1


def listing(identifier):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("Invalid public Drive folder identifier")
    url = "https://drive.google.com/embeddedfolderview?id=" + identifier
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        parser = FolderParser()
        parser.feed(response.read().decode("utf-8"))
    if not parser.rows:
        raise ValueError("Public Drive folder is empty or its listing is unavailable")
    return parser.rows


def source_folder(identifier):
    rows = listing(identifier)
    names = {r["name"]: r for r in rows if r["folder"]}
    if "corpus-intake-v3" in names:
        return source_folder(names["corpus-intake-v3"]["id"])
    if "source-archives" in names:
        return names["source-archives"]["id"]
    if set(read_json(ROOT / "data/corpus-intake-v3.json")["source_releases"]) <= set(names):
        return identifier
    raise ValueError(
        "Expected corpus-intake-v3/source-archives; refusing an unrelated or incomplete folder"
    )


def mapped_items(identifier):
    manifest = read_json(ROOT / "data/corpus-intake-v3.json")
    sources = {r["name"]: r for r in listing(source_folder(identifier)) if r["folder"]}
    located = {}
    for source in manifest["source_releases"]:
        if source not in sources:
            raise ValueError("Missing source folder: " + source)
        entries = listing(sources[source]["id"])
        for entry in entries:
            key = (source, entry["name"])
            if key in located:
                raise ValueError("Ambiguous public Drive filename")
            located[key] = entry
    missing = [
        (r["source_id"], r["filename"])
        for r in manifest["items"]
        if (r["source_id"], r["filename"]) not in located
    ]
    if missing:
        raise ValueError("Public folder lacks pinned distributions: " + repr(missing))
    items = []
    for original in manifest["items"]:
        entry = located[(original["source_id"], original["filename"])]
        if entry["folder"]:
            raise ValueError("A pinned distribution is a folder")
        items.append(
            {
                **original,
                "publisher_url": original["url"],
                "public_drive_file_id": entry["id"],
                "acquisition_route": "user-public-drive-complete-embedded-listing",
                "url": f"https://drive.usercontent.google.com/download?id={entry['id']}&export=download&confirm=t",
            }
        )
    return items


def validate_plan(plan):
    expected = {
        (r["source_id"], r["filename"]): r
        for r in read_json(ROOT / "data/corpus-intake-v3.json")["items"]
    }
    observed = [(r["source_id"], r["filename"]) for r in plan["items"]]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError("Saved acquisition plan is incomplete or repeats distributions")
    for item in plan["items"]:
        original = expected[(item["source_id"], item["filename"])]
        if any(
            item.get("publisher_url" if key == "url" else key) != value
            for key, value in original.items()
        ):
            raise ValueError("Saved acquisition plan differs from pinned publisher metadata")


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "public-drive-acquisition.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["folder_id"] != args.folder_id:
            raise ValueError("Acquisition folder changed; preserve the old intake directory")
    else:
        plan = {"folder_id": args.folder_id, "items": mapped_items(args.folder_id)}
        write_json(plan_path, plan)
    validate_plan(plan)
    print(
        f"Pinned sources: {len(plan['items'])}, bytes: {sum(i['bytes'] for i in plan['items']):,}",
        flush=True,
    )
    if args.inventory_only:
        return

    def job(item):
        for attempt in range(3):
            try:
                result = acquire(item, args.output / "archives")
                print(
                    json.dumps({"verified": item["source_id"] + "/" + item["filename"]}), flush=True
                )
                return result
            except Exception as error:
                if attempt == 2:
                    raise RuntimeError("Acquisition failed: " + item["filename"]) from error
                print(json.dumps({"retry": item["filename"], "reason": str(error)}), flush=True)
                time.sleep(2**attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(job, plan["items"]))
    write_json(args.output / "acquisition-complete.json", {"distributions": results})
    print("Every distribution checksum verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--inventory-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
