#!/usr/bin/env python3
"""Check every declared available original/export pair using exact native audio."""

import argparse
import collections
import zipfile
from pathlib import Path

from drone_detector.common import file_hash, read_json, read_jsonl, write_json, write_jsonl
from drone_detector.fpv_lineage import ExactNativeIndex
from audit_fpv_lineage import read_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work/fpv-lineage-v1"))
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    summary = read_json(args.work / "summary.json")
    if file_hash(args.archive) != summary["archive"]["sha256"]:
        raise ValueError("The previously verified source archive changed")
    groups = collections.defaultdict(list)
    outcomes = []
    for row in read_jsonl(args.work / "lineage.jsonl"):
        if len(row["candidate_parents"]) == 1:
            groups[row["candidate_parents"][0]].append(row)
        else:
            outcomes.append(dict(member=row["member"], status="original_not_uniquely_available"))
    with zipfile.ZipFile(args.archive) as archive:
        for count, (member, exports) in enumerate(sorted(groups.items()), 1):
            try:
                source, rate, metadata = read_audio(archive, member)
                index = ExactNativeIndex(source)
            except (RuntimeError, ValueError) as error:
                outcomes.extend(dict(member=r["member"], parent=member, status="original_decode_failed", error=str(error)) for r in exports)
                continue
            for row in exports:
                try:
                    child, child_rate, info = read_audio(archive, row["member"])
                    match = index.match(child) if rate == child_rate else dict(status="different_sample_rates")
                    outcomes.append(dict(member=row["member"], parent=member, **match,
                                         source_sha256=metadata["sha256"], export_sha256=info["sha256"]))
                except (RuntimeError, ValueError) as error:
                    outcomes.append(dict(member=row["member"], parent=member, status="export_decode_failed", error=str(error)))
            if count % 25 == 0:
                print("Checked original/export families", count, "/", len(groups), flush=True)
    write_jsonl(args.work / "direct-export-to-original.jsonl", outcomes)
    result = dict(
        archive_sha256=summary["archive"]["sha256"],
        exports_accounted=len(outcomes), status_counts=dict(collections.Counter(r["status"] for r in outcomes)),
        source_families_checked=len(groups),
        scope="Unique exact native-rate/channel float32 decoded subsequence matching. No resampling, gain fitting or nearest-match acceptance; no detector inference.",
    )
    write_json(args.work / "direct-lineage-summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
