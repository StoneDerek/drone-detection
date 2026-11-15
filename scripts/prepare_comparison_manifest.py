#!/usr/bin/env python3
"""Prepare original-source catalogs/manifests without assigning an automatic split.

Example plan (paths relative to the plan file):
{"sources": [{"adapter": "urbansound8k", "metadata": "UrbanSound8K/metadata/UrbanSound8K.csv",
              "audio_root": "UrbanSound8K"},
             {"adapter": "tut2017", "release": "evaluation",
              "metadata_root": "TUT-acoustic-scenes-2017-evaluation",
              "audio_root": "TUT-acoustic-scenes-2017-evaluation"}]}

--catalog-only parses available metadata before audio downloads complete. The
catalog is not a trainable manifest. Full preparation hashes native files and
retains weak labels and unassigned splits unless --review supplies explicit
id -> {split, provenance_review} decisions for complete provenance components.
DNC is discovered from its extracted audio root. Generic per-file JSONL mappings
can supply other sources; no drone/speech labels are guessed from filenames.
Mono/stereo inputs select physical channel 0; channel_index can be specified per
source or per generic row. Arrays with more than two channels require an explicit
valid physical channel index. Source files remain unchanged; the probe runner
selects the declared channel at decoding.
"""

import argparse
from pathlib import Path

from drone_detector.common import read_json, write_jsonl
from drone_detector.source_adapters import metadata_plan, prepare_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve it and choose a new path")
    records = metadata_plan(read_json(args.plan), args.plan.resolve().parent)
    if args.catalog_only:
        if args.review:
            parser.error("--review applies only to full audio manifest preparation")
        write_jsonl(args.output, [{"catalog_schema_version": 1, **row} for row in records])
        print(f"Cataloged {len(records)} metadata rows; audio bytes/labels/splits are not verified")
    else:
        result = prepare_manifest(records, args.output,
                                  read_json(args.review) if args.review else None)
        print(f"Prepared {result['recordings']} native files in "
              f"{result['provenance_components']} conservative components")


if __name__ == "__main__":
    main()
