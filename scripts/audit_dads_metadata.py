"""Read every DADS path/label while excluding its embedded audio-bytes column."""

import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import sys

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--work', type=Path, default=Path('work/dads-source-census-v1'))
PARSER.add_argument('--workers', type=int, default=4, choices=range(1, 5))
ARGS = PARSER.parse_args()
ROOT = ARGS.work.resolve()
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
sys.path.insert(0, str(REPO / 'scripts'))
import pyarrow  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from ddl_audio import RangeReader  # noqa: E402

META = json.loads((ROOT / 'dataset.json').read_text())
FILES = json.loads((ROOT / 'data-tree.json').read_text())
if META['sha'] != '981b832c35b45a57518c989f0f79101adb4ae91f':
    raise ValueError('Different dataset revision; preserve this audit and use a new protocol')
if len(FILES) != 39 or sum(item['size'] for item in FILES) != 6809972235:
    raise ValueError('Pinned shard catalogue changed')
if any(not re.fullmatch(r'data/train-\d{5}-of-00039\.parquet', item['path']) for item in FILES):
    raise ValueError('Unreviewed source path in the metadata catalogue')


def inspect(item):
    url = (f'https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples/'
           f'resolve/{META["sha"]}/{item["path"]}')
    reader = RangeReader(url, item['size'], block_size=65536,
                         cache_dir=ROOT / 'ranges' / Path(item['path']).stem)
    parquet = pq.ParquetFile(reader, pre_buffer=False)
    physical = [parquet.metadata.row_group(0).column(i).path_in_schema
                for i in range(parquet.metadata.num_columns)]
    if physical != ['audio.bytes', 'audio.path', 'label']:
        raise ValueError('Unreviewed physical schema; audit any additional provenance fields')
    table = parquet.read(columns=['audio.path', 'label'], use_threads=False)
    if table.column('audio').type.names != ['path'] or table.column_names != ['audio', 'label']:
        raise ValueError('Unreviewed projection: embedded audio bytes must be excluded')
    rows = [dict(shard=item['path'], row=i, path=v['audio']['path'], label=v['label'])
            for i, v in enumerate(table.to_pylist())]
    path = ROOT / 'path-labels' / (Path(item['path']).stem + '.jsonl')
    path.parent.mkdir(exist_ok=True)
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    return dict(shard=item['path'], rows=len(rows), row_groups=parquet.metadata.num_row_groups,
                physical_columns=physical,
                transferred_bytes_this_run=reader.transferred,
                projected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                published_lfs_sha256=item['lfs']['oid'], whole_parquet_checksum_verified=False)


def main():
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=ARGS.workers) as pool:
        tasks = [pool.submit(inspect, item) for item in FILES]
        for task in concurrent.futures.as_completed(tasks):
            result = task.result()
            outcomes.append(result)
            (ROOT / 'shard-progress.json').write_text(json.dumps(outcomes, indent=2) + '\n')
            print(json.dumps({'completed_shards': len(outcomes), 'rows': result['rows'],
                              'range_bytes': result['transferred_bytes_this_run']}), flush=True)
    rows = []
    for path in sorted((ROOT / 'path-labels').glob('*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    if len(rows) != 180320 or collections.Counter(r['label'] for r in rows) != {0: 16729, 1: 163591}:
        raise ValueError('Complete pinned census does not match the advertised release')
    names = [r['path'] for r in rows]
    generic, consistent = 0, 0
    ids = collections.defaultdict(set)
    for row in rows:
        match = re.fullmatch(r'(no-drone|drone)-(\d+)\.wav', row['path'] or '')
        if match:
            generic += 1
            consistent += (match[1] == 'drone') == bool(row['label'])
            ids[row['label']].add(int(match[2]))
    result = dict(
        dataset_revision=META['sha'], shards=len(outcomes), rows=len(rows),
        label_counts=dict(collections.Counter(r['label'] for r in rows)),
        null_path_values=sum(name is None for name in names), unique_literal_paths=len(set(names)),
        class_plus_ordinal_path_values=generic, class_prefix_consistent_with_label=consistent,
        ordinal_ranges={label: dict(min=min(values), max=max(values), distinct=len(values))
                        for label, values in ids.items()},
        projected_fields=['audio.path', 'label'], embedded_audio_bytes_projected=False,
        extracted_or_decoded_audio_members=0, detector_scores_generated=0,
        whole_parquet_checksums_verified=False, pyarrow_version=pyarrow.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        range_reader_sha256=hashlib.sha256((REPO / 'scripts/ddl_audio.py').read_bytes()).hexdigest(),
        cached_range_payload_bytes=sum(p.stat().st_size for p in (ROOT / 'ranges').rglob('*.bin')),
        new_range_bytes_this_run=sum(v['transferred_bytes_this_run'] for v in outcomes),
        input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in ['dataset.json', 'data-tree.json', 'README.md']},
        shard_receipts=sorted(outcomes, key=lambda v: v['shard']),
        scope='Complete path/label metadata projection at a pinned revision. HTTP range-cache blocks can contain incidental compressed audio bytes; no embedded audio field is returned or decoded. Literal filenames do not prove physical recording independence.',
    )
    (ROOT / 'census.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in {'shard_receipts', 'input_sha256'}}), flush=True)


if __name__ == '__main__':
    main()
