#!/usr/bin/env python3
"""Audit one fixed native recorder view per WSU flight; no labels are assigned."""

import argparse
import ast
import hashlib
import io
import json
import re
import urllib.request
import wave
from pathlib import Path

from audit_dronenoise_headers import wav_header

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((ROOT / 'reports/corpus-inventory-v2/wsu-provenance.json').read_text())
    rows = []
    for record, document in zip(provenance['recordings'], provenance['documents'][2:], strict=True):
        url = document['url'].rsplit('/', 1)[0] + '/' + record['wav_files'][0]
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-65535', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=30) as response:
            extent = re.fullmatch(r'bytes 0-65535/(\d+)', response.headers.get('Content-Range', ''))
            if response.status != 206 or extent is None:
                raise ValueError('Source did not honor the bounded header request')
            prefix = response.read(65537)
            modified = response.headers.get('Last-Modified')
        if len(prefix) != 65536:
            raise ValueError('Wrong header prefix length')
        # Diagnostic allowance reports the incomplete frame explicitly. It does
        # not repair source bytes or make the source eligible for training.
        facts = wav_header(prefix, int(extent[1]), allow_partial_frame=True)
        with wave.open(io.BytesIO(prefix)) as reader:
            if reader.getnframes() != facts['complete_frames']:
                raise ValueError('Independent stdlib WAV frame count disagrees')
        rows.append({'url': url, 'capture_group': record['conservative_group'],
                     'date': record['date'], 'flight_id': record['flight_id'],
                     'source_bytes': int(extent[1]), 'last_modified': modified,
                     'prefix_sha256': hashlib.sha256(prefix).hexdigest(),
                     'strict_header_accepted': facts['partial_frame_bytes'] == 0, **facts})
        print(record['date'], record['flight_id'], facts['duration_seconds'], 'seconds;',
              facts['partial_frame_bytes'], 'trailing frame bytes', flush=True)
    (args.output / 'fixed-view-headers.json').write_text(json.dumps(rows, indent=2) + '\n')
    code_url = 'https://rds.westernsydney.edu.au/Centres/ICNS/2026/Jahir_Rodriguez/code/dataset_creation/create_dataset.py'
    with urllib.request.urlopen(code_url, timeout=30) as response:
        code = response.read(100001)
    if len(code) > 100000:
        raise ValueError('Unexpectedly large generation script')
    tree = ast.parse(code)
    report = {'url': code_url, 'sha256': hashlib.sha256(code).hexdigest(),
              'skip_seconds_end_references': [node.lineno for node in ast.walk(tree)
                  if isinstance(node, ast.Constant) and node.value == 'skip_seconds_end'],
              'scope': 'Static source inspection only; no external code executed and no paper results regenerated'}
    (args.output / 'generator-inspection.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
