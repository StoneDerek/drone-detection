#!/usr/bin/env python3
"""Create an isolated train/dev manifest using verified original-precision DDL."""

import hashlib
import os
from pathlib import Path

import numpy as np

from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, write_json, write_jsonl
from drone_detector.data import components, load_manifest


def main():
    original = Path('runs/real-v2/corpus/manifest-v2.jsonl').resolve()
    output = Path('runs/training-audit-v1/precision-corpus/manifest.jsonl').resolve()
    if output.exists():
        raise ValueError('Preserve the completed precision variant')
    proof = read_json('reports/training-audit-v1/source-conversion.json')
    assert proof['passed'] and proof['recovered_original_precision_files'] == 15
    replacements = {row['id']:row['recovered_original_precision'] for row in proof['converted']
                    if row['recovered_original_precision']}
    rows = [dict(row) for row in load_manifest(original,True) if row['split'] in {'train','development'}]
    for row in rows:
        source = original.parent/row['path']
        if row['id'] in replacements:
            recovered = replacements[row['id']]
            assert recovered['source_precision_float32_roundtrip_exact']
            source = Path(recovered['path']).resolve()
            assert file_hash(source) == recovered['sha256']
            audio,info = decode(source)
            row['source_recording'] = {**row['source_recording'],
                'precision_repair':{'prepared_parent_sha256':row['sha256'],
                                    'restored_from':'Verified original DDL PCM32 members, float32 decoder precision',
                                    'audit_sha256':file_hash('reports/training-audit-v1/source-conversion.json')}}
            pcm = np.rint(audio.astype(np.float64)*32767).clip(-32768,32767).astype('<i2')
            normalized = np.rint(audio/max(float(np.max(abs(audio))),1e-20)*1024).astype('<i2')
            row.update(info)
            row.update({'sha256':file_hash(source),'decoded_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),
                        'gain_normalized_sha256':hashlib.sha256(normalized.tobytes()).hexdigest()})
        row['path'] = os.path.relpath(source,output.parent)
    for row,group in zip(rows,components(rows),strict=True):
        row['group_id'] = group
    write_jsonl(output,rows)
    checked = load_manifest(output,True)
    assert len(checked) == 615 and sum(row['id'] in replacements for row in checked) == 15
    write_json('reports/training-audit-v1/precision-repair.json',{
        'scope':'Isolated TRAIN/DEVELOPMENT precision repair; original corpus and all frozen models unchanged. No final audio or new labels.',
        'parent_manifest_sha256':file_hash(original),'manifest':str(output),
        'manifest_sha256':file_hash(output),'recordings':len(rows),'repaired_ddl_training_files':15,
        'format':'FLOAT WAV preserves exactly the original selected-channel float32 ingress samples',
        'repaired_ids':sorted(replacements),'raw_audio_packaged':False,
        'usage':'Prepare a new feature cache from this manifest for future controlled precision experiments. Old caches/models reject the changed manifest identity.'})
    print('Created isolated615-record train/dev variant with15 verified original-precision DDL clips.')


if __name__ == '__main__':
    main()
