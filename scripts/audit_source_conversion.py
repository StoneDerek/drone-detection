#!/usr/bin/env python3
"""Compare prepared train/dev WAVs to cached original publisher bytes, offline."""

import collections
import hashlib
import io
from math import gcd
from pathlib import Path
import zipfile

import numpy as np
from scipy import signal
import soundfile as sf
from threadpoolctl import threadpool_limits

from ddl_audio import LENGTH
from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, read_jsonl, write_json
from drone_detector.data import load_manifest
from drone_detector.model import load_model
from drone_detector.spectrogram import spectrogram


class OfflineRanges(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.cache = collections.OrderedDict()
        self.checked = set()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else LENGTH+offset
        return self.position

    def read(self, size=-1):
        remaining = min(LENGTH-self.position, size if size >= 0 else LENGTH-self.position)
        result = []
        while remaining > 0:
            block, at = divmod(self.position, 4*1024*1024)
            if block not in self.cache:
                path = Path('runs/real-v2/ddl-blocks')/f'{block}.bin'
                if not path.exists():
                    raise ValueError(f'Missing offline DDL block{block}; no network fallback')
                data = path.read_bytes()
                assert hashlib.sha256(data).hexdigest() == path.with_suffix('.sha256').read_text().strip()
                self.checked.add(block)
                self.cache[block] = data
                if len(self.cache) > 8:
                    self.cache.popitem(last=False)
            data = self.cache[block]
            take = min(remaining, len(data)-at)
            assert take > 0
            result.append(data[at:at+take])
            self.position += take
            remaining -= take
        return b''.join(result)


def main():
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    rows = [row for row in load_manifest(manifest, True) if row['split'] in {'train','development'}]
    cache = read_json('runs/spectral-lab-v1/cache/linear256-fft4096/index.json')
    cached = {row['id']:row for row in cache['records']}
    model = load_model('artifacts/cnn-foundations-v1/frozen/coherent-frequency-seed42', True)
    converted, byte_identical = [], []

    def compare(row, original, rate, subtype, payload_check, physical_channels):
        path = manifest.parent/row['path']
        prepared, prepared_rate = sf.read(path, dtype='float32')
        assert prepared.ndim == original.ndim == 1
        assert prepared_rate == rate and len(prepared) == len(original)
        error = prepared.astype(np.float64)-original.astype(np.float64)
        noise = float(np.mean(error**2))
        power = float(np.mean(original.astype(np.float64)**2))
        original_peak = float(np.max(abs(original)))
        canonical_original = original
        if rate != 16000:
            divisor = gcd(rate,16000)
            canonical_original = signal.resample_poly(original,16000//divisor,rate//divisor)
        canonical_prepared, _ = decode(path)
        starts = cached[row['id']]['starts']
        start = starts[len(starts)//2]
        old = spectrogram(canonical_prepared[start:start+48000],model.config)
        reference = spectrogram(canonical_original[start:start+48000],model.config)
        scores = model.score_images(np.stack([old,reference]))
        recovered = None
        all_context_score_change = None
        if subtype != sf.info(path).subtype:
            recovered_path = Path('runs/training-audit-v1/original-precision-audio')/(row['id']+'.wav')
            recovered_path.parent.mkdir(parents=True,exist_ok=True)
            if not recovered_path.exists():
                sf.write(recovered_path,original,rate,subtype='FLOAT')
            restored,restored_rate = sf.read(recovered_path,dtype='float32')
            assert restored_rate == rate and np.array_equal(restored,original)
            originals = np.stack([spectrogram(canonical_original[s:s+48000],model.config) for s in starts])
            prepared_images = np.stack([spectrogram(canonical_prepared[s:s+48000],model.config) for s in starts])
            a,b = model.score_images(originals),model.score_images(prepared_images)
            all_context_score_change = float(np.max(abs(a-b)))
            recovered = {'path':str(recovered_path),'sha256':file_hash(recovered_path),
                         'subtype':'FLOAT','source_precision_float32_roundtrip_exact':True,
                         'all_cached_contexts_checked':len(starts)}
        result = {'id':row['id'],'source':row['source'],'label':row['label'],'split':row['split'],
            'original_subtype':subtype,'prepared_subtype':sf.info(path).subtype,'original_rate':rate,
            'selected_channel':row['source_recording']['selected_channel'],'original_channels':physical_channels,
            'original_peak':original_peak,'original_outside_unit_fraction':float(np.mean(abs(original)>1)),
            'waveform_max_error':float(np.max(abs(error))),
            'quantization_snr_db':float(10*np.log10(power/noise)) if noise and power else None,
            'quantization_mean_error':float(error.mean()),'original_payload_identity_checked':payload_check,
            'context_start_sample':start,'context_image_mean_change':float(np.mean(abs(old-reference))),
            'context_image_max_change':float(np.max(abs(old-reference))),
            'prepared_score':float(scores[0]),'original_precision_score':float(scores[1]),
            'score_change':float(scores[1]-scores[0]),'recovered_original_precision':recovered,
            'all_cached_context_max_score_change':all_context_score_change}
        assert result['waveform_max_error'] <= 1/32768+1e-7 or original_peak > 1
        converted.append(result)

    d3rows = [row for row in rows if row['source'].startswith('d3set-')]
    with zipfile.ZipFile('runs/real-v1/downloads/kaggle-d3set-v1.zip') as archive:
        names = {name.split('/',1)[1]:name for name in archive.namelist() if '/' in name}
        for number,row in enumerate(d3rows):
            provenance = row['source_recording']
            payload = archive.read(names[provenance['original_path']])
            assert hashlib.sha256(payload).hexdigest() == provenance['sha256']
            info = sf.info(io.BytesIO(payload))
            raw,rate = sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
            channel = provenance['selected_channel']
            energy = np.einsum('ij,ij->j',raw,raw,dtype=np.float64)/len(raw)
            assert channel == int(np.flatnonzero(np.sqrt(energy)>1e-6)[0])
            compare(row,raw[:,channel].copy(),rate,info.subtype,True,info.channels)
            if (number+1)%20 == 0:
                print('Original D3 conversions checked',number+1,flush=True)
    inventory = read_json('runs/real-v2/ddl-inventory.json')
    members = {}
    for item in inventory['entries']:
        path = Path(item['path'])
        if path.suffix.lower() == '.wav':
            members[(path.parent.name,int(path.stem.rsplit('-',1)[-1]))] = item['path']
    hashes = {row['path']:row['sha256'] for row in read_jsonl('runs/real-v2/ddl-selected-members.jsonl')}
    reader = OfflineRanges()
    with zipfile.ZipFile(reader) as archive:
        for row in [row for row in rows if row['source']=='ddl-real']:
            provenance = row['source_recording']
            chunks = []
            for seq in provenance['sequences']:
                name = members[(provenance['flight_id'],seq)]
                payload = archive.read(name)
                assert hashlib.sha256(payload).hexdigest() == hashes[name]
                info = sf.info(io.BytesIO(payload))
                audio,rate = sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
                chunks.append(audio[:,provenance['selected_channel']].copy())
            compare(row,np.concatenate(chunks),rate,info.subtype,True,info.channels)
            print('Original DDL reconstructed',row['id'],flush=True)
    for row in rows:
        if row['source'].startswith('d3set-') or row['source']=='ddl-real':
            continue
        assert file_hash(manifest.parent/row['path']) == row['source_recording']['sha256']
        byte_identical.append(row['id'])
    result = {'passed':True,'scope':'All615 existing TRAIN/DEVELOPMENT file origins; no final audio, downloads or model fitting',
        'converted_recordings_checked':len(converted),'unchanged_original_bytes':len(byte_identical),
        'checked_ddl_cache_blocks':len(reader.checked),'model_id':model.model_id,
        'conversion_subtypes':dict(collections.Counter(row['original_subtype']+'→'+row['prepared_subtype'] for row in converted)),
        'maximum_waveform_error':max(row['waveform_max_error'] for row in converted),
        'minimum_quantization_snr_db':min(row['quantization_snr_db'] for row in converted if row['quantization_snr_db'] is not None),
        'maximum_absolute_context_score_change':max(abs(row['score_change']) for row in converted),
        'original_clipped_files':sum(row['original_outside_unit_fraction']>0 for row in converted),
        'recovered_original_precision_files':sum(row['recovered_original_precision'] is not None for row in converted),
        'recovered_contexts_checked':sum(row['recovered_original_precision']['all_cached_contexts_checked'] for row in converted if row['recovered_original_precision']),
        'recovered_all_context_max_score_change':max((row['all_cached_context_max_score_change'] for row in converted if row['all_cached_context_max_score_change'] is not None),default=0.),
        'converted':converted,'byte_identical_ids':byte_identical,
        'limitation':'One middle cached context per converted file scored; it is not proof that faint acoustic information never changed. Float32 original decoding matches the model ingress precision.'}
    write_json('reports/training-audit-v1/source-conversion.json',result)
    print({k:v for k,v in result.items() if k not in {'converted','byte_identical_ids'}},flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
