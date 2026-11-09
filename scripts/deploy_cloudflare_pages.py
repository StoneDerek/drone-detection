#!/usr/bin/env python3
"""Deploy tested browser assets to Pages with the configured scoped token.

Uses the same check-missing/upload/upsert/manifest protocol as pinned Wrangler.
Credentials and upload JWTs stay in memory and are never written or printed.
"""

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from drone_detector.common import file_hash, read_json, write_json


def api(endpoint, token, data=None, content_type='application/json', method=None):
    body = json.dumps(data).encode() if data is not None and not isinstance(data, bytes) else data
    request = urllib.request.Request('https://api.cloudflare.com/client/v4'+endpoint, data=body,
        headers={'Authorization': 'Bearer '+token, 'Content-Type': content_type}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            details = json.loads(raw).get('errors', [])
            message = '; '.join(str(x.get('code'))+': '+x.get('message', '') for x in details)
        except (ValueError, AttributeError):
            message = 'Non-JSON API error'
        raise RuntimeError(f'Cloudflare HTTP {error.code} at {endpoint}: {message}') from None
    if not result.get('success'):
        raise RuntimeError(f'Cloudflare request failed at {endpoint}: {result.get("errors")}')
    return result.get('result')


def multipart(fields, headers):
    boundary = 'drone-lab-'+uuid.uuid4().hex
    pieces = []
    for name, value in fields.items():
        pieces.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                       value.encode(), b'\r\n'])
    pieces.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="_headers"; filename="_headers"\r\nContent-Type: text/plain\r\n\r\n'.encode(),
                   headers, b'\r\n', f'--{boundary}--\r\n'.encode()])
    return b''.join(pieces), 'multipart/form-data; boundary='+boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='drone-audio-lab')
    parser.add_argument('--account-id', required=True)
    parser.add_argument('--evidence-dir', default='reports/cloudflare-v2')
    args = parser.parse_args()
    root = Path('dist/cloudflare')
    evidence = Path(args.evidence_dir)
    for proof in ['parity.json', 'local-browser.json', 'interactions.json']:
        checked = read_json(evidence / proof)
        if not checked['passed'] or checked['app_js_sha256'] != file_hash(root / 'app.js'):
            raise ValueError(f'Deployment requires passing {proof}')
    files = read_json('runs/cloudflare-deployment/assets.json')
    for item in files:
        if file_hash(item['path']) != item['sha256'] or item['size'] > 25 * 1024 * 1024:
            raise ValueError('Asset changed since hashing or exceeds Pages limit')
    token = os.environ['CLOUDFLARE_API_TOKEN']
    base = f'/accounts/{args.account_id}/pages/projects'
    existing = api(base, token)
    project = next((p for p in existing if p['name'] == args.project), None)
    if project is None:
        project = api(base, token, {'name': args.project, 'production_branch': 'main'})
        print('Created Pages project', args.project, flush=True)
    elif project.get('source'):
        raise ValueError('Refusing to replace an existing Git-integrated project')
    project_base = base+'/'+args.project
    jwt = api(project_base+'/upload-token', token)['jwt']
    missing = set(api('/pages/assets/check-missing', jwt, {'hashes': [f['hash'] for f in files]}))
    # Batches stay below6MiB after base64 expansion, including large model shards.
    batches, batch, size = [], [], 0
    for item in sorted(files, key=lambda f: f['size'], reverse=True):
        if item['hash'] not in missing:
            continue
        predicted = 4 * ((item['size'] + 2) // 3) + 512
        if batch and size + predicted > 6 * 1024 * 1024:
            batches.append(batch)
            batch, size = [], 0
        batch.append(item)
        size += predicted
    if batch:
        batches.append(batch)
    for number, batch in enumerate(batches):
        payload = [{'key': item['hash'], 'value': base64.b64encode(Path(item['path']).read_bytes()).decode(),
                    'metadata': {'contentType': mimetypes.guess_type(item['path'])[0] or 'application/octet-stream'},
                    'base64': True} for item in batch]
        api('/pages/assets/upload', jwt, payload)
        print(f'Uploaded asset batch {number+1}/{len(batches)}', flush=True)
    api('/pages/assets/upsert-hashes', jwt, {'hashes': [f['hash'] for f in files]})
    body, content_type = multipart({'manifest': json.dumps({f['url']: f['hash'] for f in files}),
        'branch': 'main', 'commit_message': 'Deploy validated browser audio model lab and real-case observations',
        'commit_dirty': 'true'}, (root / '_headers').read_bytes())
    deployment = api(project_base+'/deployments', token, body, content_type)
    write_json('runs/cloudflare-deployment/deployment-private.json', deployment)
    for _ in range(24):
        stage = deployment.get('latest_stage', {})
        if stage.get('name') == 'deploy' and stage.get('status') == 'success':
            break
        if stage.get('status') in {'failure', 'canceled'}:
            raise RuntimeError('Pages deployment failed: '+str(stage))
        time.sleep(5)
        deployment = api(project_base+'/deployments/'+deployment['id'], token)
    else:
        raise RuntimeError('Pages deployment pending; inspect existing deployment before retrying')
    proof = {'deployed': True, 'project': args.project, 'deployment_id': deployment['id'],
        'deployment_url': deployment['url'], 'production_url': 'https://'+project['subdomain'],
        'latest_stage': deployment['latest_stage'], 'assets': len(files),
        'total_bytes': sum(f['size'] for f in files), 'asset_manifest_sha256': file_hash('runs/cloudflare-deployment/assets.json'),
        'hosting': 'Cloudflare Pages static assets; inference executes in browser; no paid plan change',
        'validation': ['Python test suite (see validation report)', 'browser numerical parity', 'interactive browser workflow and no audio uploads']}
    write_json(evidence / 'deployment.json', proof)
    write_json(evidence / 'assets.json', [{k: f[k] for k in ['url', 'sha256', 'size']} for f in files])
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
