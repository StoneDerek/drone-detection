"""Standard-library-only bootstrap, embedded verbatim in the Colab notebook."""

import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path


def git_output(checkout, *arguments):
    return subprocess.check_output(['git', '-C', str(checkout), *arguments], text=True).strip()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin_checkout(project, checkout, repo_url, first_ref='main'):
    """Resolve once, then reuse that exact revision after runtime replacement.

    One active runtime per project. Refuse dirty or unrelated local checkouts.
    The saved lock hash is also checked; a pin is never silently refreshed.
    """
    project, checkout = Path(project), Path(checkout)
    project.mkdir(parents=True, exist_ok=True)
    pin_path = project / 'bootstrap.json'
    pin = json.loads(pin_path.read_text()) if pin_path.exists() else None
    if pin is not None and (pin.get('schema_version') != 1
                            or pin.get('repo_url') != repo_url
                            or not re.fullmatch(r'[0-9a-f]{40}', pin.get('revision', ''))
                            or not re.fullmatch(r'[0-9a-f]{64}', pin.get('runtime_lock_sha256', ''))):
        raise ValueError('Invalid/different project pin; choose a new project folder deliberately')
    if checkout.exists() and any(checkout.iterdir()):
        if not (checkout / '.git').exists():
            raise ValueError('Checkout folder contains unrelated files')
        if git_output(checkout, 'remote', 'get-url', 'origin') != repo_url:
            raise ValueError('Checkout has a different repository origin')
        if git_output(checkout, 'status', '--porcelain'):
            raise ValueError('Local checkout changed; preserve those changes before continuing')
    else:
        checkout.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', '--quiet', str(checkout)], check=True)
        subprocess.run(['git', '-C', str(checkout), 'remote', 'add', 'origin', repo_url], check=True)
    target = pin['revision'] if pin else first_ref
    subprocess.run(['git', '-C', str(checkout), 'fetch', '--quiet', '--depth', '1', 'origin', target], check=True)
    revision = git_output(checkout, 'rev-parse', 'FETCH_HEAD')
    if pin and revision != pin['revision']:
        raise ValueError('Fetched revision differs from saved project pin')
    subprocess.run(['git', '-C', str(checkout), 'checkout', '--quiet', '--detach', revision], check=True)
    candidate = {'schema_version': 1, 'repo_url': repo_url, 'revision': revision,
                 'runtime_lock_sha256': file_hash(checkout / 'notebooks/runtime/uv.lock')}
    if pin is not None and pin != candidate:
        raise ValueError('Pinned environment lock changed')
    if pin is None:
        payload = json.dumps(candidate, indent=2, sort_keys=True).encode()
        temporary = pin_path.with_name('bootstrap.partial-' + uuid.uuid4().hex)
        temporary.write_bytes(payload)
        if pin_path.exists() and json.loads(pin_path.read_text()) != candidate:
            raise ValueError('Another bootstrap wrote a different pin; use one active runtime')
        os.replace(temporary, pin_path)
        if pin_path.read_bytes() != payload:
            raise ValueError('Project pin read-back failed')
    return candidate
