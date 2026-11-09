import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load(ROOT / 'notebooks/bootstrap.py')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def source_repo(tmp_path):
    repo = tmp_path / 'origin'
    repo.mkdir()
    git(repo, 'init', '--quiet', '--initial-branch=main')
    git(repo, 'config', 'user.name', 'Test')
    git(repo, 'config', 'user.email', 'test@example.invalid')
    lock = repo / 'notebooks/runtime/uv.lock'
    lock.parent.mkdir(parents=True)
    lock.write_text('original lock')
    git(repo, 'add', '.')
    git(repo, 'commit', '--quiet', '-m', 'first')
    return repo, lock


def test_pin_survives_upstream_changes_and_runtime_replacement(tmp_path):
    origin, lock = source_repo(tmp_path)
    project, checkout = tmp_path / 'drive', tmp_path / 'runtime1'
    original = bootstrap.pin_checkout(project, checkout, str(origin))
    lock.write_text('new lock')
    git(origin, 'commit', '--quiet', '-am', 'second')
    assert git(origin, 'rev-parse', 'HEAD') != original['revision']
    # New local checkout represents a replacement runtime.
    resumed = bootstrap.pin_checkout(project, tmp_path / 'runtime2', str(origin))
    assert resumed == original
    assert (tmp_path / 'runtime2/notebooks/runtime/uv.lock').read_text() == 'original lock'
    pin = json.loads((project / 'bootstrap.json').read_text())
    pin['runtime_lock_sha256'] = '0' * 64
    (project / 'bootstrap.json').write_text(json.dumps(pin))
    with pytest.raises(ValueError, match='lock changed'):
        bootstrap.pin_checkout(project, checkout, str(origin))


def test_bootstrap_refuses_dirty_checkout_and_changed_origin(tmp_path):
    origin, _ = source_repo(tmp_path)
    project, checkout = tmp_path / 'drive', tmp_path / 'runtime'
    bootstrap.pin_checkout(project, checkout, str(origin))
    (checkout / 'notes.txt').write_text('preserve me')
    with pytest.raises(ValueError, match='Local checkout changed'):
        bootstrap.pin_checkout(project, checkout, str(origin))
    assert (checkout / 'notes.txt').read_text() == 'preserve me'
    with pytest.raises(ValueError, match='Invalid/different project pin'):
        bootstrap.pin_checkout(project, checkout, 'https://example.invalid/other.git')


@pytest.mark.parametrize(('filename', 'builder'), [
    ('gpu_runtime_setup.ipynb', 'build_gpu_notebook.py'),
    ('source_intake.ipynb', 'build_intake_notebook.py'),
])
def test_notebook_schema_generated_content_and_code_compile(filename, builder):
    nbformat = pytest.importorskip('nbformat')
    path = ROOT / 'notebooks' / filename
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    expected = load(ROOT / 'scripts' / builder).notebook()
    assert json.loads(path.read_text()) == expected
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert not cell.outputs and cell.execution_count is None
            compile(cell.source, f'cell-{cell.id}', 'exec')
    embedded = next(c.source for c in notebook.cells if c.id == 'pinned-bootstrap')
    assert embedded.startswith((ROOT / 'notebooks/bootstrap.py').read_text())
