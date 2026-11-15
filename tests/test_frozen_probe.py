"""Independent workflow checks using numeric fixture embeddings, never pretend weights."""

import json

import numpy as np
import pytest
import soundfile as sf
from scipy.special import expit

from drone_detector import frozen_probe as probe
from drone_detector.common import digest, file_hash, read_json, write_jsonl
from drone_detector.data import components


class FixtureEncoder:
    """Deterministic software fixture; no acoustic representation claim."""

    identity = {'name': 'numeric-software-fixture', 'version': 1}

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def features(self, audio):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise InterruptedError('simulated process interruption during encoder call')
        self.calls += 1
        audio = np.asarray(audio, dtype=np.float64)
        return np.asarray([audio.mean(), np.mean(audio**2), audio[0], audio.std()])


def make_record(root, identifier, split, label, offset, *, rate=16000, frames=16000):
    path = root / f'{identifier}.wav'
    samples = offset + .04 * np.sin(np.arange(frames) * (71 + len(identifier)) / rate)
    sf.write(path, samples.astype(np.float32), rate, subtype='FLOAT')
    return {
        'schema_version': 1, 'id': identifier, 'path': path.name,
        'source': 'source-' + split, 'source_version': 'fixture-v1',
        'session_id': identifier, 'original_id': identifier,
        'origin': 'synthetic', 'label': label, 'annotation_status': 'weak',
        'label_evidence': 'Explicit numeric software fixture, not drone audio',
        'duration_seconds': frames / rate, 'events': [], 'ambiguous': [],
        'sha256': file_hash(path), 'split': split,
        'duplicate_review_required': False,
    }


def save_manifest(path, rows):
    for row, group in zip(rows, components(rows), strict=True):
        row['group_id'] = group
    write_jsonl(path, rows)
    return path


@pytest.fixture
def corpus(tmp_path):
    rows = [make_record(tmp_path, identifier, split, label, offset)
            for identifier, split, label, offset in [
                ('train-n0', 'train', 'non_drone', -.60),
                ('train-n1', 'train', 'non_drone', -.30),
                ('train-p0', 'train', 'drone', .20),
                ('train-p1', 'train', 'drone', .50),
                ('dev-n0', 'development', 'non_drone', -.80),
                ('dev-p0', 'development', 'drone', .80),
            ]]
    manifest = save_manifest(tmp_path / 'manifest.jsonl', rows)
    config = probe.ProbeConfig(window_seconds=.5, hop_seconds=.5,
                               min_audio_seconds=.25, max_windows=3)
    return manifest, rows, config


def test_final_audio_never_opened_and_uncertain_excluded(corpus, tmp_path, monkeypatch):
    manifest, rows, config = corpus
    # These deliberately nonexistent payloads must not even be hashed or opened.
    final = {**rows[0], 'id': 'reserved', 'path': 'NEVER_OPEN_FINAL.wav',
             'source': 'reserved-source', 'session_id': 'reserved',
             'original_id': 'reserved', 'split': 'final', 'sha256': '1' * 64}
    uncertain = {**final, 'id': 'uncertain', 'path': 'NEVER_OPEN_UNKNOWN.wav',
                 'source': 'uncertain-source', 'split': 'train',
                 'label': 'uncertain', 'annotation_status': 'unknown', 'sha256': '2' * 64}
    save_manifest(manifest, rows + [final, uncertain])
    original_hash = probe.file_hash

    def watched_hash(path):
        assert 'NEVER_OPEN' not in str(path)
        return original_hash(path)

    monkeypatch.setattr(probe, 'file_hash', watched_hash)
    report = probe.run_probe(manifest, tmp_path / 'run', tmp_path / 'cache',
                             FixtureEncoder(), config)
    assert report['excluded_final_recordings'] == 1
    assert report['excluded_unknown_or_uncertain_recordings'] == 1
    assert report['training_recordings'] == 4
    assert report['development_recordings'] == 2
    assert report['feature_recordings'] == 6
    assert report['field_reliability_claim'] is False
    assert {r['id'] for r in report['predictions']} == {'dev-n0', 'dev-p0'}


def test_shared_source_recording_rejected_before_encoding(corpus, tmp_path):
    manifest, rows, config = corpus
    rows[4].update(source=rows[0]['source'], session_id=rows[0]['session_id'])
    save_manifest(manifest, rows)
    encoder = FixtureEncoder()
    with pytest.raises(ValueError, match='leaks across splits'):
        probe.run_probe(manifest, tmp_path / 'run', tmp_path / 'cache', encoder, config)
    assert encoder.calls == 0


def test_scaler_fitted_only_on_training_recording_means(corpus, tmp_path):
    manifest, rows, config = corpus
    encoder = FixtureEncoder()
    probe.run_probe(manifest, tmp_path / 'run', tmp_path / 'cache', encoder, config)
    model = probe.load_probe(tmp_path / 'run/head', encoder)
    training_means = []
    for row in rows[:4]:
        entry = probe.cached_features(row, manifest, encoder, config, tmp_path / 'cache')
        training_means.append(entry['vectors'].mean(axis=0))
    expected = np.mean(training_means, axis=0)
    np.testing.assert_allclose(model.arrays['mean'], expected, rtol=0, atol=1e-12)
    # This held-out shift must have no influence on TRAIN normalization.
    old_mean = model.arrays['mean'].copy()
    dev = rows[4]
    sf.write(tmp_path / dev['path'], np.full(16000, 20., dtype=np.float32), 16000,
             subtype='FLOAT')
    dev['sha256'] = file_hash(tmp_path / dev['path'])
    save_manifest(manifest, rows)
    probe.run_probe(manifest, tmp_path / 'run-shift', tmp_path / 'cache', encoder, config)
    changed = probe.load_probe(tmp_path / 'run-shift/head', encoder)
    np.testing.assert_array_equal(changed.arrays['mean'], old_mean)
    np.testing.assert_array_equal(changed.arrays['scale'], model.arrays['scale'])


def test_recording_mean_logit_aggregation_and_reload(corpus, tmp_path):
    manifest, rows, config = corpus
    encoder = FixtureEncoder()
    report = probe.run_probe(manifest, tmp_path / 'run', tmp_path / 'cache', encoder, config)
    model = probe.load_probe(tmp_path / 'run/head', encoder)
    for record, prediction in zip(rows[4:], report['predictions'], strict=True):
        entry = probe.cached_features(record, manifest, encoder, config, tmp_path / 'cache')
        expected = float(expit(np.mean(model.logits(entry['vectors']))))
        assert prediction['recording_score'] == pytest.approx(expected, abs=1e-12)
        assert prediction['aggregation'] == 'sigmoid(mean context logits)'
        assert prediction['label_scope'] == 'weak'
        trace_path = tmp_path / 'run/traces' / f"{digest(record['id'])[:20]}.jsonl"
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        assert len(trace) == len(entry['bounds'])
        assert trace[0]['timestamp_seconds'] == entry['bounds'][0, 1] / config.sample_rate
        assert all('no per-window truth' in row['scope'] for row in trace)
        audio = probe.decode_record(record, manifest, config)
        a, b = entry['bounds'][0]
        score = model.score(probe.padded_context(audio, a, b, config))
        assert score == pytest.approx(float(expit(model.logits(entry['vectors'][0]))), abs=1e-12)
    summary = report['summary']
    assert 'roc_auc_recording_score' in summary
    assert 'roc_auc_max_window_score' not in summary
    assert summary['event_metrics'].startswith('Not assessed:')
    assert 'false_alarms_per_hour' not in summary
    assert summary['field_reliability_claim'] is False
    assert report['scope'].startswith('DEVELOPMENT')


def test_completed_cache_survives_interruption_and_rerun(corpus, tmp_path):
    manifest, rows, config = corpus
    cache = tmp_path / 'cache'
    with pytest.raises(InterruptedError):
        # Two windows finish the first record; interruption occurs in the second.
        probe.run_probe(manifest, tmp_path / 'run', cache,
                        FixtureEncoder(fail_after=2), config)
    assert len(list(cache.glob('*.json'))) == 1
    assert not (tmp_path / 'run/comparison.json').exists()
    encoder = FixtureEncoder()
    report = probe.run_probe(manifest, tmp_path / 'run', cache, encoder, config)
    assert report['cache_hits'] == 1
    assert encoder.calls == 10
    original_calls = encoder.calls
    again = probe.run_probe(manifest, tmp_path / 'run', cache, encoder, config)
    assert again['cache_hits'] == 6
    assert encoder.calls == original_calls
    assert again['model_id'] == report['model_id']
    assert again['predictions'] == report['predictions']


def test_partial_payload_without_receipt_is_recomputed(corpus, tmp_path):
    manifest, rows, config = corpus
    encoder = FixtureEncoder()
    cache = tmp_path / 'cache'
    entry = probe.cached_features(rows[0], manifest, encoder, config, cache)
    key = entry['cache_key']
    (cache / f'{key}.json').unlink()
    (cache / f'{key}.npz').write_bytes(b'interrupted incomplete bytes')
    recovered = probe.cached_features(rows[0], manifest, encoder, config, cache)
    assert recovered['cache_hit'] is False
    np.testing.assert_array_equal(recovered['vectors'], entry['vectors'])
    assert encoder.calls == 4


def test_corrupt_completed_cache_fails_closed(corpus, tmp_path):
    manifest, rows, config = corpus
    encoder = FixtureEncoder()
    cache = tmp_path / 'cache'
    entry = probe.cached_features(rows[0], manifest, encoder, config, cache)
    (cache / f"{entry['cache_key']}.npz").write_bytes(b'changed payload')
    with pytest.raises(ValueError, match='Corrupt feature cache'):
        probe.cached_features(rows[0], manifest, encoder, config, cache)
    assert encoder.calls == 2


def test_changed_source_bytes_rejected_even_with_cache(corpus, tmp_path):
    manifest, rows, config = corpus
    encoder = FixtureEncoder()
    cache = tmp_path / 'cache'
    probe.cached_features(rows[0], manifest, encoder, config, cache)
    (tmp_path / rows[0]['path']).write_bytes(b'changed audio')
    with pytest.raises(ValueError, match='Audio identity changed'):
        probe.cached_features(rows[0], manifest, encoder, config, cache)
    assert encoder.calls == 2


def test_nonintegral_resampling_geometry_is_accepted(tmp_path):
    row = make_record(tmp_path, 'native44100', 'train', 'drone', .2,
                      rate=44100, frames=44101)
    manifest = save_manifest(tmp_path / 'manifest.jsonl', [row])
    config = probe.ProbeConfig(window_seconds=1, hop_seconds=.5)
    encoder = FixtureEncoder()
    entry = probe.cached_features(row, manifest, encoder, config, tmp_path / 'cache')
    canonical = probe.decode_record(row, manifest, config)
    assert len(canonical) == 16001
    assert entry['bounds'][-1, 1] == 16001
    reused = probe.cached_features(row, manifest, encoder, config, tmp_path / 'cache')
    np.testing.assert_array_equal(reused['bounds'], entry['bounds'])
    assert reused['cache_hit'] is True


def test_interrupted_head_write_can_resume(corpus, tmp_path):
    manifest, rows, config = corpus
    output = tmp_path / 'run'
    (output / 'head').mkdir(parents=True)
    (output / 'head/parameters.npz.partial').write_bytes(b'half-written')
    report = probe.run_probe(manifest, output, tmp_path / 'cache', FixtureEncoder(), config)
    assert report['model_id']
    assert read_json(output / 'head/model.json')['model_id'] == report['model_id']


def test_group_weights_equalize_class_and_group_mass():
    records = [{'group_id': group, 'label': label} for group, label in [
        ('a', 'drone'), ('a', 'drone'), ('b', 'drone'),
        ('a', 'non_drone'), ('c', 'non_drone'), ('c', 'non_drone'), ('c', 'non_drone'),
    ]]
    weights = probe.group_class_weights(records)
    for label in ['drone', 'non_drone']:
        indexes = [i for i, r in enumerate(records) if r['label'] == label]
        assert sum(weights[indexes]) == pytest.approx(3.5)
        groups = {records[i]['group_id'] for i in indexes}
        for group in groups:
            chosen = [i for i in indexes if records[i]['group_id'] == group]
            assert sum(weights[chosen]) == pytest.approx(1.75)


def test_source_macro_metric_does_not_let_large_source_dominate():
    rows = [{'source': 'large', 'truth': 1, 'predicted': int(i < 99)} for i in range(100)]
    rows += [{'source': 'small', 'truth': 1, 'predicted': 0},
             {'source': 'background', 'truth': 0, 'predicted': 0}]
    assert probe.macro_balanced(rows) == pytest.approx(((.99 + 0) / 2 + 1) / 2)


def test_model_and_encoder_identity_are_enforced(corpus, tmp_path):
    manifest, rows, config = corpus
    encoder = FixtureEncoder()
    probe.run_probe(manifest, tmp_path / 'run', tmp_path / 'cache', encoder, config)
    wrong = FixtureEncoder()
    wrong.identity = {'name': 'another-checkpoint'}
    with pytest.raises(ValueError, match='encoder identity mismatch'):
        probe.load_probe(tmp_path / 'run/head', wrong)
    with (tmp_path / 'run/head/parameters.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='parameter identity mismatch'):
        probe.load_probe(tmp_path / 'run/head', encoder)


def test_fixed_reference_uses_identical_padded_development_contexts(corpus, tmp_path, monkeypatch):
    from drone_detector import model as existing_models
    from drone_detector.audio import FeatureConfig

    manifest, rows, config = corpus

    class Reference:
        model_id = 'fixed-reference-software-fixture'

        def __init__(self):
            self.config = FeatureConfig(window_seconds=.5, hop_seconds=.5,
                                        min_audio_seconds=.25)
            self.contexts = []

        def score(self, samples):
            self.contexts.append(samples.copy())
            return float(expit(2 * samples.mean()))

    reference = Reference()

    def load_reference(path, require_frozen=False):
        assert path == 'fixture-reference'
        assert require_frozen is True
        return reference

    monkeypatch.setattr(existing_models, 'load_model', load_reference)
    encoder = FixtureEncoder()
    report = probe.run_probe(manifest, tmp_path / 'run', tmp_path / 'cache',
                             encoder, config, reference_bundle='fixture-reference')
    assert len(reference.contexts) == 4
    context_index = 0
    for record in rows[4:]:
        audio = probe.decode_record(record, manifest, config)
        for start, end in probe.window_plan(len(audio), config):
            expected = probe.padded_context(audio, start, end, config)
            np.testing.assert_array_equal(reference.contexts[context_index], expected)
            context_index += 1
    assert report['reference']['model_id'] == reference.model_id
    assert 'threshold selected on this DEVELOPMENT set' in report['reference']['weights']
    assert 'identical selected crops' in report['reference']['aggregation']


def test_cross_corpus_identical_payload_cannot_cross_split(corpus, tmp_path):
    manifest, rows, config = corpus
    # Different source, session and filenames do not make the same bytes independent.
    rows[4]['sha256'] = rows[0]['sha256']
    save_manifest(manifest, rows)
    encoder = FixtureEncoder()
    with pytest.raises(ValueError, match='leaks across splits'):
        probe.run_probe(manifest, tmp_path / 'run', tmp_path / 'cache', encoder, config)
    assert encoder.calls == 0
