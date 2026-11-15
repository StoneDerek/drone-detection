"""Frozen audio embeddings and an explicit recording-mean linear probe.

No per-window truth is invented from publisher recording labels. Timestamped
window scores are diagnostic; the supervised target is the recording's label.
Final audio is never decoded by this development-only runner.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .audio import FeatureConfig
from .common import digest, file_hash, read_json, revision, versions, write_json, write_jsonl
from .data import load_manifest
from .weak import weak_metrics


@dataclass(frozen=True)
class ProbeConfig:
    window_seconds: float = 3.0
    hop_seconds: float = 0.5
    min_audio_seconds: float = 0.25
    max_windows: int = 60
    channel_index: int = 0
    sample_rate: int = 16000
    seed: int = 42
    regularization_c: float = 1.0

    def __post_init__(self):
        numbers = [self.window_seconds, self.hop_seconds, self.min_audio_seconds,
                   self.regularization_c]
        if not all(np.isfinite(v) and v > 0 for v in numbers):
            raise ValueError('Context and regularization must be positive and finite')
        if not self.min_audio_seconds <= self.window_seconds or self.hop_seconds > self.window_seconds:
            raise ValueError('Invalid context/hop/minimum')
        if self.sample_rate != 16000:
            raise ValueError('BEATs requires native 16 kHz input')
        if type(self.max_windows) is not int or self.max_windows < 1:
            raise ValueError('max_windows must be a positive integer')
        if type(self.channel_index) is not int or self.channel_index < 0:
            raise ValueError('Select a nonnegative physical channel')
        if round(self.hop_seconds * self.sample_rate) < 1:
            raise ValueError('Hop must contain at least one sample')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    write_json(temporary, value)
    temporary.replace(path)


def decode_record(record, manifest, config):
    """Select one explicit channel, preserve amplitudes, then resample offline."""
    path = (Path(manifest).resolve().parent / record['path']).resolve()
    if file_hash(path) != record['sha256']:
        raise ValueError(f"Audio identity changed: {record['id']}")
    wave, rate = sf.read(path, dtype='float32', always_2d=True)
    channel = record.get('channel_index', config.channel_index)
    if type(channel) is not int or not 0 <= channel < wave.shape[1]:
        raise ValueError(f"Invalid physical channel: {record['id']}")
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Empty or nonfinite audio')
    if abs(len(wave) / rate - record['duration_seconds']) > 1 / config.sample_rate:
        raise ValueError('Audio duration changed')
    selected = wave[:, channel]
    if rate != config.sample_rate:
        factor = gcd(rate, config.sample_rate)
        selected = resample_poly(selected, config.sample_rate // factor, rate // factor)
    return np.asarray(selected, dtype=np.float32)


def window_plan(sample_count, config):
    size = round(config.window_seconds * config.sample_rate)
    hop = round(config.hop_seconds * config.sample_rate)
    starts = list(range(0, max(1, sample_count - size + 1), hop))
    # Include a tail only when it adds new samples beyond the last full context.
    if starts[-1] + size < sample_count:
        starts.append(starts[-1] + hop)
    if len(starts) > config.max_windows:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, config.max_windows).astype(int)]
    return [(start, min(start + size, sample_count)) for start in starts
            if (min(start + size, sample_count) - start) / config.sample_rate >= config.min_audio_seconds]


def padded_context(audio, start, end, config):
    chunk = audio[start:end]
    return np.pad(chunk, (0, round(config.window_seconds * config.sample_rate) - len(chunk)))


def cached_features(record, manifest, encoder, config, cache):
    """Commit each recording independently; partial entries are recomputed."""
    specification = {
        'schema_version': 1, 'audio_sha256': record['sha256'],
        'duration_seconds': record['duration_seconds'],
        'channel_index': record.get('channel_index', config.channel_index),
        'encoder': encoder.identity, 'config': {k: v for k, v in asdict(config).items()
                                               if k not in {'seed', 'regularization_c'}},
        'implementation_sha256': file_hash(__file__), 'versions': versions(),
    }
    key = digest(specification)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    payload, receipt = cache / f'{key}.npz', cache / f'{key}.json'
    # Validate source bytes even on cache hits; metadata alone cannot certify them.
    path = (Path(manifest).resolve().parent / record['path']).resolve()
    if file_hash(path) != record['sha256']:
        raise ValueError(f"Audio identity changed: {record['id']}")
    audio_info = sf.info(path)
    sample_count = (audio_info.frames * config.sample_rate + audio_info.samplerate - 1) // audio_info.samplerate
    expected_bounds = np.asarray(window_plan(sample_count, config), dtype=np.int64).reshape(-1, 2)
    if payload.exists() and receipt.exists():
        meta = read_json(receipt)
        if meta.get('specification') != specification or file_hash(payload) != meta.get('sha256'):
            raise ValueError('Corrupt feature cache; remove this entry explicitly and recompute')
        with np.load(payload, allow_pickle=False) as saved:
            vectors, bounds = saved['vectors'].copy(), saved['bounds'].copy()
        cache_hit = True
    else:
        audio = decode_record(record, manifest, config)
        bounds = np.asarray(window_plan(len(audio), config), dtype=np.int64).reshape(-1, 2)
        if not len(bounds):
            raise ValueError(f"No eligible audio context: {record['id']}")
        vectors = np.asarray([encoder.features(padded_context(audio, a, b, config))
                              for a, b in bounds], dtype=np.float64)
        if vectors.ndim != 2 or len(vectors) != len(bounds) or not np.isfinite(vectors).all():
            raise ValueError('Invalid encoder embeddings')
        temporary = payload.with_suffix('.partial')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, vectors=vectors, bounds=bounds)
        temporary.replace(payload)
        atomic_json(receipt, {'specification': specification, 'sha256': file_hash(payload)})
        cache_hit = False
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError('Invalid cached embeddings')
    if bounds.shape != expected_bounds.shape or not np.array_equal(bounds, expected_bounds):
        raise ValueError('Cached window geometry changed')
    if len(vectors) != len(bounds):
        raise ValueError('Embedding/window mismatch')
    return {'vectors': vectors, 'bounds': bounds, 'cache_key': key, 'cache_hit': cache_hit}


class ProbeModel:
    """Portable numeric head; requires the separately pinned encoder directory."""
    def __init__(self, metadata, arrays, encoder):
        self.metadata, self.arrays, self.encoder = metadata, arrays, encoder
        self.model_id = metadata['model_id']
        self.probe_config = ProbeConfig(**metadata['config'])
        cfg = self.probe_config
        # StreamDetector reads only timing/silence fields; BEATs owns its frontend.
        self.config = FeatureConfig(window_seconds=cfg.window_seconds, hop_seconds=cfg.hop_seconds,
                                    min_audio_seconds=cfg.min_audio_seconds)

    def logits(self, vectors):
        return ((vectors - self.arrays['mean']) / self.arrays['scale']) @ self.arrays['coef'] + self.arrays['intercept'][0]

    def score(self, samples):
        if not len(samples) or len(samples) > self.config.window_samples:
            raise ValueError('Invalid inference context')
        chunk = np.pad(samples, (0, self.config.window_samples - len(samples)))
        return float(expit(self.logits(self.encoder.features(chunk))))


def load_probe(path, encoder):
    path = Path(path)
    meta = read_json(path / 'model.json')
    if meta.get('schema_version') != 'frozen-audio-probe-v1':
        raise ValueError('Unknown probe format')
    if digest({k: v for k, v in meta.items() if k != 'model_id'}) != meta.get('model_id'):
        raise ValueError('Probe metadata identity mismatch')
    if meta['encoder'] != encoder.identity:
        raise ValueError('Probe encoder identity mismatch')
    if file_hash(path / 'parameters.npz') != meta['parameters_sha256']:
        raise ValueError('Probe parameter identity mismatch')
    with np.load(path / 'parameters.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    if set(arrays) != {'mean', 'scale', 'coef', 'intercept'}:
        raise ValueError('Invalid probe parameters')
    dimension = meta['feature_dimension']
    if any(arrays[k].shape != (dimension,) for k in ['mean', 'scale', 'coef']):
        raise ValueError('Invalid probe feature dimension')
    if arrays['intercept'].shape != (1,) or np.any(arrays['scale'] <= 0):
        raise ValueError('Invalid probe normalization/intercept')
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite probe parameters')
    return ProbeModel(meta, arrays, encoder)


def group_class_weights(records):
    """Equal group mass within each class, then equal class mass."""
    counts = Counter((r['group_id'], r['label']) for r in records)
    groups = {label: len({r['group_id'] for r in records if r['label'] == label})
              for label in ['drone', 'non_drone']}
    return np.asarray([len(records) / (2 * groups[r['label']] * counts[r['group_id'], r['label']])
                       for r in records])


def metric_rows(records, scores, threshold):
    return [{'id': r['id'], 'source': r['source'], 'group_id': r['group_id'],
             'duration_seconds': r['duration_seconds'], 'truth': int(r['label'] == 'drone'),
             'predicted': int(s >= threshold), 'max_score': float(s), 'recording_score': float(s)}
            for r, s in zip(records, scores, strict=True)]


def metrics(rows):
    result = weak_metrics(rows, intervals=True)
    # Reuse existing confusion/cluster routines without misnaming mean pooling.
    result['roc_auc_recording_score'] = result.pop('roc_auc_max_window_score')
    result['average_precision_recording_score'] = result.pop('average_precision_max_window_score')
    result['positive_labelled_recording_hours'] = result.pop('positive_hours')
    result['negative_labelled_recording_hours'] = result.pop('negative_hours')
    result['duration_scope'] = 'Whole-recording weak-label duration; not verified audible or drone-absent exposure'
    return result


def macro_balanced(rows):
    rates = []
    for label in [0, 1]:
        values = []
        for source in sorted({r['source'] for r in rows}):
            selected = [r for r in rows if r['source'] == source and r['truth'] == label]
            if selected:
                values.append(np.mean([r['predicted'] == label for r in selected]))
        if not values:
            raise ValueError('Development requires both recording classes')
        rates.append(float(np.mean(values)))
    return float(np.mean(rates))


def select_threshold(records, scores):
    grid = [.01, .05, .1, .2, .35, .5, .65, .8, .9, .95, .99]
    trials = []
    for threshold in grid:
        rows = metric_rows(records, scores, threshold)
        negative = [r for r in rows if r['truth'] == 0]
        trials.append({'threshold': threshold, 'macro_source_balanced_accuracy': macro_balanced(rows),
                       'negative_recording_false_positive_fraction': float(np.mean([r['predicted'] for r in negative]))})
    chosen = max(trials, key=lambda t: (t['macro_source_balanced_accuracy'],
                 -t['negative_recording_false_positive_fraction'], -abs(t['threshold'] - .5), t['threshold']))
    return chosen['threshold'], trials


def run_probe(manifest, output, cache, encoder, config=None, reference_bundle=None):
    config = config or ProbeConfig()
    output = Path(output)
    rows = load_manifest(manifest, require_split=True)
    selected = [r for r in rows if r['split'] in {'train', 'development'}
                and r['label'] in {'drone', 'non_drone'} and r['annotation_status'] != 'unknown']
    train = [r for r in selected if r['split'] == 'train']
    development = [r for r in selected if r['split'] == 'development']
    for name, records in [('train', train), ('development', development)]:
        if {r['label'] for r in records} != {'drone', 'non_drone'}:
            raise ValueError(f'{name} needs both known recording classes')
    reference = None
    if reference_bundle:
        from .model import load_model
        reference = load_model(reference_bundle, require_frozen=True)
        if (reference.config.sample_rate != config.sample_rate
                or reference.config.window_samples != round(config.window_seconds * config.sample_rate)):
            raise ValueError('Reference must accept the identical sample rate and context')
    plan = {'schema_version': 1, 'manifest_sha256': file_hash(manifest),
            'encoder': encoder.identity, 'config': asdict(config), 'code': revision(),
            'versions': versions(), 'reference_model_id': reference.model_id if reference else None,
            'records': [{'id': r['id'], 'sha256': r['sha256'], 'group_id': r['group_id'],
                         'split': r['split'], 'label': r['label']} for r in selected],
            'supervision': 'one recording label on mean of uniformly sampled context embeddings; no window targets',
            'window_policy': 'physical channel selection, offline polyphase resample, no gain normalization; right-zero-pad short contexts; uniform cap includes endpoints if cap>=2, first context if cap=1; no silence exclusion',
            'threshold_policy': 'development-only fixed grid; highest macro source balanced accuracy, lowest negative recording FPR, closest to .5, highest threshold',
            'final_policy': 'final metadata may be validated; final audio is never decoded or scored',
            'field_reliability_claim': False}
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'plan.json'
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError('Run identity changed; choose a new output directory')
    else:
        atomic_json(plan_path, plan)
    features = {}
    for record in selected:
        features[record['id']] = cached_features(record, manifest, encoder, config, cache)
        print(f"Features {record['id']}: {len(features[record['id']]['vectors'])} contexts "
              f"({'cached' if features[record['id']]['cache_hit'] else 'computed'})", flush=True)
    means = np.asarray([features[r['id']]['vectors'].mean(axis=0) for r in train])
    labels = np.asarray([int(r['label'] == 'drone') for r in train])
    weights = group_class_weights(train)
    scaler = StandardScaler().fit(means, sample_weight=weights)
    with threadpool_limits(limits=1):
        head = LogisticRegression(C=config.regularization_c, max_iter=2000,
                                  random_state=config.seed).fit(scaler.transform(means), labels,
                                                               sample_weight=weights)
    if head.n_iter_[0] >= 2000:
        raise ValueError('Linear classifier failed to converge')
    arrays = {'mean': scaler.mean_, 'scale': scaler.scale_, 'coef': head.coef_[0],
              'intercept': head.intercept_}
    meta = {'schema_version': 'frozen-audio-probe-v1', 'encoder': encoder.identity,
            'config': asdict(config), 'feature_dimension': means.shape[1],
            'run_plan_sha256': file_hash(plan_path), 'class_mapping': {'0': 'non_drone', '1': 'drone'},
            'normalization': 'weighted TRAIN recording means only',
            'sample_weight': 'equal group mass within class; equal class mass',
            'scope': 'development recording-level probe; timestamps are diagnostic; no field/event claim'}
    provisional = ProbeModel({**meta, 'model_id': 'not-yet-frozen'}, arrays, encoder)
    scores = np.asarray([expit(provisional.logits(features[r['id']]['vectors'].mean(axis=0)))
                         for r in development])
    threshold, trials = select_threshold(development, scores)
    meta['recording_threshold'] = threshold
    # Window alarm rule is a disclosed diagnostic default, not tuned event performance.
    meta['decision'] = {'on_threshold': threshold, 'off_threshold': .75 * threshold,
                        'on_hits': 1, 'off_hits': 2}
    bundle = output / 'head'
    if (bundle / 'model.json').exists():
        model = load_probe(bundle, encoder)
        if model.metadata['run_plan_sha256'] != meta['run_plan_sha256']:
            raise ValueError('Head belongs to another run')
        if any(not np.array_equal(model.arrays[k], arrays[k]) for k in arrays):
            raise ValueError('Refit did not reproduce saved head')
    else:
        bundle.mkdir(exist_ok=True)
        with (bundle / 'parameters.npz.partial').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        (bundle / 'parameters.npz.partial').replace(bundle / 'parameters.npz')
        meta['parameters_sha256'] = file_hash(bundle / 'parameters.npz')
        meta['model_id'] = digest(meta)
        atomic_json(bundle / 'model.json', meta)
        model = load_probe(bundle, encoder)
    predictions = []
    for record, score in zip(development, scores, strict=True):
        entry = features[record['id']]
        logits = model.logits(entry['vectors'])
        predictions.append({**metric_rows([record], [score], threshold)[0],
                            'label_scope': record['annotation_status'],
                            'aggregation': 'sigmoid(mean context logits)', 'cache_key': entry['cache_key']})
        trace = [{'window_start_seconds': int(a) / config.sample_rate,
                  'timestamp_seconds': int(b) / config.sample_rate,
                  'padded': int(b - a) < round(config.window_seconds * config.sample_rate),
                  'logit': float(z), 'score': float(expit(z)),
                  'scope': 'diagnostic context score; no per-window truth', 'model_id': model.model_id}
                 for (a, b), z in zip(entry['bounds'], logits, strict=True)]
        write_jsonl(output / 'traces' / f"{digest(record['id'])[:20]}.jsonl", trace)
    summary = metrics(predictions)
    sources = {s: metrics([r for r in predictions if r['source'] == s])
               for s in sorted({r['source'] for r in predictions})}
    report = {'plan_sha256': file_hash(plan_path), 'model_id': model.model_id,
              'scope': 'DEVELOPMENT recording-label comparison; model/threshold selection used these labels',
              'summary': summary, 'source_breakdown': sources,
              'macro_source_balanced_accuracy': macro_balanced(predictions),
              'threshold': threshold, 'threshold_trials': trials, 'predictions': predictions,
              'training_recordings': len(train), 'development_recordings': len(development),
              'training_groups': len({r['group_id'] for r in train}),
              'cache_hits': sum(f['cache_hit'] for f in features.values()),
              'feature_recordings': len(features), 'field_reliability_claim': False,
              'excluded_final_recordings': sum(r['split'] == 'final' for r in rows),
              'excluded_unknown_or_uncertain_recordings': sum(r not in selected and r['split'] != 'final' for r in rows)}
    if reference:
        reference_scores = []
        for record in development:
            audio = decode_record(record, manifest, config)
            bounds = features[record['id']]['bounds']
            probabilities = np.asarray([reference.score(padded_context(audio, a, b, config))
                                        for a, b in bounds])
            probability = float(expit(np.mean(np.log(np.clip(probabilities, 1e-12, 1 - 1e-12)
                                                      / np.clip(1 - probabilities, 1e-12, 1)))))
            reference_scores.append(probability)
        reference_threshold, reference_trials = select_threshold(development, reference_scores)
        reference_rows = metric_rows(development, reference_scores, reference_threshold)
        report['reference'] = {'model_id': reference.model_id,
            'weights': 'unchanged fixed reference; recording threshold selected on this DEVELOPMENT set',
            'aggregation': 'sigmoid(mean clipped context logits), identical selected crops',
            'threshold': reference_threshold, 'threshold_trials': reference_trials,
            'summary': metrics(reference_rows), 'predictions': reference_rows,
            'source_breakdown': {s: metrics([r for r in reference_rows if r['source'] == s])
                                 for s in sorted({r['source'] for r in reference_rows})}}
    atomic_json(output / 'comparison.json', report)
    return report
