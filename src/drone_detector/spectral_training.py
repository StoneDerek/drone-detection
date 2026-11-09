"""Bounded, recording-level experiments for the spectrogram comparison lab."""

import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from .common import digest, file_hash, read_json, write_json
from .data import checked_audio, load_manifest
from .model import save_model
from .spectrogram import axes, build_cnn, cnn_spec, descriptors, spectrogram, tensorflow


def prepare_cache(manifest, config, destination):
    """Training/development only; verified, uniformly sampled available contexts."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    identity = {'manifest_sha256': file_hash(manifest), 'features': config.to_dict(),
                'frontend_source_sha256': file_hash(Path(__file__).with_name('spectrogram.py')),
                'cap': 60, 'splits': ['train', 'development']}
    index = destination / 'index.json'
    if index.exists():
        cached = read_json(index)
        if cached['identity'] != identity:
            raise ValueError('Feature cache identity changed; choose a new destination')
        for row in cached['records']:
            if file_hash(destination / row['cache']) != row['cache_sha256']:
                raise ValueError('Feature cache content changed')
        return cached
    records = [r for r in load_manifest(manifest, True) if r['split'] in identity['splits']]
    inventory = []
    for number, record in enumerate(records):
        audio = checked_audio(record, manifest, config.sample_rate)
        starts = list(range(0, max(1, len(audio) - config.window_samples + 1), config.hop_samples))
        if len(starts) > 60:
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, 60).astype(int)]
        chunks = [(start, audio[start:start + config.window_samples]) for start in starts]
        chunks = [(start, wave) for start, wave in chunks if len(wave) >= config.sample_rate * config.min_audio_seconds
                  and np.sqrt(np.mean(wave.astype(np.float64) ** 2)) >= config.silence_rms]
        if not chunks:
            continue
        name = record['id'] + '.npy'
        images = np.stack([spectrogram(wave, config) for _, wave in chunks])
        np.save(destination / name, images, allow_pickle=False)
        inventory.append({**{k: record[k] for k in ['id', 'source', 'group_id', 'sha256', 'split', 'label', 'origin']},
                          'starts': [start for start, _ in chunks], 'sampled_windows': len(chunks),
                          'cache': name, 'cache_sha256': file_hash(destination / name)})
        if (number + 1) % 40 == 0:
            print(f'Cache {config.axis}/{config.intensity}/{config.n_fft}: {number + 1}/{len(records)} files', flush=True)
    cached = {'identity': identity, 'shape': [len(axes(config)[0]), len(axes(config)[1])], 'records': inventory}
    write_json(index, cached)
    return cached


def _metadata(plan, config, records, family):
    return {'schema_version': 2, 'status': 'trained', 'family': family,
        'features': config.to_dict(), 'class_mapping': {'0': 'non_drone', '1': 'drone'},
        'experimental': True, 'code': plan.get('implementation_code', plan['code']),
        'experiment_plan_sha256': digest({k: v for k, v in plan.items() if k != 'implementation_code'}),
        'limitations': 'DEVELOPMENT-ONLY spectrogram experiment. Publisher recording labels; no fresh independent assessment, event timing, range or field-reliability claim.',
        'training': {'manifest_sha256': plan['manifest_sha256'], 'origins': sorted({r['origin'] for r in records}),
                     'records': records, 'label_scope': 'Publisher recording presence; positive bags can contain negative windows'}}


def train_cnn(manifest, cache, config, output, plan, seed=42, epochs=8, variant='compact'):
    begin = time.perf_counter()
    cached = read_json(Path(cache) / 'index.json')
    if cached['identity']['manifest_sha256'] != file_hash(manifest) or file_hash(manifest) != plan['manifest_sha256']:
        raise ValueError('Training/cache/plan manifests differ')
    records = [r for r in cached['records'] if r['split'] == 'train']
    development = [r for r in cached['records'] if r['split'] == 'development']
    bags = [np.load(Path(cache) / r['cache'], mmap_mode='r', allow_pickle=False) for r in records]
    dev_bags = [np.load(Path(cache) / r['cache'], mmap_mode='r', allow_pickle=False) for r in development]
    y = np.array([r['label'] == 'drone' for r in records], dtype=np.float32)
    dy = np.array([r['label'] == 'drone' for r in development], dtype=np.float32)
    if set(y) != {0., 1.} or set(dy) != {0., 1.}:
        raise ValueError('Both training and development classes required')
    tf = tensorflow()
    tf.keras.backend.clear_session()
    net = build_cnn(seed, variant)
    optimizer = tf.keras.optimizers.Adam(learning_rate=.001, clipnorm=1.)
    positive_weight, negative_weight = len(y) / (2 * y.sum()), len(y) / (2 * (1 - y).sum())
    f, t = cached['shape']

    @tf.function(input_signature=[tf.TensorSpec([None, f, t, 1], tf.float32),
        tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.float32)])
    def step(images, indices, labels):
        with tf.GradientTape() as tape:
            logits = tf.reshape(net(images, training=True), [-1])
            count = tf.shape(labels)[0]
            positives = tf.math.unsorted_segment_max(logits, indices, count)
            negatives = tf.math.unsorted_segment_mean(tf.nn.softplus(logits), indices, count)
            loss = tf.where(labels > .5, tf.nn.softplus(-positives), negatives)
            weights = tf.where(labels > .5, tf.cast(positive_weight, tf.float32), tf.cast(negative_weight, tf.float32))
            objective = tf.reduce_mean(loss * weights) + tf.add_n(net.losses)
        gradients = tape.gradient(objective, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        return objective

    forward = tf.function(lambda x: net(x, training=False),
                          input_signature=[tf.TensorSpec([None, f, t, 1], tf.float32)])

    def validation():
        losses = []
        for bag, label in zip(dev_bags, dy, strict=True):
            logits = np.concatenate([forward(np.asarray(bag[start:start + 16])[..., None]).numpy().ravel()
                                     for start in range(0, len(bag), 16)])
            losses.append(float(np.logaddexp(0, -logits.max()) if label else np.logaddexp(0, logits).mean()))
        values = np.asarray(losses)
        return float((values[dy == 1].mean() + values[dy == 0].mean()) / 2)

    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(bag)) for bag in bags]
    cursors = np.zeros(len(bags), dtype=int)
    visited = [set() for _ in bags]
    history, best, best_weights, selected_epoch, selected_seen = [], np.inf, None, None, None
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(bags))
        losses = []
        for start in range(0, len(order), 8):
            group = order[start:start + 8]
            selected, bag_ids = [], []
            for local, index in enumerate(group):
                count = min(8, len(bags[index]))
                choices = [int(permutations[index][(cursors[index] + j) % len(bags[index])]) for j in range(count)]
                cursors[index] += count
                visited[index].update(choices)
                selected.extend(bags[index][choices])
                bag_ids.extend([local] * count)
            loss = step(np.asarray(selected)[..., None], np.asarray(bag_ids, dtype=np.int32), y[group])
            losses.append(float(loss))
        item = {'epoch': epoch, 'training_loss': float(np.mean(losses)),
                'unique_train_windows_seen': sum(len(v) for v in visited),
                'elapsed_seconds': time.perf_counter() - begin}
        if epoch % 2 == 0 or epoch == epochs:
            item['development_balanced_bag_loss'] = validation()
            if item['development_balanced_bag_loss'] < best:
                best, best_weights = item['development_balanced_bag_loss'], net.get_weights()
                selected_epoch, selected_seen = epoch, item['unique_train_windows_seen']
        history.append(item)
        print(f'CNN {config.axis}/{config.intensity}/{config.n_fft}/{config.mel_bins} seed {seed}: {item}', flush=True)
    metadata = _metadata(plan, config, records, 'spectrogram_cnn')
    metadata['architecture'] = {'name': 'compact 2D CNN trained from scratch', 'input': '[batch, frequency, time, 1]',
        'convolutions': [{k: v for k, v in layer.items() if k != 'name'} for layer in cnn_spec(variant)],
        'activations': 'ReLU', 'pool': 'global average', 'dropout': .1, 'head': 'linear scalar logit',
        'parameters': int(net.count_params()), 'gradcam_layers': ['conv1', 'conv2', 'features']}
    metadata['weight_shapes'] = [list(w.shape) for w in best_weights]
    metadata['architecture']['variant'] = variant
    if variant == 'frequency_context':
        metadata['architecture']['name'] = 'frequency-context 2D CNN trained from scratch'
        metadata['architecture']['frequency_receptive_field_bins'] = 153
    elif variant in {'coverage', 'coverage_medium'}:
        metadata['architecture']['name'] = '2D CNN with complete input coverage'
        metadata['architecture']['frequency_receptive_field_bins'] = 21
        metadata['architecture']['frequency_output_stride_bins'] = 18
    else:
        metadata['architecture']['frequency_receptive_field_bins'] = 25
    metadata['training'].update({'seed': seed, 'epochs_completed': epochs, 'selected_epoch': selected_epoch,
        'max_windows_per_bag_step': 8, 'bag_batch_size': 8, 'optimizer': 'Adam lr .001, clipnorm 1, L2 1e-4',
        'objective': 'Positive bag: softplus(-max window logit); negative bag: mean softplus(window logit); balanced class and equal recording mass',
        'selection': 'Minimum balanced publisher-bag development loss, every 2 epochs, max 60 uniform contexts per file',
        'history': history, 'unique_windows_seen_at_selected_epoch': selected_seen,
        'available_sampled_windows': sum(len(bag) for bag in bags),
        'seconds': time.perf_counter() - begin})
    return save_model(output, metadata, {f'w{i}': value for i, value in enumerate(best_weights)})


def train_svm(cache, config, output, plan, seed=42):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    begin = time.perf_counter()
    cached = read_json(Path(cache) / 'index.json')
    if cached['identity']['manifest_sha256'] != plan['manifest_sha256'] or cached['identity']['features'] != config.to_dict():
        raise ValueError('SVM cache differs from planned inputs')
    records = [r for r in cached['records'] if r['split'] == 'train']
    bags = [descriptors(np.load(Path(cache) / r['cache'], mmap_mode='r', allow_pickle=False)).astype(np.float64)
            for r in records]
    y = np.asarray([r['label'] == 'drone' for r in records], dtype=int)
    weight = np.concatenate([np.full(len(bag), 1 / len(bag)) for bag in bags])
    scaler = StandardScaler().fit(np.concatenate(bags), sample_weight=weight)
    bags = [scaler.transform(bag) for bag in bags]
    model = SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced', random_state=seed, max_iter=20000)
    model.fit(np.stack([bag.mean(axis=0) for bag in bags]), y)
    for iteration in range(4):
        x, target, weights = [], [], []
        for bag, label in zip(bags, y, strict=True):
            chosen = bag[np.argmax(model.decision_function(bag))][None] if label else bag
            x.extend(chosen)
            target.extend([label] * len(chosen))
            weights.extend([1 / len(chosen)] * len(chosen))
        target, weights = np.asarray(target), np.asarray(weights)
        for label in [0, 1]:
            weights[target == label] *= len(target) / (2 * weights[target == label].sum())
        model = SVC(C=1, kernel='rbf', gamma='scale', random_state=seed, max_iter=20000, cache_size=512)
        model.fit(np.asarray(x), target, sample_weight=weights)
        if model.fit_status_:
            raise ValueError('SVM failed to converge')
        print(f'Non-neural SVM instance-selection update {iteration + 1}/4', flush=True)
    metadata = _metadata(plan, config, records, 'rbf_svm')
    metadata['gamma'] = float(model._gamma)
    metadata['training'].update({'seed': seed, 'C': 1., 'kernel': 'RBF',
        'method': 'Temporal mean/std per frequency; train-only weighted standardization; four max-positive-instance updates; all sampled negative windows',
        'seconds': time.perf_counter() - begin, 'support_vectors': len(model.support_vectors_)})
    arrays = {'mean': scaler.mean_, 'scale': scaler.scale_, 'support_vectors': model.support_vectors_,
              'dual_coef': model.dual_coef_[0], 'intercept': model.intercept_}
    # Check the portable numeric implementation against sklearn before saving.
    from .spectrogram import SpectralModel
    portable = SpectralModel({**metadata, 'model_id': 'verification'}, arrays)
    raw_images = np.load(Path(cache) / records[0]['cache'], allow_pickle=False)
    expected = expit(model.decision_function(scaler.transform(descriptors(raw_images).astype(np.float64))))
    np.testing.assert_allclose(portable.score_images(raw_images), expected, atol=1e-10, rtol=1e-8)
    return save_model(output, metadata, arrays)
