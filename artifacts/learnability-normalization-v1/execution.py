#!/usr/bin/env python3
"""Fixed within-session scratch-CNN learning and amplitude-reference controls."""

import hashlib
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from drone_detector.audio import decode, normalize_context_gain
from drone_detector.common import digest, file_hash, hardware, read_json, write_json
from drone_detector.model import save_model
from drone_detector.spectrogram import SpectrogramConfig, build_cnn, cnn_spec, spectrogram, tensorflow

ROOT = Path('runs/learnability-normalization-v1')
REPORT = Path('reports/learnability-normalization-v1')
ARTIFACT = Path('artifacts/learnability-normalization-v1')
DATA = Path('runs/external-v3/downloads/itu/itu-aris-lab-acoustic-drone-dataset')
BASE = SpectrogramConfig(axis='linear_filterbank', mel_bins=256, window_seconds=1.)
ARMS = {'peak': BASE, 'fixed': replace(BASE, power_reference='unit_stft_power'),
        'rms-fixed': replace(BASE, power_reference='unit_stft_power', normalize_rms=True)}


def interval_class_weights(items, assignment, split):
    selected = [i for i, row in enumerate(items) if assignment[row['file']] == split]
    counts = Counter(items[i]['group'] for i in selected)
    group_labels = {items[i]['group']: items[i]['label'] for i in selected}
    classes = Counter(group_labels.values())
    if set(classes) != {'drone', 'background'}:
        raise ValueError('Both published classes are required')
    weights = np.zeros(len(items), dtype=np.float32)
    for i in selected:
        row = items[i]
        weights[i] = len(selected) / (2 * classes[row['label']] * counts[row['group']])
    return np.array(selected), weights


def metrics(items, indices, y, scores):
    ids = np.asarray(indices)
    truth, selected = y[ids], scores[ids]
    by_group = {}
    for i in ids:
        group = items[i]['group']
        previous = by_group.get(group, (int(y[i]), -1.))
        by_group[group] = (int(y[i]), max(previous[1], float(scores[i])))
    group_y, group_score = np.array(list(by_group.values())).T
    return {'segments': len(ids), 'positive_segments': int(truth.sum()),
            'confusion_tn_fp_fn_tp': confusion_matrix(truth, selected >= .5, labels=[0, 1]).ravel().tolist(),
            'segment_auc': float(roc_auc_score(truth, selected)),
            'publisher_groups': len(by_group),
            'group_confusion_tn_fp_fn_tp': confusion_matrix(group_y, group_score >= .5, labels=[0, 1]).ravel().tolist(),
            'group_auc': float(roc_auc_score(group_y, group_score))}


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    items = read_json(DATA / 'groups.json')['segments']
    assignment = read_json(DATA / 'splits/split_interval_disjoint.json')['assignment']
    split_audit = read_json('reports/human-label-task-v1/split-audit.json')
    assert not split_audit['groups_crossing_splits'] and not split_audit['raw_half_second_fingerprints_crossing_splits']
    y = np.array([int(item['label'] == 'drone') for item in items], dtype=np.float32)
    split = {name: interval_class_weights(items, assignment, name) for name in ['train', 'val', 'test']}
    execution = {'plan_sha256': file_hash('reports/transfer-diagnosis-v1/plan.json'),
                 'code_sha256': file_hash(__file__), 'hardware': hardware(),
                 'source_groups_sha256': file_hash(DATA / 'groups.json'),
                 'source_split_sha256': file_hash(DATA / 'splits/split_interval_disjoint.json'),
                 'split_audit_sha256': file_hash('reports/human-label-task-v1/split-audit.json'),
                 'arms': {name: cfg.to_dict() for name, cfg in ARMS.items()},
                 'scope': 'CONSUMED-data released-label feasibility. No new audible truth, independent assessment or detector promotion.',
                 'epochs': 24, 'seeds': [42, 7], 'batch_size': 64,
                 'target': 'Released one-second class labels; segment BCE; equal interval mass within class and equal class mass.',
                 'selection': 'Lowest deterministic weighted validation BCE every2epochs; report test descriptively after selecting all checkpoints.'}
    path = REPORT / 'execution.json'
    if path.exists():
        if read_json(path) != execution:
            raise ValueError('Execution identity changed')
    else:
        write_json(path, execution)
    caches = {}
    for name, cfg in ARMS.items():
        destination = ROOT / f'{name}.npy'
        marker = ROOT / f'{name}-cache.json'
        identity = {'source_sha256': execution['source_groups_sha256'], 'features': cfg.to_dict(), 'shape': [len(items), 256, 47],
                    'code_sha256': execution['code_sha256']}
        if marker.exists():
            if read_json(marker)['identity'] != identity:
                raise ValueError('Feature cache identity changed')
            caches[name] = np.load(destination, mmap_mode='r')
            continue
        images = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32, shape=(len(items), 256, 47))
        valid_rms, rms_reached, clipped = [], 0, 0
        for index, item in enumerate(items):
            wave, _ = decode(DATA / item['file'], 16000)
            if len(wave) != 16000:
                raise ValueError('Exactly one second required, without padding')
            images[index] = spectrogram(wave, cfg)
            if cfg.normalize_rms:
                gained = normalize_context_gain(wave, cfg)
                rms = float(np.sqrt(np.mean(gained.astype(np.float64) ** 2)))
                valid_rms.append(rms)
                rms_reached += abs(rms - cfg.gain_target_rms) <= 1e-5
                clipped += int(np.max(np.abs(gained)) >= .999)
            if (index + 1) % 1000 == 0:
                print('CACHE', name, index + 1, '/', len(items), flush=True)
        images.flush()
        write_json(marker, {'identity': identity, 'cache_sha256': file_hash(destination),
                           'rms_target_reached': int(rms_reached), 'waveform_clipped': clipped,
                           'post_gain_rms_range': [min(valid_rms), max(valid_rms)] if valid_rms else None})
        caches[name] = np.load(destination, mmap_mode='r')
    return items, y, split, caches


def predictions(net, images, indices, batch_size=128):
    values = []
    for start in range(0, len(indices), batch_size):
        batch = images[indices[start:start + batch_size]]
        values.extend(net(batch[..., None], training=False).numpy().ravel())
    return np.asarray(values)


def train_one(name, images, config, seed, items, y, split):
    destination = ARTIFACT / f'{name}-seed{seed}'
    if (destination / 'model.json').exists():
        print('Preserve completed arm', destination, flush=True)
        return read_json(REPORT / f'{name}-seed{seed}.json')
    tf = tensorflow()
    net = build_cnn(seed=seed, variant='frequency_aware', dropout_rate=0.)
    assert net.count_params() == 154689
    initial = hashlib.sha256(b''.join(w.tobytes() for w in net.get_weights())).hexdigest()
    optimizer = tf.keras.optimizers.Adam(learning_rate=.001, clipnorm=1.)
    train_ids, train_weights = split['train']
    val_ids, val_weights = split['val']
    started, best, history = time.perf_counter(), None, []

    @tf.function(input_signature=[tf.TensorSpec([None, 256, 47, 1], tf.float32),
                                 tf.TensorSpec([None], tf.float32), tf.TensorSpec([None], tf.float32)])
    def step(x, target, weight):
        with tf.GradientTape() as tape:
            logits = tf.reshape(net(x, training=True), [-1])
            classification = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits) * weight)
            loss = classification + tf.add_n(net.losses)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        return loss

    for epoch in range(1, 25):
        order = np.random.default_rng(seed * 1000 + epoch).permutation(train_ids)
        losses = []
        for start in range(0, len(order), 64):
            ids = order[start:start + 64]
            losses.append(float(step(images[ids][..., None], y[ids], train_weights[ids])))
        if epoch % 2 == 0:
            validation_logits = predictions(net, images, val_ids)
            validation_bce = np.average(np.logaddexp(0, validation_logits) - y[val_ids] * validation_logits, weights=val_weights[val_ids])
            train_logits = predictions(net, images, train_ids)
            train_bce = np.average(np.logaddexp(0, train_logits) - y[train_ids] * train_logits, weights=train_weights[train_ids])
            entry = {'epoch': epoch, 'weighted_train_bce': float(train_bce),
                     'weighted_validation_bce': float(validation_bce), 'mean_update_loss_with_l2': float(np.mean(losses)),
                     'elapsed_seconds': time.perf_counter() - started}
            history.append(entry)
            if best is None or validation_bce < best[0]:
                best = (float(validation_bce), epoch, net.get_weights())
            print('TRAIN', name, seed, entry, flush=True)
    last_weights = net.get_weights()
    net.set_weights(best[2])
    # Test predictions are produced only after this arm's checkpoint is selected;
    # they never guide weights, epochs, normalization arms or thresholds.
    logits = predictions(net, images, np.arange(len(items)))
    scores = expit(logits)
    report = {'name': name, 'seed': seed, 'parameters': net.count_params(), 'initial_weights_sha256': initial,
              'selected_epoch': best[1], 'history': history,
              'seconds': time.perf_counter() - started,
              'splits': {key: metrics(items, indices, y, scores) for key, (indices, _) in split.items()},
              'source': 'One consumed ITU session, published labels; overlapping clips are not independent examples.'}
    metadata = {'schema_version': 2, 'status': 'trained', 'family': 'spectrogram_cnn', 'features': config.to_dict(),
                'class_mapping': {'0': 'non_drone', '1': 'drone'},
                'architecture': {'variant': 'frequency_aware', 'parameters': 154689, 'dropout': 0.,
                                 'convolutions': [{k: v for k, v in layer.items() if k != 'name'} for layer in cnn_spec('frequency_aware')]},
                'weight_shapes': [list(w.shape) for w in best[2]],
                'training': {'seed': seed, 'selected_epoch': best[1], 'epochs_completed': 24,
                             'target': 'Publisher one-second segment labels, interval/class-balanced BCE', 'history': history,
                             'execution_sha256': file_hash(REPORT / 'execution.json')},
                'limitations': 'Same-session consumed-label diagnostic, no audible event or field-reliability validation. No product promotion.'}
    model = save_model(destination, metadata, {f'w{i}': w for i, w in enumerate(best[2])})
    report['model_id'] = model.model_id
    np.savez_compressed(destination / 'last-parameters.npz', **{f'w{i}': w for i, w in enumerate(last_weights)})
    np.save(ROOT / f'{name}-seed{seed}-scores.npy', scores)
    write_json(REPORT / f'{name}-seed{seed}.json', report)
    return report


def linear_control(name, images, items, y, split):
    vectors = np.concatenate([images.mean(axis=2), images.std(axis=2)], axis=1).astype(np.float64)
    ids, weights = split['train']
    scaler = StandardScaler().fit(vectors[ids], sample_weight=weights[ids])
    standardized = scaler.transform(vectors)
    classifier = LogisticRegression(C=1, max_iter=2000, random_state=42)
    classifier.fit(standardized[ids], y[ids], sample_weight=weights[ids])
    if classifier.n_iter_[0] >= 2000:
        raise ValueError('Fixed non-neural control did not converge')
    scores = classifier.predict_proba(standardized)[:, 1]
    destination = ARTIFACT / f'{name}-logistic'
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination / 'parameters.npz', mean=scaler.mean_, scale=scaler.scale_, coef=classifier.coef_[0], intercept=classifier.intercept_)
    result = {'family': '512descriptor logistic regression', 'C': 1, 'input_arm': name,
              'training': 'Identical published segment targets and interval/class weights; train-only standardization.',
              'splits': {key: metrics(items, indices, y, scores) for key, (indices, _) in split.items()},
              'parameters_sha256': file_hash(destination / 'parameters.npz')}
    write_json(destination / 'model.json', result)
    write_json(REPORT / f'{name}-logistic.json', result)
    np.save(ROOT / f'{name}-logistic-scores.npy', scores)
    return result


def main():
    items, y, split, caches = prepare()
    tf = tensorflow()
    tf.config.experimental.enable_op_determinism()
    results, controls = {}, {}
    for name, cfg in ARMS.items():
        controls[name] = linear_control(name, caches[name], items, y, split)
        for seed in [42, 7]:
            results[f'{name}-seed{seed}'] = train_one(name, caches[name], cfg, seed, items, y, split)
    for seed in [42, 7]:
        assert len({results[f'{name}-seed{seed}']['initial_weights_sha256'] for name in ARMS}) == 1
    write_json(REPORT / 'comparison.json', {'neural': results, 'non_neural': controls,
               'execution_sha256': file_hash(REPORT / 'execution.json'), 'scope': 'Consumed single-session learning diagnosis; descriptive reuse, not independent validation.'})
    print('COMPLETE all six paired CNN runs and three fixed non-neural controls', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
