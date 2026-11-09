"""Controlled capacity/weak-supervision probes, with explicit instance audits."""

import gzip
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from .common import canonical, file_hash, read_json, write_json
from .data import load_manifest
from .model import save_model
from .spectral_training import _metadata
from .spectrogram import SpectrogramConfig, build_cnn, cnn_spec, tensorflow


class BalancedRecordingBatches:
    """Equal class counts; every class member is visited before its next cycle."""

    def __init__(self, labels, seed):
        self.rng = np.random.default_rng(seed)
        self.members = [np.flatnonzero(labels == value) for value in [0, 1]]
        if any(not len(group) for group in self.members):
            raise ValueError('Balanced batches require both classes')
        self.queues = [self.rng.permutation(group) for group in self.members]
        self.cursors = [0, 0]

    def take(self):
        selected = []
        for label in [0, 1]:
            for _ in range(4):
                if self.cursors[label] == len(self.queues[label]):
                    self.queues[label] = self.rng.permutation(self.members[label])
                    self.cursors[label] = 0
                selected.append(int(self.queues[label][self.cursors[label]]))
                self.cursors[label] += 1
        return self.rng.permutation(selected)


def pool_positive_logits(logits, indices, count, temperature=None):
    """Max or normalized logsumexp; equal logits pool identically at any count."""
    tf = tensorflow()
    maximum = tf.math.unsorted_segment_max(logits, indices, count)
    if temperature is None:
        ties = tf.cast(logits == tf.gather(maximum, indices), tf.float32)
        weights = ties / tf.gather(tf.math.unsorted_segment_sum(ties, indices, count), indices)
        return maximum, weights
    if temperature <= 0:
        raise ValueError('Positive smoothing temperature required')
    shifted = (logits - tf.gather(maximum, indices)) / temperature
    mass = tf.exp(shifted)
    total = tf.math.unsorted_segment_sum(mass, indices, count)
    number = tf.math.unsorted_segment_sum(tf.ones_like(logits), indices, count)
    pooled = maximum + temperature * (tf.math.log(total) - tf.math.log(number))
    return pooled, mass / tf.gather(total, indices)


def recording_losses(logits, indices, labels, *, consistent=False, temperature=None):
    """Weak recording labels require the same event score for both classes."""
    tf = tensorflow()
    pooled, weights = pool_positive_logits(logits, indices, tf.shape(labels)[0], temperature)
    negative = (tf.nn.softplus(pooled) if consistent else
                tf.math.unsorted_segment_mean(tf.nn.softplus(logits), indices, tf.shape(labels)[0]))
    return tf.where(labels > .5, tf.nn.softplus(-pooled), negative), weights


def full_bag_max_indices(images, forward, batch_size=16):
    """Select all full-bag maxima without retaining an activation graph.

    Recomputing only these instances gives the full max objective's gradient
    when forward and training modes are identical. Exact ties are all retained
    so segment-max differentiation averages their gradients as usual.
    """
    if len(images) == 0 or batch_size < 1:
        raise ValueError('Full-recording selection requires nonempty inputs')
    logits = np.concatenate([np.asarray(forward(np.asarray(images[i:i+batch_size])[..., None])).ravel()
                             for i in range(0, len(images), batch_size)])
    if not np.isfinite(logits).all() or len(logits) != len(images):
        raise ValueError('Invalid full-recording selection logits')
    return np.flatnonzero(logits == logits.max())


def train_foundation_cnn(manifest, cache, config, output, plan, *, seed=42, epochs=24,
                         full_positive=False, smooth_positive=False, snapshot_epochs=(8,),
                         balanced_batches=False, learning_rate=.001, variant='coverage_medium',
                         consistent_loss=False, dropout_rate=.1, full_recording_bags=False):
    """Same negative sampling/optimizer as the old trial; change named factors."""
    started = time.perf_counter()
    if full_recording_bags and (not consistent_loss or dropout_rate != 0 or smooth_positive or full_positive):
        raise ValueError('Exact full-recording selection requires symmetric max loss, dropout0 and no other pooling/sampling override')
    output, cache = Path(output), Path(cache)
    audit = output.parent / (output.name + '-audit')
    audit.mkdir(parents=True, exist_ok=True)
    index = read_json(cache / 'index.json')
    if file_hash(manifest) != plan['manifest_sha256'] or index['identity']['manifest_sha256'] != plan['manifest_sha256']:
        raise ValueError('Manifest/cache/plan disagree')
    if SpectrogramConfig(**index['identity']['features']).to_dict() != config.to_dict():
        raise ValueError('Cache preprocessing differs from planned frontend')
    manifest_rows = {r['id']: r for r in load_manifest(manifest, True) if r['split'] in {'train', 'development'}}
    for row in index['records']:
        for key in ['source', 'group_id', 'sha256', 'split', 'label', 'origin']:
            if row[key] != manifest_rows[row['id']][key]:
                raise ValueError('Cached recording identity changed')
        if file_hash(cache / row['cache']) != row['cache_sha256']:
            raise ValueError('Cached tensor content changed')
    records = [r for r in index['records'] if r['split'] == 'train']
    development = [r for r in index['records'] if r['split'] == 'development']
    bags = [np.load(cache / r['cache'], mmap_mode='r', allow_pickle=False) for r in records]
    dev_bags = [np.load(cache / r['cache'], mmap_mode='r', allow_pickle=False) for r in development]
    labels = np.asarray([r['label'] == 'drone' for r in records], dtype=np.float32)
    dev_labels = np.asarray([r['label'] == 'drone' for r in development], dtype=np.float32)
    if set(labels) != {0., 1.} or set(dev_labels) != {0., 1.}:
        raise ValueError('Both classes required')
    tf = tensorflow()
    tf.keras.backend.clear_session()
    net = build_cnn(seed, variant, dropout_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=1.)
    positive_weight = 1. if balanced_batches else len(labels) / (2 * labels.sum())
    negative_weight = 1. if balanced_batches else len(labels) / (2 * (1-labels).sum())
    f, t = index['shape']
    temperature = .5 if smooth_positive else None

    @tf.function(input_signature=[tf.TensorSpec([None, f, t, 1], tf.float32),
        tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.float32)])
    def step(images, ids, y):
        with tf.GradientTape() as tape:
            logits = tf.reshape(net(images, training=True), [-1])
            loss, weights = recording_losses(logits, ids, y, consistent=consistent_loss,
                                              temperature=temperature)
            class_weight = tf.where(y > .5, tf.cast(positive_weight, tf.float32), tf.cast(negative_weight, tf.float32))
            objective = tf.reduce_mean(loss * class_weight) + tf.add_n(net.losses)
        gradients = tape.gradient(objective, net.trainable_variables)
        norm = tf.linalg.global_norm(gradients)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables, strict=True))
        return objective, logits, weights, norm

    forward = tf.function(lambda x: net(x, training=False),
        input_signature=[tf.TensorSpec([None, f, t, 1], tf.float32)])

    def evaluate(selected_records, selected_bags, y):
        rows, losses, coherent_losses, maxima = [], [], [], []
        for row, bag, label in zip(selected_records, selected_bags, y, strict=True):
            logits = np.concatenate([forward(np.asarray(bag[s:s+16])[..., None]).numpy().ravel()
                                     for s in range(0, len(bag), 16)])
            loss = float(np.logaddexp(0, -logits.max()) if label else np.logaddexp(0, logits).mean())
            coherent = float(np.logaddexp(0, (1-2*label)*logits.max()))
            score = float(expit(logits.max()))
            losses.append(loss)
            coherent_losses.append(coherent)
            maxima.append(score)
            winner = int(logits.argmax())
            rows.append({k: row[k] for k in ['id', 'source', 'group_id', 'split', 'label']} | {
                'max_score': score, 'bag_loss': loss, 'consistent_max_loss': coherent,
                'winning_start_sample': row['starts'][winner]})
        losses = np.asarray(losses)
        coherent_losses = np.asarray(coherent_losses)
        return {'balanced_bag_loss': float((losses[y == 1].mean()+losses[y == 0].mean())/2),
                'consistent_max_balanced_loss': float((coherent_losses[y == 1].mean()+coherent_losses[y == 0].mean())/2),
                'max_context_auc': float(roc_auc_score(y, maxima)), 'records': rows}

    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(bag)) for bag in bags]
    balanced_sampler = BalancedRecordingBatches(labels, seed+1000) if balanced_batches else None
    cursors = np.zeros(len(bags), dtype=int)
    visited = [set() for _ in bags]
    positive_signal = [set() for _ in bags]
    history, best_loss, best_weights, best_epoch, best_seen = [], np.inf, None, None, None
    selection_path = audit / 'positive-selection.jsonl.gz'

    def metadata(selected_epoch, selected_seen, completed, selection):
        meta = _metadata(plan, config, records, 'spectrogram_cnn')
        meta['architecture'] = {'name': '2D CNN with complete input coverage and an explicit readout',
            'variant': variant, 'input': '[batch, frequency, time, 1]',
            'convolutions': [{k: v for k, v in layer.items() if k != 'name'} for layer in cnn_spec(variant)],
            'activations': 'ReLU', 'pool': 'time mean, retain15 frequency positions' if variant == 'frequency_aware' else 'global average', 'dropout': dropout_rate,
            'head': {'frequency_aware': 'Dense32 ReLU then scalar logit', 'global_readout': 'Dense473 ReLU then scalar logit'}.get(variant, 'linear scalar logit'),
            'parameters': int(net.count_params()),
            'gradcam_layers': ['conv1', 'conv2', 'features'],
            'frequency_receptive_field_bins': 21, 'frequency_output_stride_bins': 18}
        meta['weight_shapes'] = [list(w.shape) for w in net.get_weights()]
        meta['training'].update({'seed': seed, 'epochs_completed': completed, 'selected_epoch': selected_epoch,
            'history': list(history), 'unique_windows_seen_at_selected_epoch': selected_seen,
            'available_sampled_windows': sum(len(bag) for bag in bags), 'bag_batch_size': 8,
            'max_negative_windows_per_step': None if full_recording_bags else 8,
            'positive_sampling': ('Select full cached-bag maxima every update; recompute all tied maxima with gradients' if full_recording_bags else
                                  'all cached contexts' if full_positive else 'sample8 cyclic without replacement'),
            'negative_sampling': 'Same full cached-bag maximum selection as positives' if full_recording_bags else 'sample8 cyclic without replacement',
            'full_recording_bags': full_recording_bags,
            'positive_pool': 'normalized logsumexp temperature .5' if smooth_positive else 'max',
            'negative_loss': 'softplus(max window logit), same pool as positives' if consistent_loss else 'mean softplus(window logits)',
            'optimizer': f'Adam {learning_rate}, clipnorm1, dropout{dropout_rate}, L2 .0001; balanced expected class mass',
            'batch_sampling': 'Four recordings per class, no extra class weights; same51 steps/epoch' if balanced_batches else 'Original random recording batches and inverse class weights',
            'selection': selection, 'seconds': time.perf_counter()-started,
            'positive_audit': 'Per-step pooling weights, selected starts and effective count; not audible annotations'})
        return meta

    with gzip.open(selection_path, 'wb') as stream:
        for epoch in range(1, epochs+1):
            order = rng.permutation(len(bags))
            batches = ([balanced_sampler.take() for _ in range((len(bags)+7)//8)] if balanced_sampler
                       else [order[start:start+8] for start in range(0, len(order), 8)])
            objectives, gradient_norms, effective_counts = [], [], []
            positive_forwarded = positive_nonzero = selection_scanned = 0
            for step_index, group in enumerate(batches):
                selected, local_ids, choices_by_bag = [], [], []
                for local, bag_index in enumerate(group):
                    if full_recording_bags:
                        choices = full_bag_max_indices(bags[bag_index], forward).tolist()
                        selection_scanned += len(bags[bag_index])
                        visited[bag_index].update(range(len(bags[bag_index])))
                        count = len(choices)
                    else:
                        count = len(bags[bag_index]) if full_positive and labels[bag_index] else min(8, len(bags[bag_index]))
                        choices = [int(permutations[bag_index][(cursors[bag_index]+j) % len(bags[bag_index])]) for j in range(count)]
                        cursors[bag_index] += count
                        visited[bag_index].update(choices)
                    selected.extend(bags[bag_index][choices])
                    local_ids.extend([local]*count)
                    choices_by_bag.append(choices)
                objective, logits, weights, norm = step(np.asarray(selected)[..., None],
                    np.asarray(local_ids, dtype=np.int32), labels[group])
                objectives.append(float(objective))
                gradient_norms.append(float(norm))
                if not np.isfinite(objectives[-1]) or not np.isfinite(gradient_norms[-1]):
                    raise ValueError('Nonfinite objective/gradient')
                values, mass, offset = logits.numpy(), weights.numpy(), 0
                for bag_index, choices in zip(group, choices_by_bag, strict=True):
                    count = len(choices)
                    if labels[bag_index]:
                        probabilities = mass[offset:offset+count]
                        active = probabilities > 1e-10
                        positive_forwarded += count
                        positive_nonzero += int(active.sum())
                        positive_signal[bag_index].update(int(choices[j]) for j in np.flatnonzero(active))
                        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30))))
                        effective = float(np.exp(entropy))
                        effective_counts.append(effective)
                        winners = np.argsort(-probabilities, kind='stable')[:3]
                        stream.write(canonical({'epoch': epoch, 'step': step_index,
                            'positive_recordings_in_batch': int(labels[group].sum()),
                            'id': records[bag_index]['id'],
                            'source': records[bag_index]['source'], 'forwarded_contexts': count,
                            'pool_gradient_contexts_above1e_10': int(active.sum()), 'effective_contexts': effective,
                            'top': [{'start_sample': records[bag_index]['starts'][choices[j]],
                                     'weight': float(probabilities[j]), 'logit': float(values[offset+j])} for j in winners]})+b'\n')
                    offset += count
            item = {'epoch': epoch, 'training_loss': float(np.mean(objectives)),
                'mean_gradient_norm': float(np.mean(gradient_norms)),
                'unique_train_windows_seen': sum(len(s) for s in visited),
                'positive_contexts_forwarded': positive_forwarded,
                'selection_contexts_scanned': selection_scanned,
                'positive_pool_gradient_contexts_above1e_10': positive_nonzero,
                'unique_positive_contexts_with_pool_gradient': sum(len(s) for s in positive_signal),
                'mean_effective_positive_contexts': float(np.mean(effective_counts)),
                'elapsed_seconds': time.perf_counter()-started}
            if epoch % 2 == 0 or epoch == epochs:
                train_eval = evaluate(records, bags, labels)
                dev_eval = evaluate(development, dev_bags, dev_labels)
                write_json(audit / f'evaluation-{epoch:02d}.json', {'train': train_eval, 'development': dev_eval})
                item.update({'train_balanced_bag_loss': train_eval['balanced_bag_loss'],
                    'train_consistent_max_balanced_loss': train_eval['consistent_max_balanced_loss'],
                    'development_consistent_max_balanced_loss': dev_eval['consistent_max_balanced_loss'],
                    'train_auc': train_eval['max_context_auc'], 'development_auc': dev_eval['max_context_auc'],
                    'development_balanced_bag_loss': dev_eval['balanced_bag_loss']})
                selection_metric = 'consistent_max_balanced_loss' if consistent_loss else 'balanced_bag_loss'
                if dev_eval[selection_metric] < best_loss:
                    best_loss, best_weights, best_epoch = dev_eval[selection_metric], net.get_weights(), epoch
                    best_seen = item['unique_train_windows_seen']
                np.savez_compressed(audit / f'weights-{epoch:02d}.npz', **{f'w{i}': w for i, w in enumerate(net.get_weights())})
            history.append(item)
            write_json(audit / 'history.json', history)
            print(output.name, item, flush=True)
            if epoch in snapshot_epochs:
                save_model(output.parent / (output.name+f'-at{epoch}'),
                    metadata(best_epoch, best_seen, epoch, f'Minimum development {selection_metric} through this prespecified snapshot'),
                    {f'w{i}': w for i, w in enumerate(best_weights)})
    save_model(output.parent / (output.name+'-last'),
        metadata(epochs, sum(len(s) for s in visited), epochs, 'Last epoch, diagnostic only'),
        {f'w{i}': w for i, w in enumerate(net.get_weights())})
    return save_model(output,
        metadata(best_epoch, best_seen, epochs, f'Minimum development {selection_metric}, every2 epochs, all cached contexts'),
        {f'w{i}': w for i, w in enumerate(best_weights)})
