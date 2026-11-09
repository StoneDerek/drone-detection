#!/usr/bin/env python3
"""Independent real-input gradient/Adam oracle and checkpoint identity audit."""

from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from drone_detector.common import read_json, write_json
from drone_detector.foundation_training import full_bag_max_indices, recording_losses
from drone_detector.model import load_model
from drone_detector.spectrogram import tensorflow


def main():
    tf = tensorflow()
    root = Path('runs/cnn-foundations-v1/trained')
    names = ['coherent-global-seed42', 'coherent-frequency-seed42', 'coherent-frequency-seed7']
    checkpoint_checks = []
    for name in names:
        for suffix in ['', '-last', '-at8']:
            model = load_model(root/(name+suffix))
            epoch = model.metadata['training']['selected_epoch']
            with np.load(root/(name+'-audit')/f'weights-{epoch:02d}.npz', allow_pickle=False) as archive:
                assert all(np.array_equal(value, archive[key]) for key, value in model.arrays.items())
            if not suffix:
                history = model.metadata['training']['history']
                best = min((row for row in history if 'development_consistent_max_balanced_loss' in row),
                           key=lambda row: row['development_consistent_max_balanced_loss'])
                assert best['epoch'] == epoch
                frozen = load_model(Path('artifacts/cnn-foundations-v1/frozen')/name, True)
                assert all(np.array_equal(value, frozen.arrays[key]) for key,value in model.arrays.items())
            checkpoint_checks.append({'name': name+suffix, 'selected_epoch': epoch, 'weights_exact': True})
    cache = Path('runs/spectral-lab-v1/cache/linear256-fft4096')
    index = read_json(cache/'index.json')
    rows = []
    for source in ['d3set-ground', 'svanstrom', 'd3set-noise', 'esc10']:
        candidates = [row for row in index['records'] if row['split'] == 'train' and row['source'] == source
                      and (row['label'] == 'drone' if source == 'svanstrom' else True)]
        rows.extend(candidates[:2])
    bags = [np.load(cache/row['cache'], allow_pickle=False) for row in rows]
    y = tf.constant([float(row['label'] == 'drone') for row in rows])
    assert float(tf.reduce_sum(y)) == 4.
    model = load_model(root/'coherent-frequency-seed42')
    net = model.network()
    forward = lambda x: net(x, training=False)  # noqa: E731

    def objective(bags, oracle=False):
        images = np.concatenate(bags)[...,None]
        ids = np.repeat(np.arange(len(bags)), [len(bag) for bag in bags]).astype(np.int32)
        with tf.GradientTape() as tape:
            logits = tf.reshape(net(images, training=True), [-1])
            if oracle:
                # Independent per-recording Python slicing and standard BCE;
                # no production segment pooling or loss helper.
                maxima, offset = [], 0
                for bag in bags:
                    maxima.append(tf.reduce_max(logits[offset:offset+len(bag)]))
                    offset += len(bag)
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.stack(maxima))
            else:
                losses, _ = recording_losses(logits, ids, y, consistent=True)
            value = tf.reduce_mean(losses)+tf.add_n(net.losses)
        gradients = tape.gradient(value, net.trainable_variables)
        assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients)
        return float(value), [g.numpy() for g in gradients]

    full_value, full_gradients = objective(bags)
    oracle_value, oracle_gradients = objective(bags, oracle=True)
    choices = [full_bag_max_indices(bag, forward) for bag in bags]
    selected_value, selected_gradients = objective([bag[c] for bag,c in zip(bags,choices,strict=True)])
    max_gradient_error = max(float(np.max(abs(a-b))) for a,b in zip(full_gradients,oracle_gradients,strict=True))
    selection_gradient_error = max(float(np.max(abs(a-b))) for a,b in zip(full_gradients,selected_gradients,strict=True))
    assert abs(full_value-oracle_value) < 1e-6 and max_gradient_error < 1e-5
    assert abs(full_value-selected_value) < 2e-5 and selection_gradient_error < 2e-4
    images = np.concatenate([bag[:2] for bag in bags])[...,None]
    mode_error = float(np.max(abs(net(images,training=True).numpy()-net(images,training=False).numpy())))
    batch_error = float(np.max(abs(net(images,training=False).numpy()-np.concatenate([net(image[None],training=False).numpy() for image in images]))))
    assert mode_error == 0 and batch_error < 2e-5
    before = net.get_weights()
    optimizer = tf.keras.optimizers.Adam(.001, clipnorm=1.)
    optimizer.apply_gradients(zip([tf.constant(g) for g in selected_gradients], net.trainable_variables, strict=True))
    after = net.get_weights()
    expected = []
    for value, gradient in zip(before,selected_gradients,strict=True):
        norm = np.linalg.norm(gradient.astype(np.float64))
        gradient = gradient/max(1.,norm)
        first, second = .1*gradient, .001*gradient**2
        expected.append(value-.001*np.sqrt(.001)/.1*first/(np.sqrt(second)+1e-7))
    adam_error = max(float(np.max(abs(a-b))) for a,b in zip(after,expected,strict=True))
    assert adam_error < 2e-6
    result = {'passed':True,'scope':'Software/gradient oracle on8 existing real TRAIN recordings; no acoustic accuracy evidence',
        'model_id':model.model_id,'recordings':[row['id'] for row in rows],
        'full_contexts':sum(len(bag) for bag in bags),'selected_maxima':sum(len(c) for c in choices),
        'full_objective_including_l2':full_value,'independent_bce_objective':oracle_value,
        'two_pass_objective':selected_value,'independent_gradient_max_error':max_gradient_error,
        'two_pass_gradient_max_error':selection_gradient_error,'train_inference_logit_error':mode_error,
        'batch_single_logit_error':batch_error,'independent_first_adam_step_weight_error':adam_error,
        'clipping_semantics':'Keras clipnorm1 clips each parameter tensor, not one global concatenated gradient.',
        'gradient_norms':{v.path:float(np.linalg.norm(g)) for v,g in zip(net.trainable_variables,selected_gradients,strict=True)},
        'checkpoint_checks':checkpoint_checks}
    write_json('reports/training-audit-v1/gradients-and-checkpoints.json',result)
    print({k:v for k,v in result.items() if k not in {'gradient_norms','checkpoint_checks','recordings'}},flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
