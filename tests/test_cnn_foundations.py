import numpy as np
import pytest

from drone_detector.foundation_training import (BalancedRecordingBatches, full_bag_max_indices,
                                              pool_positive_logits, recording_losses, train_foundation_cnn)
from drone_detector.spectrogram import build_cnn, cnn_spec, cnn_weight_shapes, tensorflow


def input_coverage(frequency_rows, variant):
    tf = tensorflow()
    inputs = tf.zeros([1, frequency_rows, 172, 1])
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        values = inputs
        # Positive unit kernels expose every possible path regardless of learned
        # weights, ReLU inactivity or cancellation. This is a software control.
        for layer in cnn_spec(variant):
            kernel = tf.ones([*layer['kernel'], int(values.shape[-1]), 1])
            values = tf.nn.conv2d(values, kernel, strides=[1, *layer['stride'], 1], padding='SAME')
        total = tf.reduce_sum(values)
    gradient = tape.gradient(total, inputs).numpy()[0, :, :, 0]
    return np.any(gradient > 0, axis=1)


@pytest.mark.parametrize('rows', [64, 256, 1933])
def test_repaired_strides_cover_every_frequency_row_without_extra_parameters(rows):
    assert not input_coverage(rows, 'compact').all()
    assert input_coverage(rows, 'coverage').all()
    assert input_coverage(rows, 'coverage_medium').all()
    original, repaired = build_cnn(42), build_cnn(42, 'coverage')
    assert original.count_params() == repaired.count_params() == 1781
    for a, b in zip(original.get_weights(), repaired.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)
    medium = build_cnn(42, 'coverage_medium')
    assert medium.count_params() == 93313
    assert [tuple(w.shape) for w in medium.get_weights()] == cnn_weight_shapes('coverage_medium')


def test_smoothing_distributes_positive_gradient_and_preserves_equal_logits():
    tf = tensorflow()
    values = tf.Variable([-2., 0., 1., .4, .4])
    ids = tf.constant([0, 0, 0, 1, 1])
    with tf.GradientTape() as tape:
        pooled, weights = pool_positive_logits(values, ids, 2, .5)
        loss = tf.reduce_sum(tf.nn.softplus(-pooled))
    gradient = tape.gradient(loss, values).numpy()
    assert np.all(gradient < 0)
    assert float(pooled[1]) == pytest.approx(.4)
    assert float(tf.reduce_sum(weights[:3])) == pytest.approx(1.)
    assert float(tf.reduce_sum(weights[3:])) == pytest.approx(1.)
    with tf.GradientTape() as tape:
        maximum, _ = pool_positive_logits(values, ids, 2)
        hard_loss = tf.reduce_sum(tf.nn.softplus(-maximum))
    hard_gradient = tape.gradient(hard_loss, values).numpy()
    assert np.count_nonzero(hard_gradient[:3]) == 1
    assert pooled[0] < maximum[0]


def test_invalid_architecture_is_not_silently_loaded():
    with pytest.raises(ValueError):
        cnn_weight_shapes('unknown')


def test_full_recording_selection_matches_full_loss_and_gradients_including_ties():
    tf = tensorflow()
    # Two tied maxima with distinct feature gradients, including one beyond
    # the first sample8. This is a software oracle, not acoustic data.
    net = tf.keras.Sequential([tf.keras.Input((1, 2, 1)), tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, kernel_initializer='ones', bias_initializer='zeros')])
    positive = np.zeros((12, 1, 2), dtype=np.float32)
    positive[3, 0] = [2., 0.]
    positive[9, 0] = [0., 2.]
    negative = np.zeros((5, 1, 2), dtype=np.float32)
    negative[4, 0] = [1., 0.]
    forward = lambda x: net(x, training=False)  # noqa: E731
    choices = [full_bag_max_indices(bag, forward, batch_size=4) for bag in [positive, negative]]
    np.testing.assert_array_equal(choices[0], [3, 9])
    np.testing.assert_array_equal(choices[1], [4])

    def objective_and_gradients(bags):
        images = np.concatenate(bags)[..., None]
        ids = np.repeat(np.arange(2), [len(bag) for bag in bags]).astype(np.int32)
        with tf.GradientTape() as tape:
            logits = tf.reshape(net(images, training=True), [-1])
            losses, _ = recording_losses(logits, ids, tf.constant([1., 0.]), consistent=True)
            loss = tf.reduce_mean(losses)
        return float(loss), [g.numpy() for g in tape.gradient(loss, net.trainable_variables)]

    full_loss, full_gradients = objective_and_gradients([positive, negative])
    selected_loss, selected_gradients = objective_and_gradients([bag[indices] for bag, indices in zip([positive, negative], choices, strict=True)])
    assert selected_loss == pytest.approx(full_loss, abs=1e-7)
    for actual, expected in zip(selected_gradients, full_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_exact_selection_rejects_stochastic_or_different_pooling_modes():
    for kwargs in [{'consistent_loss': True, 'dropout_rate': .1},
                   {'consistent_loss': False, 'dropout_rate': 0.},
                   {'consistent_loss': True, 'dropout_rate': 0., 'smooth_positive': True}]:
        with pytest.raises(ValueError, match='Exact full-recording'):
            train_foundation_cnn(None, None, None, None, None, full_recording_bags=True, **kwargs)


def test_same_recording_score_cannot_reward_class_independent_noise():
    tf = tensorflow()
    # Exactly enumerate the two possible maxima of eight independent dropout
    # draws. Both labels receive identical bag distributions and no signal.
    probability_all_kept = .9**8
    expected_old = expected_repaired = 0.
    for probability, values in [(probability_all_kept, [-1.]*8),
                                (1-probability_all_kept, [-1.]*7+[1.5])]:
        logits = tf.constant(values*2)
        ids = tf.constant([0]*8+[1]*8)
        labels = tf.constant([0., 1.])
        repaired, _ = recording_losses(logits, ids, labels, consistent=True)
        expected_repaired += probability*float(tf.reduce_mean(repaired))
    # The old negative expectation is its per-window mean, independent of n;
    # positive maximum gets eight opportunities to exploit the random spike.
    negative = .9*np.logaddexp(0, -1.)+.1*np.logaddexp(0, 1.5)
    positive = probability_all_kept*np.logaddexp(0, 1.)+(1-probability_all_kept)*np.logaddexp(0, -1.5)
    expected_old = (negative+positive)/2
    assert expected_old == pytest.approx(.5660522037)
    assert expected_old < np.log(2)
    assert expected_repaired >= np.log(2)


def test_disabling_dropout_preserves_initial_weights_and_training_inference_agree():
    tf = tensorflow()
    original = build_cnn(42, 'frequency_aware')
    repaired = build_cnn(42, 'frequency_aware', 0.)
    for a, b in zip(original.get_weights(), repaired.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)
    images = tf.ones([2, 256, 172, 1])
    np.testing.assert_array_equal(repaired(images, training=True), repaired(images, training=False))


def test_balanced_sampler_has_four_per_class_and_visits_each_member_before_repeating():
    labels = np.asarray([0] * 20 + [1] * 8)
    sampler = BalancedRecordingBatches(labels, 42)
    batches = [sampler.take() for _ in range(5)]
    assert all(len(batch) == 8 and labels[batch].sum() == 4 for batch in batches)
    negatives = np.concatenate([batch[labels[batch] == 0] for batch in batches])
    assert len(np.unique(negatives)) == 20
    positives = np.concatenate([batch[labels[batch] == 1] for batch in batches[:2]])
    assert len(np.unique(positives)) == 8


def test_frequency_preserving_readout_retains_distinctions_at_matched_parameter_count():
    tf = tensorflow()
    global_net = build_cnn(42, 'global_readout')
    frequency_net = build_cnn(42, 'frequency_aware')
    assert global_net.count_params() == 154675
    assert frequency_net.count_params() == 154689
    for a, b in zip(global_net.get_weights()[:6], frequency_net.get_weights()[:6], strict=True):
        np.testing.assert_array_equal(a, b)
    images = np.zeros((2, 256, 172, 1), dtype=np.float32)
    images[0, [24, 60, 96, 132], :, 0] = .8
    images[1, [24, 60, 132, 204], :, 0] = .8
    global_features = tf.keras.Model(global_net.inputs, global_net.get_layer('pool').output)(images).numpy()
    frequency_features = tf.keras.Model(frequency_net.inputs, frequency_net.get_layer('pool').output)(images).numpy()
    np.testing.assert_allclose(global_features[0], global_features[1], atol=1e-7)
    assert np.max(abs(frequency_features[0]-frequency_features[1])) > 1e-5
