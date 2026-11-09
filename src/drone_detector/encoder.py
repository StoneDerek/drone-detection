"""Official frozen YAMNet SavedModel, loaded lazily with no network at inference."""

from pathlib import Path

import numpy as np

from .common import file_hash

_CACHE = {}


class YAMNet:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.key = file_hash(self.directory / "saved_model.pb")

    def features(self, audio):
        key = self.key
        if key not in _CACHE:
            try:
                import tensorflow as tf
            except ImportError as error:
                raise RuntimeError("Install the locked pretrained extra: uv sync --extra pretrained") from error
            try:
                tf.config.threading.set_intra_op_parallelism_threads(2)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except RuntimeError:
                pass  # TensorFlow may already be initialized by the host process.
            _CACHE[key] = (tf, tf.saved_model.load(str(self.directory)))
        tf, encoder = _CACHE[key]
        outputs = encoder(tf.convert_to_tensor(audio, dtype=tf.float32))
        embeddings = outputs[1] if isinstance(outputs, (tuple, list)) else outputs["embeddings"]
        vector = embeddings.numpy().mean(axis=0).astype(np.float64)
        if vector.shape != (1024,) or not np.isfinite(vector).all():
            raise ValueError("Invalid YAMNet embedding output")
        return vector
