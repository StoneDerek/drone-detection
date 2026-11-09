"""Train-only standardization and a portable linear classifier, without pickle."""

import time
import shutil
from pathlib import Path

import numpy as np
from scipy.special import expit

from .audio import FeatureConfig, features
from .common import digest, file_hash, overlap, read_json, revision, versions, write_json
from .data import VERIFIED, checked_audio, load_manifest


class Model:
    def __init__(self, metadata, arrays, encoder=None):
        self.metadata = metadata
        self.arrays = arrays
        self.config = FeatureConfig(**metadata["features"])
        self.encoder = encoder

    def score(self, samples):
        vector = features(samples, self.config, self.encoder)
        if vector.shape != self.arrays["mean"].shape:
            raise ValueError("Feature dimension disagrees with artifact")
        standardized = (vector - self.arrays["mean"]) / self.arrays["scale"]
        return float(expit(standardized @ self.arrays["coef"] + self.arrays["intercept"][0]))

    @property
    def model_id(self):
        return self.metadata["model_id"]


def save_model(output, metadata, arrays, encoder_source=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a model package")
    output.mkdir(parents=True)
    np.savez_compressed(output / "parameters.npz", **arrays)
    metadata = {k: v for k, v in metadata.items() if k != "model_id"}
    if encoder_source is not None:
        encoder_source = Path(encoder_source)
        shutil.copytree(encoder_source, output / "encoder")
        files = {str(p.relative_to(encoder_source)): file_hash(p)
                 for p in sorted(encoder_source.rglob("*")) if p.is_file()}
        metadata["encoder"] = {**metadata.get("encoder", {}), "directory": "encoder", "files_sha256": files}
    metadata["parameters_sha256"] = file_hash(output / "parameters.npz")
    metadata["model_id"] = digest(metadata)
    write_json(output / "model.json", metadata)
    return load_model(output)


def load_model(path, require_frozen=False):
    path = Path(path)
    meta = read_json(path / "model.json")
    if meta.get('schema_version') == 2:
        from .spectrogram import load_spectral_model
        return load_spectral_model(path, meta, require_frozen)
    if meta.get("schema_version") != 1 or meta.get("class_mapping") != {"0": "non_drone", "1": "drone"}:
        raise ValueError("Unknown model schema/class mapping")
    identity = digest({k: v for k, v in meta.items() if k != "model_id"})
    if meta.get("model_id") != identity:
        raise ValueError("Model metadata identity mismatch")
    if file_hash(path / "parameters.npz") != meta["parameters_sha256"]:
        raise ValueError("Model parameters hash mismatch; do not substitute statistics")
    if require_frozen and meta.get("status") != "frozen":
        raise ValueError("Tune and freeze decision logic on development audio first")
    with np.load(path / "parameters.npz", allow_pickle=False) as loaded:
        if set(loaded.files) != {"mean", "scale", "coef", "intercept"}:
            raise ValueError("Missing or unexpected parameters")
        arrays = {key: loaded[key].copy() for key in loaded.files}
    size = meta["feature_dimension"]
    if any(arrays[k].shape != (size,) for k in ["mean", "scale", "coef"]):
        raise ValueError("Parameter shape mismatch")
    if arrays["intercept"].shape != (1,) or np.any(arrays["scale"] <= 0):
        raise ValueError("Invalid normalization/intercept")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite model artifact")
    if require_frozen:
        decision = meta.get("decision", {})
        if not (0 < decision.get("off_threshold", -1) <= decision.get("on_threshold", -1) <= 1):
            raise ValueError("Invalid decision thresholds")
        if any(not isinstance(decision.get(k), int) or decision[k] < 1
               for k in ["on_hits", "off_hits"]):
            raise ValueError("Invalid alarm confirmation")
    encoder = None
    if meta["features"]["representation"].startswith('yamnet'):
        from .encoder import YAMNet
        spec = meta.get("encoder", {})
        if spec.get("directory") != "encoder" or not spec.get("files_sha256"):
            raise ValueError("Missing portable encoder package")
        for name, expected in spec["files_sha256"].items():
            file = (path / "encoder" / name).resolve()
            if not file.is_relative_to((path / "encoder").resolve()) or file_hash(file) != expected:
                raise ValueError("Encoder artifact hash mismatch")
        encoder = YAMNet(path / "encoder")
    return Model(meta, arrays, encoder)


def train(manifest, output, representation="absolute_mel", seed=42, gain_augment=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    started = time.perf_counter()
    rows = load_manifest(manifest, require_split=True)
    records = [r for r in rows if r["split"] == "train" and r["annotation_status"] in VERIFIED]
    config = FeatureConfig(representation=representation)
    rng = np.random.default_rng(seed)
    vectors, labels, weights, inventory = [], [], [], []
    for row in records:
        audio = checked_audio(row, manifest, config.sample_rate)
        record_vectors, record_labels = [], []
        for start in range(0, len(audio), config.hop_samples):
            chunk = audio[start:start + config.window_samples]
            if len(chunk) < config.window_samples:
                break  # No padded synthetic tails in supervised training.
            a, b = start / config.sample_rate, (start + len(chunk)) / config.sample_rate
            if overlap(a, b, row["ambiguous"]) > 0:
                continue
            event_overlap = overlap(a, b, row["events"])
            if event_overlap >= b - a - 1e-8:
                label = 1
            elif event_overlap == 0:
                label = 0
            else:
                continue
            record_vectors.append(features(chunk, config))
            record_labels.append(label)
            if gain_augment:
                # A train-only, label-preserving scalar gain; no clipping.
                gain = min(10 ** (rng.uniform(-18, 12) / 20),
                           0.98 / max(float(np.max(np.abs(chunk))), 1e-12))
                record_vectors.append(features(chunk * gain, config))
                record_labels.append(label)
        if not record_vectors:
            continue
        vectors.extend(record_vectors)
        labels.extend(record_labels)
        weights.extend([1 / len(record_vectors)] * len(record_vectors))
        inventory.append({"id": row["id"], "group_id": row["group_id"], "sha256": row["sha256"],
                          "source": row["source"], "windows": len(record_vectors)})
    if not vectors or set(labels) != {0, 1}:
        raise ValueError("Need both classes of fully annotated training windows")
    x, y, weight = np.asarray(vectors), np.asarray(labels), np.asarray(weights)
    # Each recording contributes equal initial mass; balance class mass after
    # windowing, rather than silently letting long files dominate.
    for label in [0, 1]:
        weight[y == label] *= len(y) / (2 * weight[y == label].sum())
    scaler = StandardScaler().fit(x, sample_weight=weight)
    with threadpool_limits(limits=1):
        classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(
            scaler.transform(x), y, sample_weight=weight
        )
    if int(classifier.n_iter_[0]) >= 1000:
        raise ValueError("Classifier did not converge")
    metadata = {
        "schema_version": 1, "status": "trained", "features": config.to_dict(),
        "class_mapping": {"0": "non_drone", "1": "drone"}, "feature_dimension": x.shape[1],
        "training": {"manifest_sha256": file_hash(manifest), "seed": seed,
                     "gain_augment": gain_augment, "records": inventory,
                     "window_counts": {str(k): int((y == k).sum()) for k in [0, 1]},
                     "seconds": time.perf_counter() - started,
                     "sample_weight": "equal recording mass then balanced class mass",
                     "classifier": "logistic regression C=1, lbfgs, max_iter=1000",
                     "origins": sorted({r["origin"] for r in records})},
        "normalization": "StandardScaler fit to weighted training windows only; parameters in NPZ",
        "versions": versions(), "code": revision(),
        "limitations": "Experimental. No reliability claim without independent authentic audio evaluation.",
    }
    arrays = {"mean": scaler.mean_, "scale": scaler.scale_,
              "coef": classifier.coef_[0], "intercept": classifier.intercept_}
    return save_model(output, metadata, arrays)
