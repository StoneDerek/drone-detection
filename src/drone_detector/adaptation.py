"""Source-held-out development comparisons of positive coverage and adaptation.

All records have been consumed in earlier research. These folds diagnose that
development corpus; they are not a new independent generalization benchmark.
"""

from collections import Counter, defaultdict

import numpy as np

from .common import digest

FIELD_SOURCES = {"wsu-air3", "audrok-dresden"}


def cohort_signature(rows):
    return digest(
        [
            {key: r[key] for key in ("id", "sha256", "label", "source", "group_id")}
            for r in sorted(rows, key=lambda r: r["id"])
        ]
    )


def group_weights(rows, total_mass=9959):
    sources = {
        label: {r["source"] for r in rows if r["label"] == label}
        for label in ("drone", "non_drone")
    }
    if any(not source for source in sources.values()):
        raise ValueError("Training needs both classes")
    groups = defaultdict(set)
    counts = Counter((r["label"], r["source"], r["group_id"]) for r in rows)
    for r in rows:
        groups[r["label"], r["source"]].add(r["group_id"])
    return np.array(
        [
            total_mass
            / (
                2
                * len(sources[r["label"]])
                * len(groups[r["label"], r["source"]])
                * counts[r["label"], r["source"], r["group_id"]]
            )
            for r in rows
        ],
        dtype=np.float64,
    )


def assert_partitions(rows, train_ids, selection_ids, validation_ids):
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate study IDs")
    identities = []
    for ids in (train_ids, selection_ids, validation_ids):
        if len(ids) != len(set(ids)) or set(ids) - by_id.keys():
            raise ValueError("Repeated or unknown partition IDs")
        identities.append(
            {
                (kind, by_id[i][kind])
                for i in ids
                for kind in ("id", "group_id", "sha256", "decoded_sha256")
                if by_id[i].get(kind)
            }
        )
    if any(identities[i] & identities[j] for i in range(3) for j in range(i)):
        raise ValueError("Recording/identity leakage across development partitions")


def make_folds(rows):
    if not FIELD_SOURCES | {"fsd50k-eval"} <= {r["source"] for r in rows}:
        raise ValueError("Both field sources and the full background challenge are required")
    base = {r["id"] for r in rows if r["split"] == "train"}
    selection = sorted(r["id"] for r in rows if r["split"] == "development")
    if not base or not selection:
        raise ValueError("Previous full TRAIN and DEVELOPMENT partitions are required")
    folds = {}
    for held_source in sorted(FIELD_SOURCES):
        held = {r["id"] for r in rows if r["source"] in {held_source, "fsd50k-eval"}}
        field_training = [r for r in rows if r["source"] in FIELD_SOURCES - {held_source}]
        negatives = {r["id"] for r in field_training if r["label"] == "non_drone"}
        positives = {r["id"] for r in field_training if r["label"] == "drone"}
        current, broader = sorted(base | negatives), sorted(base | negatives | positives)
        for train in (current, broader):
            assert_partitions(rows, train, selection, sorted(held))
        by_id = {r["id"]: r for r in rows}
        if not held or {by_id[i]["source"] for i in current + broader + selection} & {
            held_source,
            "fsd50k-eval",
        }:
            raise ValueError("Held source entered fitting or model selection")
        folds[held_source] = dict(
            current_positive_train=current,
            broader_positive_train=broader,
            selection=selection,
            validation=sorted(held),
            added_positive_ids=sorted(positives),
            matched_extra_negative_ids=sorted(negatives),
        )
    return folds


def batches_by_length(rows, batch_size, seed, epoch, shuffle=True):
    """Visit every example once, with no external waveform padding or truncation."""
    if batch_size < 1:
        raise ValueError("Positive batch size required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[round(row["duration_seconds"] * 16000)].append(i)
    batches = []
    for length, indices in sorted(groups.items()):
        if length < 2800:
            raise ValueError("Input shorter than native BEATs minimum")
        if shuffle:
            rng.shuffle(indices)
        batches.extend(indices[i : i + batch_size] for i in range(0, len(indices), batch_size))
    if shuffle:
        rng.shuffle(batches)
    return batches


def partition_summary(rows, ids):
    ids_set = set(ids)
    selected = [r for r in rows if r["id"] in ids_set]
    return dict(
        clips=len(selected),
        ids_sha256=digest(sorted(ids)),
        by_source_label={
            f"{source}/{label}": count
            for (source, label), count in sorted(
                Counter((r["source"], r["label"]) for r in selected).items()
            )
        },
        positive_groups=len({r["group_id"] for r in selected if r["label"] == "drone"}),
        negative_groups=len({r["group_id"] for r in selected if r["label"] == "non_drone"}),
    )
