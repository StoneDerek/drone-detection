"""Causal rolling context and a stateful alarm, shared by file and replay paths."""

import time

import numpy as np

from .audio import decode, quality_measures


class Alarm:
    def __init__(self, decision):
        self.decision = decision
        self.active = False
        self.hits = self.misses = 0

    def step(self, score):
        if score is None:
            return self.active
        if not self.active:
            self.hits = self.hits + 1 if score >= self.decision["on_threshold"] else 0
            if self.hits >= self.decision["on_hits"]:
                self.active, self.hits = True, 0
        else:
            self.misses = self.misses + 1 if score < self.decision["off_threshold"] else 0
            if self.misses >= self.decision["off_hits"]:
                self.active, self.misses = False, 0
        return self.active


class StreamDetector:
    """Accept canonical-rate float mono blocks; do not normalize whole recordings.

    Audio timestamps are deterministic. Delivery timestamps separately include
    input-block waiting and compute time. End-of-file permits one zero-padded
    partial window, disclosed in each trace row. Flush is single-use.
    """

    def __init__(self, model, decision=None, sample_rate=None):
        self.model, self.config = model, model.config
        if sample_rate is not None and sample_rate != self.config.sample_rate:
            raise ValueError("Resample at file ingress; live-rate conversion is not implemented")
        self.decision = decision or model.metadata.get("decision")
        if self.decision is None:
            raise ValueError("Missing alarm decision")
        self.alarm = Alarm(self.decision)
        self.buffer = np.empty(0, dtype=np.float32)
        self.next_start = 0
        self.received = 0
        self.finished = False

    def _score(self, samples, final=False):
        begin = time.perf_counter()
        duration = len(samples) / self.config.sample_rate
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        if duration < self.config.min_audio_seconds:
            score, reason = None, "insufficient_audio"
        elif rms < self.config.silence_rms:
            score, reason = 0.0, "silence"
        else:
            score, reason = self.model.score(samples), "scored"
        active = self.alarm.step(score)
        quality = quality_measures(samples, self.config.sample_rate)
        compute = time.perf_counter() - begin
        timestamp = (self.next_start + len(samples)) / self.config.sample_rate
        return {
            "window_start_seconds": self.next_start / self.config.sample_rate,
            "timestamp_seconds": timestamp,
            "delivered_at_seconds": self.received / self.config.sample_rate + compute,
            "score": score, "alarm": active, "reason": reason,
            "rms": rms, "padded": len(samples) < self.config.window_samples,
            "end_of_file": final, "compute_seconds": compute,
            "model_id": self.model.model_id,
            **quality,
        }

    def push(self, samples):
        if self.finished:
            raise ValueError("Cannot push after flush")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Expected finite canonical-rate mono samples")
        if not len(samples):
            return []
        self.received += len(samples)
        self.buffer = np.concatenate([self.buffer, samples])
        rows = []
        while len(self.buffer) >= self.config.window_samples:
            rows.append(self._score(self.buffer[:self.config.window_samples]))
            self.buffer = self.buffer[self.config.hop_samples:]
            self.next_start += self.config.hop_samples
        return rows

    def flush(self):
        if self.finished:
            raise ValueError("Already flushed")
        self.finished = True
        # There is usually overlap left after the last full window. Emit one
        # additional tail only if it contains NEW audio beyond the last window.
        last_full_end = self.next_start - self.config.hop_samples + self.config.window_samples
        has_new_audio = self.next_start == 0 or self.received > last_full_end
        if len(self.buffer) and has_new_audio:
            return [self._score(self.buffer, final=True)]
        return []


def replay_audio(model, samples, decision=None, block_samples=1600):
    if block_samples <= 0:
        raise ValueError("Block size must be positive")
    stream = StreamDetector(model, decision)
    rows = []
    for start in range(0, len(samples), block_samples):
        rows.extend(stream.push(samples[start:start + block_samples]))
    rows.extend(stream.flush())
    return rows


def infer_file(model, path, decision=None):
    begin = time.perf_counter()
    samples, info = decode(path, model.config.sample_rate)
    rows = replay_audio(model, samples, decision)
    elapsed = time.perf_counter() - begin
    return rows, {**info, "end_to_end_seconds": elapsed,
                  "real_time_factor": elapsed / info["duration_seconds"],
                  "mode": "recorded_stream_replay; file decoding/resampling precedes replay",
                  "model_id": model.model_id}


def alarm_intervals(rows, duration, merge_gap=1.0):
    intervals, start = [], None
    for row in rows:
        if row["alarm"] and start is None:
            start = row["timestamp_seconds"]
        elif not row["alarm"] and start is not None:
            intervals.append([start, row["timestamp_seconds"]])
            start = None
    if start is not None:
        intervals.append([start, duration])
    merged = []
    for start, end in intervals:
        # Zero-length terminal triggers are still alarms; they can match an
        # event within the collar, and count against negative exposure.
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged
