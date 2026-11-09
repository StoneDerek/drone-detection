from dataclasses import asdict, dataclass
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    window_seconds: float = 3.0
    hop_seconds: float = 0.5
    n_fft: int = 512
    stft_hop: int = 160
    mel_bins: int = 48
    f_min: float = 80.0
    f_max: float = 7600.0
    representation: str = "absolute_mel"
    min_audio_seconds: float = 0.25
    silence_rms: float = 0.00001
    gain_target_rms: float = 0.05
    gain_max_db: float = 20.0

    def __post_init__(self):
        if self.representation not in {"absolute_mel", "relative_mel", "relative_harmonic", "yamnet", "yamnet_rms"}:
            raise ValueError("Unknown representation")
        if not (0 < self.hop_seconds <= self.window_seconds):
            raise ValueError("Require 0 < hop <= window")
        if self.hop_samples < 1:
            raise ValueError("Hop must contain at least one sample")
        if not (0 < self.min_audio_seconds <= self.window_seconds):
            raise ValueError("Invalid minimum context")
        if not (0 <= self.f_min < self.f_max <= self.sample_rate / 2):
            raise ValueError("Invalid frequency range")
        if self.window_samples < self.n_fft or not 0 < self.stft_hop <= self.n_fft:
            raise ValueError("Invalid FFT configuration")
        if not (0 < self.gain_target_rms <= .5 and 0 <= self.gain_max_db <= 40):
            raise ValueError('Invalid context-gain configuration')

    @property
    def window_samples(self):
        return round(self.window_seconds * self.sample_rate)

    @property
    def hop_samples(self):
        return round(self.hop_seconds * self.sample_rate)

    def to_dict(self):
        return asdict(self)


def decode(path, sample_rate=16000):
    """Decode float amplitudes, arithmetic channel mean, deterministic polyphase resampling.

    This is a file ingress adapter. The streaming engine itself only accepts the
    canonical rate; this offline resampler is not presented as live capture.
    """
    samples, original_rate = sf.read(Path(path), dtype="float32", always_2d=True)
    if len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError(f"Empty or nonfinite audio: {Path(path).name}")
    if original_rate <= 0:
        raise ValueError("Invalid sample rate")
    channels = samples.shape[1]
    raw_peak = float(np.max(np.abs(samples)))
    clipped = float(np.mean(np.abs(samples) >= 0.999))
    mono = samples.mean(axis=1, dtype=np.float64).astype(np.float32)
    if original_rate != sample_rate:
        divisor = gcd(original_rate, sample_rate)
        mono = signal.resample_poly(mono, sample_rate // divisor, original_rate // divisor)
    return np.asarray(mono, dtype=np.float32), {
        "original_sample_rate": original_rate,
        "channels": channels,
        "duration_seconds": len(samples) / original_rate,
        "peak": raw_peak,
        "clipped_fraction": clipped,
        "rms": float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))),
        "silence_fraction": float(np.mean(np.abs(mono) < 1e-5)),
        "canonical_sample_rate": sample_rate,
    }


@lru_cache(maxsize=16)
def mel_bank(config):
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)  # noqa: E731
    mel_to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)  # noqa: E731
    edges = mel_to_hz(np.linspace(hz_to_mel(config.f_min), hz_to_mel(config.f_max),
                                  config.mel_bins + 2))
    freq = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
    weights = np.maximum(0, np.minimum(
        (freq[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None],
        (edges[2:, None] - freq[None, :]) / (edges[2:] - edges[1:-1])[:, None],
    ))
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def normalize_context_gain(samples, config):
    """Fixed causal gain from available samples; never fit or read a whole file."""
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms < config.silence_rms:
        return samples
    peak = float(np.max(np.abs(samples)))
    gain = min(config.gain_target_rms / rms, 10 ** (config.gain_max_db / 20), .98 / max(peak, 1e-12))
    return (samples * gain).astype(np.float32)


def quality_measures(samples, sample_rate):
    """Advisory signal measurements, not wind-speed or SNR estimates."""
    wave = np.asarray(samples, dtype=np.float64)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all() or sample_rate <= 0:
        raise ValueError('Quality measurements require finite nonempty mono audio and a positive rate')
    flags = []
    clipped = float(np.mean(np.abs(wave) >= .999))
    if clipped > .001:
        flags.append('clipping')
    centered = wave - wave.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    total = float(spectrum.sum())
    frequencies = np.fft.rfftfreq(len(wave), 1 / sample_rate)
    low_fraction = float(spectrum[frequencies < 100].sum() / total) if total > 1e-20 else 0.0
    if low_fraction > .6:
        flags.append('low_frequency_dominance')
    return {'quality_flags': flags, 'low_frequency_fraction_below_100hz': low_fraction,
            'window_clipped_fraction': clipped}


def features(samples, config, encoder=None):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Features require finite mono audio")
    if len(samples) > config.window_samples:
        raise ValueError("Window exceeds configured context")
    if not len(samples):
        raise ValueError('Features require nonempty audio')
    if config.representation == 'yamnet_rms':
        samples = normalize_context_gain(samples, config)
    samples = np.pad(samples, (0, config.window_samples - len(samples)))
    if config.representation.startswith('yamnet'):
        if encoder is None:
            raise ValueError("Missing frozen audio encoder")
        return encoder.features(samples)
    _, _, stft = signal.stft(
        samples, fs=config.sample_rate, nperseg=config.n_fft,
        noverlap=config.n_fft - config.stft_hop, boundary=None, padded=False,
    )
    power = np.abs(stft).astype(np.float64) ** 2
    mel = mel_bank(config) @ power
    # Scale-relative floor preserves gain invariance, except exactly silent input.
    floor = max(float(mel.max()) * 1e-10, np.finfo(np.float64).tiny)
    logmel = 10 * np.log10(np.maximum(mel, floor))
    if config.representation != "absolute_mel":
        logmel = logmel - logmel.mean()
    vector = np.concatenate([logmel.mean(axis=1), logmel.std(axis=1)])
    if config.representation == "relative_harmonic":
        # Spectral flatness, crest and normalized cepstral periodicity. These
        # describe tones, not a unique drone signature (fans are confounders).
        spectrum = power.mean(axis=1) + max(float(power.max()) * 1e-10, 1e-30)
        norm = spectrum / spectrum.sum()
        flatness = np.exp(np.log(spectrum).mean()) / spectrum.mean()
        cepstrum = np.fft.irfft(np.log(spectrum))
        low, high = config.sample_rate // 500, config.sample_rate // 60
        periodicity = np.max(np.abs(cepstrum[low:high])) / (np.linalg.norm(cepstrum[1:]) + 1e-12)
        vector = np.r_[vector, flatness, norm.max(), periodicity]
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite features")
    return vector
