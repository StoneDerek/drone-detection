"""Frozen official BEATs representation with explicit, verified local assets.

Run scripts/fetch_beats.py once. Inference never downloads code or weights.
The checkpoint is a pinned third-party mirror of the published pretrained
BEATs_iter3_plus_AS2M checkpoint; equivalence to the OneDrive bytes is unverified.
"""

import importlib.metadata
import sys
import threading
import types
from pathlib import Path

import numpy as np

from .common import digest, file_hash, read_json

CODE_COMMIT = "732d834db70ee0fc3886b4bcbcfb4ce7fb829be2"
MODEL_NAME = "BEATs_iter3_plus_AS2M"
MIRROR_REVISION = "2ec75307f57dc8892c553df85fa90cacb2d62af3"
CHECKPOINT_SHA256 = "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"
CHECKPOINT_BYTES = 361499833
CHECKPOINT_URL = (
    "https://huggingface.co/SongsongBaobao8694/beats-iter3/resolve/"
    f"{MIRROR_REVISION}/BEATs_iter3_plus_AS2M.pt"
)
OFFICIAL_CHECKPOINT_LINK = "https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf"
SOURCE_ASSETS = {
    "BEATs.py": (6798, "27f289db7c56ce26f2ceb50d3719854b91b2dec1c2830d8b1dd8de1bbee19eeb"),
    "backbone.py": (30743, "31c0378379a7e0f1d1069f9da444fb86890fe1ea078959a2dcd39640cdcadbaa"),
    "modules.py": (7387, "edeb6b6cd6a784da749f932c3e0783c0bce556fc768d0d23a4d53d4b819eb424"),
    "README.md": (5501, "907a4a7f5f93ec6fb7ed32514ac55ad88659fd0ba93c37dc6af69815612e6162"),
    "LICENSE": (1104, "904dc4d8749877f1dba1cda48200d2462dccbeb7c134d5e4ef6fa75e0198c8fe"),
}
_IMPORT_LOCK = threading.Lock()


def source_url(name):
    upstream = name if name == "LICENSE" else f"beats/{name}"
    return f"https://raw.githubusercontent.com/microsoft/unilm/{CODE_COMMIT}/{upstream}"


def verify_assets(directory):
    """Validate source code and checkpoint before importing/executing either."""
    directory = Path(directory)
    manifest = read_json(directory / "assets.json")
    if manifest.get("schema_version") != 1 or manifest.get("model") != MODEL_NAME:
        raise ValueError("Unsupported BEATs asset manifest")
    expected = {name: checksum for name, (_, checksum) in SOURCE_ASSETS.items()}
    expected["model.pt"] = CHECKPOINT_SHA256
    if manifest.get("sha256") != expected:
        raise ValueError("BEATs asset manifest differs from pinned identities")
    for name, checksum in expected.items():
        path = directory / name
        size = CHECKPOINT_BYTES if name == "model.pt" else SOURCE_ASSETS[name][0]
        if not path.is_file() or path.stat().st_size != size or file_hash(path) != checksum:
            raise ValueError(f"BEATs asset missing or changed: {name}")
    return manifest


def _official_module(directory):
    """Load pinned source under a private package, adapting only bare imports."""
    package_name = "_drone_beats_" + digest(SOURCE_ASSETS)[:16]
    with _IMPORT_LOCK:
        if package_name + ".BEATs" in sys.modules:
            return sys.modules[package_name + ".BEATs"]
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]
        sys.modules[package_name] = package
        try:
            for name in ["modules", "backbone", "BEATs"]:
                path = directory / f"{name}.py"
                source = path.read_text()
                source = source.replace("from modules import (", "from .modules import (")
                source = source.replace("from backbone import (", "from .backbone import (")
                module = types.ModuleType(package_name + "." + name)
                module.__file__ = str(path)
                module.__package__ = package_name
                sys.modules[module.__name__] = module
                exec(compile(source, str(path), "exec"), module.__dict__)
        except Exception:
            for key in list(sys.modules):
                if key == package_name or key.startswith(package_name + "."):
                    del sys.modules[key]
            raise
        return sys.modules[package_name + ".BEATs"]


class BEATsEncoder:
    """Mean-pooled 768-D BEATs tokens for mono 16 kHz float waveforms.

    Inputs shorter than 2800 samples are rejected. Any crop/padding, channel
    selection or resampling is the caller's explicit policy. BEATs applies its
    released Kaldi filterbank frontend and normalization; no per-clip gain rule
    is introduced. CUDA must be requested explicitly and never falls back to CPU.
    """

    sample_rate = 16000
    feature_dim = 768
    minimum_samples = 2800

    def __init__(self, directory, device="cpu"):
        self.directory = Path(directory).resolve()
        self.asset_manifest = verify_assets(self.directory)
        self.device = str(device)
        if self.device != "cpu" and self.device != "cuda" and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("BEATs device must be cpu, cuda or cuda:<index>")
        self._model = None
        packages = {}
        for package in ["torch", "torchaudio", "numpy"]:
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = "not-installed"
        self.identity = {
            "family": "beats-frozen",
            "adapter_sha256": file_hash(Path(__file__)),
            "model": MODEL_NAME,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_source": CHECKPOINT_URL,
            "code_commit": CODE_COMMIT,
            "code_sha256": {name: value[1] for name, value in SOURCE_ASSETS.items()},
            "import_adaptation": "private-package-relative-imports-only-v1",
            "frontend": {
                "sample_rate": 16000, "waveform_scale": 32768,
                "kaldi_fbank": {"num_mel_bins": 128, "frame_length_ms": 25,
                                "frame_shift_ms": 10},
                "fbank_mean": 15.41663, "fbank_std_multiplier": 2,
                "fbank_std": 6.55582, "other_options": "torchaudio Kaldi defaults",
                "minimum_samples": self.minimum_samples, "short_input": "reject",
            },
            "pooling": "mean-all-native-time-frequency-tokens",
            "feature_dim": self.feature_dim,
            "precision": "float32",
            "device": self.device,
            "packages": packages,
            "checkpoint_provenance_limit": (
                "Pinned mirror content; not independently byte-compared with official OneDrive"
            ),
        }

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            import torchaudio  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "BEATs requires the separate torch/torchaudio runtime; see frozen-audio workflow"
            ) from error
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for BEATs but unavailable")
        verify_assets(self.directory)
        module = _official_module(self.directory)
        checkpoint = torch.load(
            self.directory / "model.pt", map_location="cpu", weights_only=True,
        )
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("cfg"), dict):
            raise ValueError("BEATs checkpoint must contain a plain configuration dictionary")
        config = module.BEATsConfig(checkpoint["cfg"])
        if config.finetuned_model or config.encoder_embed_dim != self.feature_dim:
            raise ValueError("Expected pretrained BEATs representations, not classifier probabilities")
        if config.input_patch_size != 16:
            raise ValueError("BEATs patch size differs from the declared minimum-input policy")
        # Initialization consumes random values even though every weight is replaced.
        # Preserve the caller's random state used for subsequent probe training.
        with torch.random.fork_rng(devices=[]):
            model = module.BEATs(config)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().requires_grad_(False)
        self._model = model.to(self.device)
        return self._model

    def features(self, audio):
        audio = np.asarray(audio)
        if audio.ndim != 1 or len(audio) < self.minimum_samples:
            raise ValueError("BEATs requires mono 16 kHz audio with at least 2800 samples")
        if not np.isfinite(audio).all():
            raise ValueError("BEATs audio contains non-finite values")
        model = self._load()
        import torch
        waveform = torch.from_numpy(np.array(audio, dtype=np.float32, copy=True))[None]
        with torch.inference_mode():
            tokens, padding = model.extract_features(waveform.to(self.device))
            if padding is not None and padding.any():
                raise ValueError("Unexpected masked BEATs tokens for an unmasked input")
            if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[2] != self.feature_dim:
                raise ValueError("Unexpected BEATs representation shape")
            vector = tokens.mean(dim=1)[0].cpu().numpy().astype(np.float64)
        if vector.shape != (self.feature_dim,) or not np.isfinite(vector).all():
            raise ValueError("Invalid BEATs embedding output")
        return vector
