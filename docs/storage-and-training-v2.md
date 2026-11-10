# GPU notebook, durable storage and recovery

The selected host is **a free Colab GPU notebook with Google Drive**. Docker CPU
credits remain a possible fallback. No paid compute has been provisioned.

[Open the setup notebook in Colab](https://colab.research.google.com/github/StoneDerek/drone-detection/blob/main/notebooks/gpu_runtime_setup.ipynb)
(or inspect [its source](../notebooks/gpu_runtime_setup.ipynb)). It verifies the
runtime and checkpoint mechanism; it does **not** train a new drone detector or
claim that the corpus is ready. The [data-first research plan](research-plan-v2.md)
remains the acquisition and evaluation gate.

## First use

1. Select **Runtime → Change runtime type → GPU**. Free GPU availability and
   runtime lifetime vary; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html).
2. Edit `PROJECT_DRIVE_PATH` if desired. The default is
   `/content/drive/MyDrive/drone-detection-v2`; shared drives can use
   `/content/drive/Shareddrives/<drive>/<folder>`. Authenticate through Colab's
   Drive prompt. No credential values belong in chat or the repository.
3. Run all cells. The notebook checks for an allocated NVIDIA GPU, mounts Drive,
   verifies a small write/read, pins the current repository commit in Drive and
   creates a separate, locked PyTorch/CUDA environment on the runtime disk.
4. Require a passing recovery report, then run the final flush/unmount cell.
   Select **Runtime → Disconnect and delete runtime**, reconnect with a GPU,
   and run all cells again. Require another pass with
   `replayed_checkpoint_from_earlier_invocation: true`.

The first run saves `bootstrap.json` with a commit and environment-lock hash.
Replacement runtimes reuse that commit even if `main` advances. Updates require
a deliberately new project folder. Dirty local checkouts, incompatible identities,
missing GPUs and failed checks stop execution; there is no silent CPU fallback.
Use one active runtime per project folder. Keep original data and old run folders.

The pinned CUDA 12.6/PyTorch 2.7.1 environment must support the allocated GPU and
driver. Its actual device operation and AMP/backpropagation recovery probe are
checked in the notebook. A different GPU may change numerical results. The probe
requires absolute state agreement within `1e-4` on CUDA, and exact equality on CPU;
a failure must be investigated rather than relaxing the limit to hide it.

## Source intake after runtime verification

The [CPU-only source-intake notebook](https://colab.research.google.com/github/StoneDerek/drone-detection/blob/main/notebooks/source_intake.ipynb)
uses a separate pinned project subfolder and the standard library. Its default
profile acquires 6.890 GB, including the original AuDroK outdoor pair; it does not
train a model. Confirm available Drive allocation before running. Completed files
are hashed and reused on a replacement runtime; interrupted AuDroK archives must
restart because that server ignores HTTP ranges. See the
[acquisition audit and remaining data gaps](../reports/corpus-acquisition-v2/README.md).

## Native point-file preparation

The [completed intake audit](../reports/outdoor-source-audit-v1/README.md) found
58 point recordings among 232 simultaneous recorder-view files. Three of four
views are array averages, so they are excluded from the primary point cache.
The [CPU preparation notebook](https://colab.research.google.com/github/StoneDerek/drone-detection/blob/main/notebooks/prepare_point_recordings.ipynb)
creates 1.867 GB of unchanged native WAVs with full source/member checks and
restart receipts. It requires additional Drive allocation and about 4 GB of local
SSD for one verified working archive. It leaves labels/splits unaccepted and
does not create additional independent recordings.

## Storage arrangement

- GitHub holds code, locked environments, configurations and small manifests.
- Drive holds source archives, checksummed data packages, manifests, checkpoints
  and results. Confirm quota and source permissions before bulk transfer; the
  proposed initial allocation is 100–200GB, subject to the corpus inventory.
- The runtime stages active data on its local SSD. Read batches locally rather
  than thousands of small window files from a Drive mount. Verify each staged
  package against its manifest. This data-staging adapter is still to be built.

A bigger Drive quota does not enlarge the runtime disk. The setup's 1MiB
write/read check verifies mount access, not quota or independent cloud durability.
A Drive filesystem rename/read-back alone is insufficient to prove sync to the
service. Flush before planned disconnects and verify recovery after actual
runtime deletion. Unexpected termination can lose work since the last checkpoint,
including a checkpoint whose bytes had not reached durable storage.

## Implemented recovery mechanism

[`torch_recovery.py`](../src/drone_detector/torch_recovery.py) captures model
parameters and buffers, per-module training/evaluation modes, optimizer,
scheduler, AMP scaler, step, supplied sampler/cursor and selection state, and
Python/NumPy/Torch/CUDA random state. Snapshots are detached CPU copies.

Checkpoints are serialized locally, copied to immutable content-hashed names,
read back and verified before a complete manifest is published. Partial files
are ignored. Missing/corrupt bytes fall back to an older verified checkpoint.
Run-identity mismatches, stale writers and visible divergent branches are rejected.
This is a single-writer filesystem protocol, not a distributed lock service.
All recovery points are retained in this small probe; a full trainer still needs
a retention policy and a separately retained best-validation model.

[`notebook_recovery_probe.py`](../scripts/notebook_recovery_probe.py) exercises
batch-normalization buffers, dropout, AdamW, scheduling, AMP on CUDA, three RNG
families and a shuffled data cursor. It hard-exits after step 5, resumes step 4 in
a fresh process, and compares the complete step-12 state to an uninterrupted
control. Later invocations replay the original persisted step-4 checkpoint in a
new branch directory. The fixture is numerical, not acoustic evidence.

The probe uses an explicit sampler and `num_workers=0`. Saving global RNG alone
would not recover arbitrary prefetched loaders or external augmentation RNGs.
A full audio trainer must pin code, data-manifest/split/preprocessing/configuration
hashes and seed in its run identity, integrate all loader state, checkpoint after
completed optimizer updates (initially targeting every 2–5 minutes), retain model
selection state and undergo the same interruption check. Existing diagnostic
trainers export weights and do not acquire these guarantees automatically.

## Verification and remaining work

Local CPU fresh-process verification matched the uninterrupted state exactly,
including a second invocation replaying the persisted checkpoint. Automated tests
cover partial/corrupt checkpoints, incompatible identities, stale/divergent
writers, immutable snapshots, module modes, pinned revisions across upstream
changes, notebook schema and code compilation. The dedicated CI job uses the
separate CPU lock; the ordinary inference environment does not require PyTorch.

On 2026-11-09 the user reported the notebook output
`Replayed an earlier invocation: True`. That confirms the later-invocation path
ran against an earlier saved checkpoint. The flag also becomes true on an ordinary
rerun; it does not by itself attest that the runtime was deleted. The Codex host
has no exposed NVIDIA GPU or authenticated Drive mount; the full Colab report,
GPU identity and runtime-replacement details have not been independently inspected.
Full audio training remains unverified.

To repeat the software checks on a Linux CPU host with Python 3.12 and uv 0.13.5:

```bash
uv sync --project notebooks/runtime --frozen --extra cpu --extra test
PYTHONPATH=src notebooks/runtime/.venv/bin/python -m pytest \
  --confcutdir=tests/notebook tests/notebook -q
```

Next: verify Drive quota and usable GPU in Colab; complete the source/access,
label, parent-recording and diversity inventory; acquire and split suitable real
data; integrate the selected public pretrained CNN and its matched scratch and
non-neural controls; measure actual audio memory/throughput; then begin the bounded
study. Notebook availability does not resolve the data-quality problem.
