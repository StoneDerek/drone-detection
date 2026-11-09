# ITU ARIS Lab. Acoustic Drone Dataset

A two-class acoustic dataset for drone detection, recorded outdoors with
multiple drone models flying at ranges from a few metres out to roughly two
kilometres, annotated by hand from a single continuous session.

| | |
|---|---|
| Classes | `drone` (4 491 segments), `background` (1 000 segments) |
| Segment length | 1.0 s |
| Sampling rate | 22 050 Hz, mono, 16-bit PCM |
| Source recording | one continuous outdoor session, 3.12 h, 44.1 kHz, mono |
| Total size | 244 MB |
| Licence | CC-BY-4.0 |

---

## Read this before you split the data

**Segments overlap by 50 %.** They were cut from the annotated intervals with a
half-second hop, so two consecutive files share half of their samples.

A random train/test split therefore leaks: `drone_0004.wav` shares 0.5 s of
audio with both `drone_0003.wav` and `drone_0005.wav`, and if one lands in
training and another in test, the model is evaluated on audio it has already
seen. In the work this set was built for, that inflated the reported figure
substantially before we caught it.

`groups.json` exists so you do not have to repeat the mistake. It maps every
segment to the source interval it was cut from — 35 annotated drone intervals,
and contiguous blocks for background. **Split at group boundaries, never at file
level.** Two ready-made protocols are in `splits/`.

---

## Layout

```
audio/
  drone/        drone_0001.wav … drone_4491.wav
  background/   background_0001.wav … background_1000.wav
groups.json     every segment -> its source interval, plus interval times
splits/
  split_interval_disjoint.json   whole intervals dealt to train/val/test
  split_temporal.json            session cut in time: first 65 % / 15 % / last 20 %
metadata.json   counts, recording parameters, annotation summary
build_acolab_groups.py  how groups.json was reconstructed from the export log
SHA256SUMS      checksum for every file
```

Both split files map a relative audio path to `train`, `val` or `test`.

---

## Collection

Recorded outdoors in a single continuous session of 3 h 7 min. Several drone
models were flown through a range of manoeuvres, at distances from a few metres
to about two kilometres, so the set spans near-field passes and far-field
detections under the same conditions. Weather, background traffic and wind were
whatever the day provided; nothing was staged.

The recording is single-channel. This set supports detection and classification,
not direction finding.

Operator speech is audible in some segments. The people recorded are members of
the research group and consented to publication.

## Annotation

Drone activity was annotated as time intervals directly on the continuous
recording, and segments were cut from those intervals. Background segments come
from unlabelled stretches of the same recording, spaced 0.5 s apart, capped at
1 000.

The session was annotated **independently by two people**. Comparing them at
100 ms resolution:

| | |
|---|---|
| Intervals marked | 35 vs 34 |
| Cohen's κ | **0.587** |
| Temporal IoU | 0.476 |

That is moderate agreement, and it is reported rather than hidden: deciding
whether a distant drone is audible in a windy outdoor recording is genuinely
ambiguous, and the label noise this implies puts a ceiling on what any model
trained here can reach. The released labels follow the first annotator.

---

## Baselines

Four architectures on 128-band log-Mel spectrograms, mean ± sd over 7 seeds,
using `split_interval_disjoint.json`:

| Training data | Balanced accuracy on this set |
|---|---|
| Public drone datasets only | 61.4 ± 5.8 % |
| Public datasets + this set | **99.2 ± 0.8 %** |

The gap is why the set exists: models trained on the public acoustic drone
corpora do not transfer to these recording conditions.

A sweep over how much of this set is used in training shows the benefit
saturates early — about four of the 35 intervals recover most of it:

| Share used in training | Balanced accuracy |
|---|---|
| 0 % | 60.9 ± 4.8 % |
| 12 % (4 intervals) | 93.2 ± 6.3 % |
| 25 % (8 intervals) | 98.3 ± 1.6 % |
| 100 % (35 intervals) | 99.3 ± 0.6 % |

## Known limitations

- **One session.** Everything comes from a single day at a single site, so this
  set cannot measure cross-session or cross-site generalisation. A model scoring
  well here has learned these conditions, not drone acoustics in general.
- **50 % overlap** between neighbouring segments — see the warning above.
- **Single channel**, so no array or direction-of-arrival work.
- **Class imbalance**, 4.5 drone : 1 background. Report balanced accuracy or F1;
  plain accuracy is misleading here.
- **Moderate annotator agreement** (κ = 0.587), which bounds achievable scores.
- Drone model, distance and manoeuvre are not annotated per segment; only
  presence is.

## Citation

```bibtex
@dataset{muhaciroglu_itu_aris_lab_drone,
  author    = {Muhac{\i}ro{\u{g}}lu, {\.I}hsan Mert and Akg{\"u}l, Tayfun},
  title     = {{ITU ARIS Lab. Acoustic Drone Dataset}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.22682339},
  url       = {https://doi.org/10.5281/zenodo.22682339}
}
```

## Loading

```python
import json, soundfile as sf
from pathlib import Path

root = Path("itu-aris-lab-acoustic-drone-dataset")
split = json.loads((root / "splits/split_interval_disjoint.json").read_text())

train = [p for p, s in split["assignment"].items() if s == "train"]
audio, sr = sf.read(root / train[0])          # (22050,) at 22 050 Hz

# label is in the path; group is in groups.json if you want to re-split
groups = {r["file"]: r["group"]
          for r in json.loads((root / "groups.json").read_text())["segments"]}
```

## Contact

İhsan Mert Muhacıroğlu — muhaciroglu19@itu.edu.tr
Electronics and Communication Engineering, Istanbul Technical University
