# Source selection before frozen inference

The model panel and decisions were fixed before external scoring. Neither source
was part of the task-specific training/development or consumed v2 assessment.
Exact original-byte, canonical-decoded and gain-normalized whole-file hashes
were compared with all 2,356 earlier recordings. No matches were found. This
does not rule out shifted excerpts, lossy copies, undisclosed ancestry or every
pretrained-encoder exposure.

| Source | Version and license | Evidence and limits |
| --- | --- | --- |
| [Ralf Saarmets FPV dataset](https://github.com/saarmets/fpv-drone-audio-classification-dataset/tree/387f7c52516d3633e81930503127446f82008b32) | Git revision `387f7c52516d3633e81930503127446f82008b32`; CC BY 4.0, Ralf Saarmets 2026 | All 37 WAV files: 21 publisher-positive and 16 publisher-negative, 5,091.51 s. Every original Git blob size/SHA-1 verified. Class folders provide weak recording labels. Session, microphone and original-flight relationships are undocumented; retain one conservative source component. This is unfamiliar-source testing with limited provenance, not 37 verified independent capture sessions. |
| [ITU ARIS Lab dataset](https://zenodo.org/records/22682339) / [author mirror](https://huggingface.co/datasets/imm61/itu-aris-lab-acoustic-drone-dataset/tree/c17df31a416412b245f371f127716de3db679c0f) | Zenodo v1.0.0; author mirror revision `c17df31a416412b245f371f127716de3db679c0f`; CC BY 4.0, Muhaciroglu and Akgul 2026 | Publisher describes one original continuous 3.12 h outdoor session, 35 manually annotated drone intervals and 1,000 background clips from unlabelled stretches. Metadata reports two annotators and Cohen kappa 0.5868742262. Individual disagreements and the full original waveform are not supplied. Flight ranges in the card are collection metadata, not this detector's operating range. |

The ITU archive's publisher MD5 and every listed SHA-256 entry were checked.
Reconstruction tested all 5,436 within-group candidate joins. All 4,456 positive
joins matched exactly, producing 35 intervals totaling 2,263 s. None of the 980
background candidate joins matched: those 1,000 clips remain separate one-second
inputs. The publisher's background blocks do not justify joining audio. Missing
original-time gaps are never synthesized or filled. All 1,035 reconstructed/raw
inputs remain one source-session component.

This matters for our existing product: its two-hit decisions require at least
two scores. A one-second file yields one padded score. A zero alarm count on
these backgrounds is therefore not a usable continuous-negative FPR estimate.
The assessment retains their actual replay behavior and separately reports
full-context/confirmation-eligible recordings. It does not report event recall,
audible onset delay or true false alarms per negative hour.

The small published-label task uses the original one-second segments for both
classes, the publisher's interval-disjoint assignment and an explicit exact
half-second cross-split fingerprint audit. It is same-session feasibility.
Those intervals/blocks cannot be counted as independently captured sessions.

## Other candidates inspected

- `imm-hub/ITU-ARIS-DRONE-DATASET` is an empty GitHub repository. The author's
  Hugging Face/Zenodo release above supplies the accessible data.
- [wnilmini/DroneAcousticDataset](https://github.com/wnilmini/DroneAcousticDataset)
  concerns drone-mounted microphones, measured impulse responses and motor
  noise. Its stated purpose does not establish an appropriate ground-recorded
  drone-presence/absence assessment, so it was not selected.
- [ComprehensiveDataset.ForAcousticDroneDetection](https://github.com/daisycramirez/ComprehensiveDataset.ForAcousticDroneDetection)
  describes drone categories and visualization. The inspected README did not
  establish an independently annotated audible positive/negative protocol;
  it was not selected for this assessment.
- AuDroK is an aggregation with some already-consumed source material, including
  Svanstrom. It is not accepted as an untouched assessment by aggregate name;
  any future use needs a sub-source ancestry audit.

Original metadata, pinned trees, reconstruction decisions, manifests, hashes
and frozen plans are retained in `runs/external-v3/` and this report directory.
Raw publisher audio stays outside tracked source. No audio reliability claim
is based on synthetic probes or a favorable demonstration.
