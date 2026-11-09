# Versioned real-audio sources

The public-source retrieval/preparation scripts pin revisions and verify original
Git/LFS content. The manifest records actual hashes, origin/label evidence,
original IDs, source versions, and conservative groups. Labels remain weak unless
independently temporally reviewed. No synthetic waveform is used as real evidence.

| Source | Role and version | Terms / provenance limit |
| --- | --- | --- |
| [IntRoLab D3Set](https://github.com/introlab/d3set) / [author-linked Kaggle mirror](https://www.kaggle.com/datasets/francoismarcoux1212/d3set-introlab) | Kaggle v1, archive SHA-256 `955f0010ec37f0c4b580e75a95b6e5782a676763277211a0d328ac6d32f9c860`; primary Git content verified. Ground-array physical channel 0, no onboard audio. Flight days grouped; overlapping Delivery/SNR drone derivatives excluded. Unknown-parent SNR background excluded in v2. | GPL-3.0 repository; imported noise retains source terms (publisher cites Pixabay). Seven helicopter originals also appear in Svanstrom and move together in training. |
| [Svanstrom](https://github.com/DroneDetectionThesis/Drone-detection-dataset) | Revision `442d0c6750708cc221636f7a1fd31cbd8cb0aeb8`; entire unknown-original source grouped. Initial v1 assessment is consumed; v2 moves the whole connected source/duplicate component into training. | CC0 release; publisher classes, no independent timing or audibility review. |
| [Batear](https://github.com/batear-io/batear-datasets) | Revision `d0d6e67d9e7d167be9e0db249ef6a44acb52f732`; original LFS SHA-256 checked. Earlier FPV/rural data is development; April 28 capture day remains fresh assessment. | MIT. Whole days grouped; telemetry/flight directory does not prove every interval is acoustically audible or negative. |
| [DDL real-field archive](https://zenodo.org/records/6459183) | Real archive only. Contiguous sequences reconstructed using measured WAV headers and one physical channel. March 29 is training; March 31 is fresh assessment. Three metadata-positioned excerpts per usable flight remain grouped by day. | CC BY 4.0. Empty/nonstandard fragments are inventoried and excluded from reconstruction. One flight has no valid ten-second run. The separate synthetic archive is not fetched. |
| [ESC](https://github.com/karolpiczak/ESC-50) | Revision `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`. Original Freesound IDs remain together. ESC-10 is used for earlier training/development; unused categories and nonoverlapping original IDs are fresh assessment. | ESC-10: CC BY. Remaining ESC-50: CC BY-NC, assessment only. Per-clip attribution remains in the source LICENSE/metadata fetched with the corpus. |
| [Google YAMNet v1](https://tfhub.dev/google/yamnet/1) | Frozen AudioSet-pretrained 1024-D embeddings, official SavedModel. Archive SHA-256 `b80da2a1a56926fb0767205051a200dd7b3beaf3ea1ea126c42a53943996e5e0`. | Google release; attribution/license retained. Public pretraining overlap is not ruled out for every source. |

DDL uses HTTPS range retrieval to avoid fetching unused audio. Every selected ZIP
member is CRC-checked and its content SHA retained. The publisher's **whole**
archive MD5 is recorded but is **not** claimed verified for a partial download.
These integrity checks do not validate labels, recording independence or field
operating conditions. Raw audio remains in ignored `runs/` directories.

The supplied historical review/packet contains no WAVs or model weights. Historical
Kaggle mirrors with obscured original identities or conflicting upstream terms
were not accepted merely because they advertised many files or a permissive label.
