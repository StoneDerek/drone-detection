# Published human-label search

The user prefers published human annotations. Two primary releases were inspected
without opening a new detector assessment. Their original metadata, checksummed
datasheet and spreadsheet are retained in the research directory.

## Kenya Forest Sounds

[Primary release](https://zenodo.org/records/21443074), CC BY 4.0, and the
[maintainers' repository](https://github.com/SCES-Makerspace-Lab/Kenya-Forest-Sounds-Dataset)
describe human review of original field audio. The datasheet says several
annotators independently supplied labels, with cross-checks. Recordings were
collected during visits in March 2024, May 2024 and January 2025 using a Zoom H2n.

The spreadsheet has five primary-label Drone clips:
`ZOOM0071_16_16k.wav` through `ZOOM0071_20_16k.wav`, each about five seconds.
They share the same apparent original recording prefix. The release's file count
therefore does not establish five independent drone captures. A different row
has primary label Axe and an additional Drone entry, illustrating why secondary
spreadsheet cells must not silently be converted into clean class truth.

The datasheet explicitly says overlapping sounds were labelled by the **dominant
sound**. Thus a clip labelled Wind, Vehicle or Chainsaw is not a verified
drone-absent interval. It also says full raw recordings are available only to the
research team. The public isolated clips do not establish continuous audible
on/off boundaries or reviewed negative duration.

This is a useful candidate for a small human-labelled positive check and hard
confounder review. It does not yet qualify as the planned complete audible-event
assessment. No detector scores were inspected on this source.

## IDMT Berne 2022

[Primary release](https://zenodo.org/records/17525811), CC BY 4.0, contains eight
flight recordings totaling about 106 minutes, distributed in 30-second segments.
The description supplies synchronized GPS-derived drone coordinates. Those are
physical-flight/localization annotations, not published reviewed audible-event
boundaries or proof of acoustic absence. The inspected metadata does not resolve
the missing negative/audibility evidence. No model scores were generated.

## Exact remaining gap

The missing input is a new recording session containing matched recording
conditions for audible drone and background, with human-reviewed audible
intervals, uncertain regions and negative stretches. Human review of the dominant
class and flight-position labels answer different questions. These releases
should not be upgraded to event truth merely to produce another benchmark.
