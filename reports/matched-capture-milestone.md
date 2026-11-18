# Acoustic detection across matched recording conditions

The next milestone is to obtain a dataset that can answer whether a detector
learns audible drone presence across recording conditions. Each condition needs
both audible drone and background, recorded through the same microphone and
gain chain. The project has enough model baselines to test that change.

The old ITU ARIS experiment showed that a simple classifier could separate
published interval labels within one session. At its fixed cutoff it detected
10 of 10 test intervals, but only 183 of 674 positive one-second crops. Label
inheritance and overlapping crops limit what those counts say about audibility.
The later controlled field release provided explicit backgrounds but shared one
site, array and drone family, and many positive captures lacked dated session
provenance. These experiments already exposed the missing measurement.
[ITU experiment](human-label-task-v1/RESULTS.md) ·
[Controlled field experiment](cross-condition-v1/RESULTS.md)

## What the public audio can support today

I joined all 249 original labelled FPV intervals to the existing annotation
and raw-link ledgers. A view is the complete archive folder immediately above
the class folders, including any nested device or take folder. The capture
group is the shared outer capture folder. Under each criterion I retained views
with both publisher classes.

| Interval evidence | Both-class views | Capture folders | Drone / background intervals in those views |
|---|---:|---:|---:|
| Original publisher folder label | 7 | 3 | 119 / 108 |
| Unique duration-consistent annotation candidate with no opposing-label overlap | 5 | 2 | 63 / 66 |
| Exact decoded link to an available raw member | 0 | 0 | 0 / 0 |

The 129 candidate intervals are a bounded public-data listening pool. Their
label and timing candidates still need review. The twelve exact raw links in
the release are all drone-labelled. There is no both-class view with verified
raw links from which to measure continuous background exposure. Additional
phone views do not provide additional independent capture groups, and the
publisher's exported splits share capture groups.
[Archive and ledger provenance](fpv-lineage-v1/README.md)

## The experiment that changes the question

1. **Qualify the sessions before fitting.** Start from original recordings and
   verified intervals. Have listeners mark audible drone, reviewed background
   and uncertain spans without seeing detector scores. Retain quiet and
   intermittent drones, hard mechanical negatives and continuous background
   when the source supplies it. Group all derivatives and simultaneous views
   with the parent capture. See whether a public release or a small shared
   collection can supply the remaining independent sessions.
2. **Freeze a test at the capture level.** Reserve entire unfamiliar
   campaigns and their matched backgrounds before inspecting scores. Use separate
   sessions for training and threshold selection. For each held campaign,
   report events detected, false alarms per reviewed background hour and
   detection delay. For clip-only sources report clip decisions and the number
   of distinct captures. Choose the acceptable alarm budget before evaluation.
3. **Change the examples with the encoder fixed.** Compare one existing frozen
   encoder and the same head-fitting recipe under two training conditions. One
   uses the existing pooled public examples. The other uses same-session
   audible drone/background pairs. Match example count, duration, fitting and
   threshold-selection rules as closely as the available sessions permit.
   Keep the existing detector as an out-of-the-box reference. The main result
   is whether pairing improves recall at the prespecified alarm budget across
   held capture conditions.

If the paired training improves on fresh sessions, the next investment is more
independent paired recordings. If listeners can recognize the same intervals
and the paired classifier still fails, the next model experiment should test
one specific hypothesis about the signal, such as whether tracking tonal
structure over time improves discrimination from machinery. If listeners
also disagree on many intervals, measure performance against audibility and
context before changing the model.

The verified broader BEATs result reinforces the distinction between a better
development comparison and a reliable field detector. Fine-tuning improves
AuDroK recall from 25/58 to 36/58 and lowers FSD false positives from 397/7,931
to 301/7,931 against its matched frozen control. The current-mix adapted arm
detects 34/58 with 141 FSD false positives under the same cutoff-selection rule.
The 58 positives still come from one campaign and the FSD negatives come from
a different recording source. [Broader BEATs paired result](beats-adaptation-v1/paired-analysis/audrok-dresden/broader/README.md)
