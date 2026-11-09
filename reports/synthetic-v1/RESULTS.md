# Controlled synthetic results

No measured reliability for authentic drones. No real microphones, distance, outdoor environment or SNR range validated.

Selected on development only: **gain-aug-42**. One final batch compared it with the frozen baseline.

## Development ablations

| Arm | Recall | FA/h | p95 delay (s) | Gain score drift | Gain flips | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| absolute | 1.000 | 0.00 | 3.0014427287501007 | 0.034913 | 0.057 | 1.74 |
| relative | 1.000 | 0.00 | 2.5011071409000123 | 0.000000 | 0.000 | 1.31 |
| harmonic | 1.000 | 0.00 | 2.501281163550016 | 0.000000 | 0.000 | 1.47 |
| temporal | 1.000 | 0.00 | 2.501429830749885 | 0.000000 | 0.000 | 1.37 |
| gain-aug-7 | 1.000 | 0.00 | 2.5010939719499903 | 0.000000 | 0.000 | 2.20 |
| gain-aug-42 | 1.000 | 0.00 | 2.5011999509001273 | 0.000000 | 0.000 | 2.16 |
| gain-aug-123 | 1.000 | 0.00 | 2.5014306217500346 | 0.000000 | 0.000 | 2.21 |

## Frozen generator holdout

This is a held-out signal generator, not an independent acoustic dataset. Thresholds and alarm confirmation were frozen before scoring.

### absolute

Model `fe7502bf87afbf78d5f74e7d88952f2f2bef0ade68f0433337d10d7e0a463edf`.

- Decision: `{'off_hits': 2, 'off_threshold': 0.48750000000000004, 'on_hits': 1, 'on_threshold': 0.65}`.
- 30 files / 10 generator session groups; 10 eligible events, 0.0806 negative hours.
- Recall 0.900 (9/10); cluster 95% interval [0.7, 1.0]; conservative lower bound 0.555.
- 1 false alarm events; 12.41/h; conservative upper 95% bound 69.17/h.
- p95 delay among detections 4.201276801800076 s; 2 recording-level errors; precision 0.9.
- End-to-end real-time factor 0.0021; hardware INTEL(R) XEON(R) PLATINUM 8573C.
- Numerical checks: `{'delay': False, 'false_alarms': False, 'recall': False, 'runtime_on_reported_hardware': True}`. Evidence checks: `{'authentic_only': False, 'devices': False, 'drone_types': False, 'events': False, 'independent_groups': False, 'negative_hours': False, 'sources': False}`.

Observed failures:

- `g2-s02-hard_negative` (synthetic-generator-2): 0 missed events, 1 false alarms; alarm intervals [[3.0, 12.0]].
- `g2-s07-positive` (synthetic-generator-2): 1 missed events, 0 false alarms; alarm intervals [].

### gain-aug-42

Model `aa9a710c030a5a628f320d9741d1f920581d84b9816002dbfb1cfac0c6b8a330`.

- Decision: `{'off_hits': 2, 'off_threshold': 0.6000000000000001, 'on_hits': 1, 'on_threshold': 0.8}`.
- 30 files / 10 generator session groups; 10 eligible events, 0.0806 negative hours.
- Recall 0.900 (9/10); cluster 95% interval [0.7, 1.0]; conservative lower bound 0.555.
- 0 false alarm events; 0.00/h; conservative upper 95% bound 45.79/h.
- p95 delay among detections 4.201023517800058 s; 1 recording-level errors; precision 1.0.
- End-to-end real-time factor 0.0021; hardware INTEL(R) XEON(R) PLATINUM 8573C.
- Numerical checks: `{'delay': False, 'false_alarms': False, 'recall': False, 'runtime_on_reported_hardware': True}`. Evidence checks: `{'authentic_only': False, 'devices': False, 'drone_types': False, 'events': False, 'independent_groups': False, 'negative_hours': False, 'sources': False}`.

Observed failures:

- `g2-s02-positive` (synthetic-generator-2): 1 missed events, 0 false alarms; alarm intervals [].

## Interpretation and next experiment

Per-window relative log-Mel removes scalar gain from the representation by construction; the gain probe measures that software property. Tonal/harmonic features are also activated by generated motor/fan-like negatives. These observations do not establish the cause of historical failures or an advantage on authentic capture.

The current experiment compares representations and alarm policies on fixed development recordings. Seed variants only apply to stochastic gain augmentation. Larger audio-pretrained models are deferred: there is no verified audio with which to assess generalization. No distances, SNR range, microphone coverage, drone coverage or field-readiness claim is supported.

Next: audit the original audio archive and recover original/session identities; independently annotate full recordings, then rerun the frozen protocol on authentic audio with 100+ events, 30+ groups and 10+ negative hours across at least two sources/devices and three drone types. See docs/collection-protocol.md. Do not relabel generated signals as collected drone data.

The demo's six fixed recordings were chosen by ID before training; all positives and ordinary/difficult negatives from those sessions are included. A separately labeled post-hoc appendix includes every candidate holdout failure. All benchmark traces are retained. The demo is a convenience, not the benchmark.
