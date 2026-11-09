# Fixed one-second published-label feasibility task

A simple, predeclared classifier can separate the held-out published interval
labels within this single recording session. It is much less sensitive when
each one-second segment is judged separately. The latter labels inherit their
parent interval; this experiment does not establish which individual seconds
contain audible drone sound.

## Fixed task and split

Both classes use the original one-second ITU ARIS exports. No background clips
are looped, stretched or falsely concatenated to obtain three-second inputs.
The publisher's interval-disjoint assignment is retained. All 55 interval/block
groups stay in one split, and no exact raw half-second fingerprint crosses
splits. These are portions of one session, not 55 independent capture sessions.

The descriptor uses 256 uniformly spaced frequency bands over 50–7,600 Hz,
4,096-sample periodic-Hann STFT, 256-sample hop and 16 kHz mono audio. Exactly
47 complete STFT frames fit each second. Relative log power uses the peak of
those valid frames and an 80 dB floor; no padded frames contribute. Per-band
temporal mean/std produce 512 features. There is no shared-reference waveform
RMS normalization.

Standardization is fitted only on train data, with equal interval/block mass.
L2 logistic regression uses C=1 and seed 42. A bag-mean initialization is followed
by four train-only maximum-positive-instance updates; negative bags retain all
their segments. Interval mass and class mass are balanced. The decision
threshold is fixed at 0.5. There is no hyperparameter or threshold sweep.

The execution and protocol hashes, predictions and numeric parameters are
retained. This is a separate diagnostic classifier; it has no position in the
deployed product catalog.

## Results

| Split | Positive segments detected | Negative segments classified positive | Segment ROC AUC | Positive intervals detected by any segment | Negative blocks classified positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,601 / 3,144 | 0 / 700 | 0.947 | 16 / 16 | 0 / 14 |
| Validation | 193 / 673 | 0 / 150 | 0.926 | 9 / 9 | 0 / 3 |
| Test | 183 / 674 | 0 / 150 | 0.888 | 10 / 10 | 0 / 3 |

The test segment recall is 27.2%; interval presence is detected in 10/10 test
intervals. A single high-scoring second can satisfy an interval label. This
illustrates why good interval-level performance alone cannot establish reliable
continuous detection. Segment counts include overlapping examples and are not
independent evidence. No uncertainty claim treats 824 test segments as 824
capture trials. The three negative test blocks are especially limited evidence.

## Label and interpretation limits

The publisher reports two annotators and Cohen kappa 0.5868742262. Positive
intervals were manually selected; background clips came from unlabelled stretches.
Individual disagreements and independently reviewed absence intervals are not
supplied. This provides a more explicit human-label task than flight metadata,
but does not deliver a complete trustworthy audible-event benchmark.

The result establishes learnability of these published group labels under the
fixed same-session protocol. It does not prove drone harmonic relationships were
learned, rule out session-specific cues, or justify a field range, event recall,
false alarms/hour or live-input claim. It is not directly comparable with the
frozen product's three-second, two-hit recording assessment: input length,
training data and decision aggregation differ.

The next missing evidence remains a separate capture session with reviewed
audible positive and negative intervals, including annotation disagreements or
ambiguous regions. Further tuning on this task cannot supply that evidence.
