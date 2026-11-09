# Product update validation

Version 0.2.0 adds audio-quality review states and trace measurements. The frozen
version-2 default model and its assessment are preserved. This update does not
establish a new acoustic operating range or improve the measured field results.

- **58 software tests passed**, including available-context normalization before
  padding, bounded gain, silence handling, clipped/low-frequency input,
  file/API trace agreement and preservation of active alarms when quality flags
  are present. Static and repository whitespace checks passed.
- Chromium passed upload, playback/seek, alarm display, timestamped rows, JSON
  export and invalid-audio handling on existing real development recordings.
  The long recording was 365.42 seconds with 726 score rows. These are software
  checks, not 726 independent trials.
- Generated tones, clipping, short audio and silence checked the new display states:
  **Review audio quality**, **Silent input** and **Insufficient audio**, alongside
  the existing **ALARM** and **No alarm** states. Generated signals supply no
  acoustic detection evidence. No page errors were observed.
- The installable wheel and hash-checked dependencies were installed in a new
  virtual environment. Python `-I` ran from an unrelated directory using a
  copied model and an existing real development recording. All 14 windows
  matched the prior 0.1.0 package exactly in scores, alarms, reasons, padding and
  context timestamps. The installed module came from that environment's
  `site-packages`, rather than an editable checkout.
- The isolated wheel also served upload analysis without the evaluation corpus
  or examples, and passed the quality-state browser checks. It emitted the new
  measurements in JSON while preserving the default model identity.
- All 14 checksums in the original `artifacts/real-v2/SHA256SUMS` still pass.
  Its original wheel, model packages and evaluated-source snapshot were not
  overwritten.

The first long-upload browser run exceeded its 30-second test deadline while
training and other checks were running. The repeated run completed with a
120-second response allowance. The server now handles a client disconnect
without attempting a second error response. This is not a new latency benchmark;
cold-start and concurrent-load times are retained in the JSON where measured.

Evidence: [real upload/playback](browser-verification.json),
[quality states](quality-browser-verification.json),
[default regression](default-parity.json),
[isolated inference](clean-install-verification.json),
[standalone upload](standalone-upload-verification.json), and
[isolated browser check](clean-quality-browser-verification.json).

[Real-input screenshot](product-screenshot.png) ·
[Quality-review screenshot](clean-quality-screenshot.png)

The original real-audio assessment remains **100/133 publisher-positive files
detected and 126/1608 publisher-negative files alarmed**. Those files share source
and capture groups; they are not independent field trials. The original event
recall, false alarms per negative hour and detection-delay contract remains
unassessed. No physical microphone capture, deployment or environment publication
was performed in this update.
