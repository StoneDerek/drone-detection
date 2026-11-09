# Multi-recording and Grad-CAM interface update

The public app remains https://drone-audio-lab.pages.dev . Frozen model identities, weights, scoring, thresholds and normalization are unchanged by this update.

The deployed [full browser workflow](public-browser.json) and [multi-recording/zoom interactions](public-interactions.json) both passed on the public URL, with the same JavaScript hash as the validated local build. The public interaction check evaluated seven loaded recordings (five examples plus two uploads) and exercised all model/view/zoom controls without audio uploads or page errors. Deployment identity and checksummed assets are in deployment.json and assets.json.

The [interaction test](interactions.json) loads five examples automatically, adds two uploads simultaneously, evaluates seven recordings, switches between retained results, exports hashed batch results, and confirms all eight spectrogram CNNs can be inspected independently of score selection. It verifies preservation of the selected interval when switching models and exact equality of displayed input matrices between the full-resolution CNN and its same-frontend frequency-context variant.

It also checks the native8×22 final attribution grid, heatmap-only/overlay modes, opacity0 matching the input image, native versus interpolated maps, linked rectangle zoom, wheel zoom, Shift-drag pan and reset. It observes no audio uploads or browser page errors.

The [full workflow check](local-browser.json) verifies playback/seek, the default CNN/SVM/YAMNet comparison, score/Grad-CAM agreement, genuine zero maps, both attribution targets, downloads, example selection, hashed listening notes and invalid audio handling. The [numerical port check](parity.json) covers the same176 classifier/context results and six Grad-CAM cases used previously. Every proof records the exact JavaScript SHA-256 used for deployment.

Ruff and all67 Python tests passed, including the API's native-attribution grid. New scientific findings are separate from UI tests: [structural coverage](../cnn-diagnosis-v1/frequency-coverage.json), [gain and level-transition controls](../level-transition-v1/RESULTS.md), and [competing explanations](../../docs/cnn-next-steps.md). These do not constitute a new field-accuracy benchmark.

The interface explains its model-specific preprocessing, independent map normalization, context-ending timestamps and the shared5% development false-positive calibration budget. The 3D/multi-view idea remains unimplemented.
