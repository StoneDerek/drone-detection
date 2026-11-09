# Paired scratch-CNN and numerical-reference diagnostics

Eight selected CNNs and their last checkpoints are retained across four amplitude
rules and two seeds. Four train-fitted descriptor-logistic controls use the same
published labels and interval/class weights. All are diagnostic packages, not
product replacements or independently validated acoustic detectors.

CNN packages use the project's numeric NPZ/JSON format and are loaded with
`drone_detector.model.load_model`. Use one second of16kHz mono audio and call
`score`; there are no promoted/frozen alarm decisions. Exact-RMS packages declare
`normalize_rms: "exact"`. This numerical mode retains the silence guard, targets
RMS0.05 without caps, and may have float peaks above1. It does not clip, play audio
at a new gain or claim physical SPL calibration. Boolean policies retain their
previous bounded behavior.

The non-neural packages store complete feature configuration plus train-only
mean, scale, coefficients and intercept. Apply the declared spectrogram transform,
take each band's temporal mean/std, standardize and apply the sigmoid to the
linear margin. The explicit one-second input has47 valid STFT frames and no
padding.

`execution.py` and `original-execution-source.tar.gz` preserve the original six
runs' exact source. The exact-RMS follow-up has its own execution source and
metadata. SHA256SUMS covers all retained files. Raw audio and multi-hundred-MB
feature caches stay in ignored runs directories. Data attribution: ITU ARIS Lab,
Muhaciroglu and Akgul2026, CC BY4.0, Zenodo10.5281/zenodo.22682339.

Full results and limitations are in `reports/learnability-normalization-v1/`.
