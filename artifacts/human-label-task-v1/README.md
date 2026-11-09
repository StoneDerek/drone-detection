# Published-label feasibility classifier

This is a separate, one-second, same-session diagnostic classifier. It is not a
replacement for a deployed detector. Its numeric mean, scale, coefficient and
intercept arrays are stored without pickle; `model.json` states the descriptor
and protocol identity. `execution.py` is the exact executed training source.

Input is one second of mono 16 kHz audio. A 4,096-sample Hann STFT with 256-sample
hop supplies 47 complete frames. The 256 uniform-Hz relative log-power bands
produce temporal mean/std features; the positive score is
`sigmoid(((features - mean) / scale) @ coef + intercept[0])`. Threshold 0.5 and
maximum segment score provide the separately reported interval decision.

Labels came from the CC BY 4.0 ITU ARIS dataset by Muhaciroglu and Akgul, 2026,
Zenodo DOI 10.5281/zenodo.22682339. It has one session and incomplete audible
absence/disagreement information. No raw audio is bundled. Full label limits,
split audit, predictions and results are in `reports/human-label-task-v1/`.
