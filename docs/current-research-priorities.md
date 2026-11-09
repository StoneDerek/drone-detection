# User priorities and current research state

The central objective is to understand what distinguishes drone audio, verify
that the pipeline preserves it, and establish that the model learns it before
adding complexity. A working detector and an explanation of its failures remain
the goal. This note records the user's recovered preferences and links them to
the current evidence; it is not a claim that the research problem is solved.

| Priority | Established work | Still open |
| --- | --- | --- |
| Fundamental scratch-CNN problem | Actual parameter counts, frequency-path coverage, dropout/max-loss defect, sampled-bag target mismatch, gradients, optimizer, caches, checkpoints and replay audited. Tiny CNN memorization on real examples works. | No single sufficient cause explains cross-source failures. Repairs do not prove that CNNs inherently underperform YAMNet. |
| Varied real data and stronger tests | Original405/210 train/development files, prior consumed assessment, frozen FPV/ITU source check, hashes and conservative grouping. | Few positive capture groups, weak presence labels and incomplete audible negatives. Repeated crops are not independent evidence. |
| Fine acoustic detail | Genuine4096-sample analysis windows, linear/mel/filterbank comparisons, real tensor dimensions and physical receptive fields. | Finer input does not prove useful harmonic or motion relationships are learned. |
| Richer representations | Multi-view/context and3D CNN suggestions retained as future hypotheses. | No implementation requested; first identify useful distinctions from actual cases. |
| Normalization and quieter sounds | Peak-relative cancellation, fixed numerical reference, available-context gain rules and explicit capped versus exact RMS controls. | Which rule aids learning and transfer under trustworthy labels; no physical SPL/SNR calibration follows. |
| Specific examples and simpler tasks | Real spectrograms, score disagreements, synchronized-versus-independent time probes, fixed within-session label task. | Published labels do not establish every second's audibility. Keep observations distinct from causal explanations. |
| Inspectable behavior | Cloudflare lab, retained recordings, playback, zoom, selectable Grad-CAM and exports are established. | A heatmap is not proof of recognition; preserve zero maps and actual native resolution. |
| Neural versus non-neural | Frozen SVM/YAMNet/CNN source comparisons and matched-input weighted logistic controls. | Any observed advantage is conditional on data, target, representation and operating point. |

Published human annotations are preferred over asking the user to label a batch.
Do not reset old project files, republish the environment, invent listening
judgments, or implement deferred richer-input ideas as an inferred request.
