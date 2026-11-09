#!/usr/bin/env python3
"""Preserve the audit evidence and isolated control without replacing releases."""

import os
from pathlib import Path
import shutil
import tarfile

from drone_detector.common import digest, file_hash, read_json, write_json
from drone_detector.model import load_model


def preserve(source, destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        assert file_hash(destination)==file_hash(source),destination
    else:
        shutil.copyfile(source,destination)


def main():
    root,report,artifact = Path('runs/training-audit-v1'),Path('reports/training-audit-v1'),Path('artifacts/training-audit-v1')
    name = 'full-recording-frequency-seed42'
    for stage in ['trained','frozen']:
        for suffix in ['', '-at8', '-last']:
            for source in (root/stage/(name+suffix)).iterdir():
                if source.is_file():
                    preserve(source,artifact/stage/(name+suffix)/source.name)
    for source in root.glob('*.tar.gz'):
        preserve(source,artifact/'sources'/source.name)
    with tarfile.open(root/'full-recording-implementation.tar.gz') as archive:
        code = {item.name:__import__('hashlib').sha256(archive.extractfile(item).read()).hexdigest()
                for item in archive.getmembers() if item.isfile() and item.name.startswith('src/') and item.name.endswith('.py')}
    assert digest(code)==read_json(report/'full-recording-plan.json')['code']['source_sha256']
    with tarfile.open(report/'control-traces-and-selections.tar.gz','w:gz') as archive:
        for source in sorted((root/'traces').glob('*.jsonl.gz')):
            archive.add(source,arcname='traces/'+source.name)
        selection = root/'trained'/(name+'-audit')/'positive-selection.jsonl.gz'
        archive.add(selection,arcname='positive-selection.jsonl.gz')
    catalog = read_json('reports/cnn-foundations-v1/catalog.json')
    for suffix in ['', '-last']:
        model = load_model(artifact/'frozen'/(name+suffix),True)
        catalog['models'].append({'name':name+suffix,'bundle':os.path.relpath(artifact/'frozen'/(name+suffix),report)})
        assert model.metadata['training']['full_recording_bags']
    catalog['defaults'] = ['coherent-frequency-seed42',name,'svm-linear-fft4096','yamnet-default']
    catalog['scope'] = 'Training-pipeline audit lab. Full-recording controls are diagnostic, use the same existing development data and have not improved recall. Public deployment/defaults are unchanged.'
    write_json(report/'catalog.json',catalog)
    frontend = read_json(report/'frontend.json')
    gradients = read_json(report/'gradients-and-checkpoints.json')
    sources = read_json(report/'source-conversion.json')
    coverage = read_json(report/'context-coverage.json')
    channels = read_json(report/'channel-mix.json')
    control = read_json(report/(name+'-evaluation.json'))
    last = read_json(report/(name+'-last-evaluation.json'))
    baseline = read_json('reports/cnn-foundations-v1/coherent-frequency-seed42-evaluation.json')
    assert all(row['passed'] for row in [frontend,gradients,sources,coverage])
    tests = Path('runs/training-audit-full-tests.log')
    assert '80 passed' in tests.read_text()
    write_json(report/'software-checks.json',{'passed':True,'python_tests_passed':80,'test_log_sha256':file_hash(tests),
        'source_archive_matches_plan':True,'preserved_checkpoint_source_hash':digest(code),
        'scope':'Software checks and source identities; not acoustic performance trials'})
    lines = ['# Remaining training and preprocessing audit', '',
        'Two concrete issues remain after the earlier dropout-loss repair: sampled sub-bags change '
        'the meaning of whole-recording labels, and preparation unnecessarily reduced15 DDL '
        'training recordings from32-bit PCM to16-bit PCM. Both now have isolated, reviewable '
        'corrections. The controlled training correction did not improve detection recall; '
        'neither finding is established as the sole cause of the generalization failures.', '',
        'The audit opens only the existing405 training and210 development recordings. No fresh '
        'data, final-assessment audio, new audible labels, architecture search, deployed-model '
        'replacement or environment publication is involved.', '',
        '## 1. Whole-recording labels were applied to sampled sub-bags', '',
        'The repaired CNN used symmetric BCE of the maximum score for either class, but sampled '
        'up to8 windows per recording at each update. A positive recording guarantees at most '
        'that some part contains the event. A sampled subset need not contain it. Conversely, '
        'sampling negative windows can omit rare high-scoring confounders. The resulting '
        'stochastic objective is not an unbiased estimate of full-recording max-BCE.', '',
        'For a60-window recording with one+4 logit and59−4 logits, full-bag positive loss is '
        '0.0181. Uniform sample8 misses the rare maximum86.7% of the time, making expected '
        'positive loss3.4848. For a negative bag with the same scores, full loss4.0181 falls '
        'to expected sampled loss0.5515. This demonstrates a target discrepancy without '
        'claiming that the toy values describe audible event occupancy in real files.', '',
        'On the frozen selected CNN and all405 real training recordings, mean positive full-bag '
        'loss is0.1244 versus0.1517 averaged over its actual cyclic sub-bags; negative loss is '
        '0.4225 versus0.4091. At these already-trained weights, the observed discrepancy is '
        'modest. Frozen model maxima are not audible-window truth. See [complete audit](sampled-objective.json).', '',
        'The new optional `full_recording_bags=True` training path scans all cached windows '
        'in each selected recording, retains all tied maxima, then recomputes those instances '
        'with gradients. With dropout0 and no batch normalization, this gives the full max '
        'objective’s gradient without retaining every activation graph. Unsupported stochastic '
        'or alternate-pooling modes are rejected. The old experimental paths remain available '
        'to reproduce their controls.', '',
        'An independent per-recording BCE oracle matches the existing loss/gradients on8 real '
        'training recordings. Two-pass selection reduces179 forwarded contexts to8 gradient '
        'instances, matching full-backpropagation loss within1.5e-8 and gradients within '
        '4.43e-6. A separate exact-tie test verifies tied gradients are averaged. These are '
        'numerical controls, not detection trials.', '',
        '## 2. DDL precision was unnecessarily reduced', '',
        'All615 train/development file origins were checked against retained publisher bytes. '
        '507 files are byte-identical originals;93 D3 files were already16-bit PCM and selected '
        'the recorded physical channel correctly. The remaining15 DDL training clips were '
        'reconstructed from32-bit PCM fragments and written as16-bit PCM.', '',
        f"The DDL conversion introduces up to{sources['maximum_waveform_error']:.8f} absolute sample error. "
        f"Waveform power relative to conversion-error power ranges from{min(row['quantization_snr_db'] for row in sources['converted'] if row['source']=='ddl-real'):.1f} "
        f"to{max(row['quantization_snr_db'] for row in sources['converted'] if row['source']=='ddl-real'):.1f}dB; "
        'this is a quantization measurement, not drone-to-background acoustic SNR. No source '
        'clipping was introduced.', '',
        f"Original-precision FLOAT WAV copies now roundtrip exactly at the model’s float32 ingress precision. "
        f"All{sources['recovered_contexts_checked']} cached DDL contexts were compared with the same frozen CNN; "
        f"maximum score change is{sources['recovered_all_context_max_score_change']:.6f}. "
        'This does not establish a large current accuracy effect or rule out loss of faint cues.', '',
        '`scripts/restore_training_precision.py` creates a separate615-record train/development '
        'manifest with the same labels/splits, replacing only these15 file references and '
        'updating their hashes. The [repair record](precision-repair.json) identifies it. '
        'Original corpus files, existing caches and frozen models remain untouched. Future '
        'training on this variant requires a new cache/manifest identity. No model was trained '
        'on the repaired precision in this audit, so its effect is not mixed with the sampling control.', '',
        '## Other checks', '',
        '| Area | Result and qualification |', '| --- | --- |',
        "| Decode/resampling and amplitude units | All 615 ingresses agree with float64 channel-mean/polyphase references; correct header rates and sample counts. No common scale/rate error found. |",
        f"| Spectrogram cache | All {frontend['contexts_reextracted']:,} tensors exactly match fresh extraction. Independent direct-FFT references on first/middle/last contexts differ by at most {frontend['maximum_independent_fft_error']:.6f} on the 0–1 input scale. |",
        '| Axes and transform | Independent periodic-Hann FFT/filterbank implementation plus known-frequency software probes verify frequency/time orientation, FFT sizes, band placement and relative log-power scaling. Probe signals are never training or acoustic accuracy evidence. |',
        '| Normalization | Current CNNs use per-context spectral-peak reference and 80 dB log range, with no shared waveform RMS target. No train/validation normalization-mode mismatch found. |',
        f"| Stereo mixing | All 101 stereo files inspected, 301 first/middle/last contexts checked by frequency band. Worst broad-band attenuation is {channels['worst_band_mean_channel_attenuation_db']:.2f} dB versus mean channel power; at most {100*channels['maximum_energy_fraction_in_deep_notches']:.2f}% of spectral energy lies in bins attenuated over 10 dB. This does not rule out narrow harmonic cancellation or establish the best microphone choice. |",
        '| Class mapping, losses and gradients | Raw logits feed BCE once; correct positive/negative signs. Independent per-recording BCE agrees with segment pooling. Current balanced batches have 4 recordings/class and no second class-weight multiplier. No reversed labels, missing gradient or double sigmoid found. |',
        f"| Optimizer | Independent first-step Adam calculation matches weights within {gradients['independent_first_adam_step_weight_error']:.3g}. `clipnorm=1` clips each parameter tensor; logged global gradient norms are measured before clipping. This verifies implementation, not that the learning rate is optimal or convergence complete. |",
        f"| Training/inference mode | Dropout 0 models produce identical train/inference logits; individual/batched logits differ by at most {gradients['batch_single_logit_error']:.3g} in the checked real batch. No batch-normalization layers exist in these CNNs. |",
        '| Checkpoint selection | All 9 selected/best-at8/last packages for the three current CNNs exactly match their archived epoch arrays. Selected epochs minimize the declared development loss. Calibrated packages preserve those weights. |',
        '| Split isolation | Optimizer inputs are selected only from train records; development inputs are used for checkpoint selection/calibration. All 615 cache identities, sources, labels and original file hashes are checked. Existing weak labels are not independently verified by this audit. |',
        f"| Padding and replay | {coverage['padded_files']} development files have a padded tail. Removing tails changes none of the three current CNNs’ recording alarm outcomes at their original decisions. |",
        f"| Context coverage | Uniform validation contains {coverage['uniform_contexts']:,} contexts versus {coverage['dense_full_contexts']:,} full scored replay contexts. All {coverage['dense_scored_contexts_freshly_verified']:,} scored dense tensors were freshly re-extracted exactly, including padding, and timestamps checked. The 60-window cap remains a training/product distribution difference. |", '',
        '## Same-data training control', '',
        'One seed42 run changes only instance selection to full cached-recording maxima. '
        'The256×172 frontend,154,689-parameter frequency readout, original precision/data, '
        'initial weights, balanced4+4 batches,51 steps/epoch, Adam0.001/per-tensor clipnorm1, '
        'L2=1e-4, dropout0 and24-epoch budget match the sampled baseline. Checkpoints minimize '
        'the same deterministic full-cache development max-BCE every2 epochs. All contexts '
        'used for selection count as seen, even though only winners receive classification gradients.', '',
        '| Model/checkpoint | Epoch | TP /22 | FP /188 | Dense max-score AUC |', '| --- | ---: | ---: | ---: | ---: |']
    for title,row in [('Preserved sampled baseline',baseline),('Full-recording selected/best-at8',control),('Full-recording last',last)]:
        metric = row['development']['metrics']
        lines.append(f"| {title} | {row['training']['selected_epoch']} | {metric['confusion']['tp']} | {metric['confusion']['fp']} | {metric['roc_auc_max_window_score']:.3f} |")
    lines += ['', 'All alarm thresholds are calibrated on the same reused development data under '
        'the5% negative-file budget. These matching FP counts are a constraint, not independently '
        'measured field FPR. The exact-target control improves some ranking measures but detects '
        '14/22 versus16/22 for the selected sampled baseline. It is not promoted. Late full-target '
        'training reduces training loss while development loss deteriorates and oscillates; '
        'training dynamics and weak-label generalization are not declared solved.', '',
        '![Deterministic full-cache training and development losses](training-control.png)', '',
        'The audit clears several concrete implementation suspects and identifies two repairable '
        'issues. It does not prove the remaining failures are exclusively data-related, that '
        'hyperparameters are optimal, or that either repair alone will improve field performance.', '',
        '## Reproduce and inspect', '',
        'Run `scripts/audit_audio_frontend.py`, `scripts/audit_source_conversion.py`, '
        '`scripts/audit_training_objective.py`, `scripts/audit_cnn_gradients.py`, '
        '`scripts/audit_channel_mix.py` and `scripts/audit_context_coverage.py` with the '
        'existing corpus/caches. The source-conversion audit uses cached original bytes only; '
        'there is no network fallback. Plans and execution snapshots precede the conditional '
        'training run. The initial gradient-oracle identity is retained separately from a '
        'later correction to per-layer display keys; numerical checks are unchanged.', '',
        'All80 Python tests and Ruff pass. The control’s selected/early/last numeric model '
        'packages, exact execution code, compressed traces and instance-selection audit are '
        'retained under `artifacts/training-audit-v1/` and this report. Original-precision '
        'audio remains in ignored `runs/`; no raw publisher recordings are packaged.', '',
        'For optional local comparison, run `bash scripts/start_lab.sh --catalog '
        'reports/training-audit-v1/catalog.json`. The public Cloudflare deployment and existing '
        'model packages are unchanged.', '']
    # Prose formatting leaves literal code and model identifiers intact.
    import importlib.util
    spec = importlib.util.spec_from_file_location('formatting','scripts/package_cnn_foundations.py')
    formatting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formatting)
    prose = formatting.readable_prose(lines).replace('D 3','D3').replace('float 32','float32')
    prose = prose.replace('separate615-record','separate 615-record').replace('these15 file','these 15 file')
    (report/'RESULTS.md').write_text(prose)
    make_plot(baseline,control,report)
    (artifact/'README.md').write_text('# Training/preprocessing audit\n\nSee[the report](../../reports/training-audit-v1/RESULTS.md). This package preserves a diagnostic full-recording training control; it did not improve development recall and is not deployed. No raw audio is included. Existing frozen releases are unchanged.\n')
    with tarfile.open(artifact/'sources/audit-source.tar.gz','w:gz') as archive:
        for directory in ['src','tests']:
            for source in sorted(Path(directory).rglob('*.py')):
                archive.add(source,arcname=str(source))
        for source in sorted(Path('scripts').glob('*.py')):
            archive.add(source,arcname=str(source))
        for source in ['pyproject.toml','uv.lock','README.md','docs/model-and-data.md']:
            archive.add(source,arcname=source)
    files = sorted(path for path in artifact.rglob('*') if path.is_file() and path.name!='SHA256SUMS')
    (artifact/'SHA256SUMS').write_text(''.join(f'{file_hash(path)}  {path.relative_to(artifact)}\n' for path in files))
    write_json(report/'summary.json',{'status':'Completed bounded audit; residual training/generalization issues remain',
        'verified_issues':['Sampled sub-bag target differs from whole-recording weak label','DDL32-bit→16-bit conversion loses precision'],
        'files_audited':615,'cached_contexts_reextracted':7299,'original_precision_files_recovered':15,
        'tests_passed':80,'control_detected_positives':14,'baseline_detected_positives':16,
        'development_negatives_alarmed':9,'development_negative_files':188,
        'existing_models_and_public_deployment_unchanged':True,'raw_audio_packaged':False,
        'artifact_files':len(files),'artifact_checksums_sha256':file_hash(artifact/'SHA256SUMS')})
    print('Audit report,isolated control models and exact execution code preserved.')


def make_plot(baseline,control,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,panels = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for label,row in [('Sampled8 baseline',baseline),('Full cached recording',control)]:
        history = [item for item in row['training']['history'] if 'train_consistent_max_balanced_loss' in item]
        epochs = [item['epoch'] for item in history]
        panels[0].plot(epochs,[item['train_consistent_max_balanced_loss'] for item in history],marker='.',label=label)
        panels[1].plot(epochs,[item['development_consistent_max_balanced_loss'] for item in history],marker='.',label=label)
    for panel,title in zip(panels,['Training, deterministic full-cache loss','Development, deterministic full-cache loss'],strict=True):
        panel.set(title=title,xlabel='Epoch',ylabel='Balanced max-BCE, no L2 term')
        panel.grid(alpha=.2)
        panel.legend(fontsize=8)
    fig.suptitle('Same inputs,encoder,readout,optimizer and seed; sampling target differs\nExisting train/development data only, no fresh acoustic assessment',fontsize=11)
    fig.savefig(report/'training-control.png',dpi=145)
    plt.close(fig)


if __name__=='__main__':
    main()
