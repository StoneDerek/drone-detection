#!/usr/bin/env python3
"""Preserve completed controls and write a source-aware development account."""

import os
import re
import shutil
import tarfile
from pathlib import Path

import numpy as np

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.model import load_model

ROOT = Path('runs/cnn-foundations-v1')
REPORT = Path('reports/cnn-foundations-v1')
ARTIFACT = Path('artifacts/cnn-foundations-v1')
REQUIRED = ['compact8', 'coverage8', 'medium-sampled-max', 'medium-full-smooth',
            'medium-balanced-max', 'medium-full-max', 'medium-balanced-global-head',
            'medium-balanced-frequency-head', 'coherent-global-seed42',
            'coherent-frequency-seed42', 'coherent-frequency-seed7']


def preserve(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ValueError(f'Packaged content differs: {destination}')
    else:
        shutil.copyfile(source, destination)


def readable_prose(lines):
    """Space numerical descriptions without altering model IDs or code/links."""
    result = []
    for line in lines:
        if line.startswith(('|', '`', '!')):
            result.append(line)
            continue
        pieces = re.split(r'(`[^`]*`|\[[^\]]*\]\([^)]*\))', line)
        for index in range(0, len(pieces), 2):
            value = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', pieces[index])
            value = re.sub(r'(?<=\d)(?=(?:kHz|Hz|dB|seconds|second)\b)', ' ', value)
            value = re.sub(r'(?<=[^\d])([,:;])(?=\d)', r'\1 ', value)
            pieces[index] = value.replace('L 2=', 'L2=').replace('logit−', 'logit −').replace('and+', 'and +').replace('Tensor[', 'Tensor [')
        result.append(''.join(pieces))
    return '\n'.join(result)


def make_figures(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    audit = read_json(REPORT / 'mil-objective-audit.json')
    checkpoints = audit['checkpoints']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    n = np.arange(len(checkpoints))
    axes[0].bar(n-.18, [r['old_loss_with_dropout_mean'] for r in checkpoints.values()], .36, label='Dropout on, identical inputs/weights')
    axes[0].bar(n+.18, [r['old_loss_inference'] for r in checkpoints.values()], .36, label='Dropout off, product inference')
    axes[0].set_xticks(n, ['Tiny original', 'Larger last', 'Smoothed last', 'Balanced last'])
    axes[0].tick_params(axis='x', labelsize=8)
    axes[0].set_ylabel('Old asymmetric loss, fixed first ≤8 contexts/file')
    axes[0].set_title('Training noise can make an unchanged model look better')
    axes[0].legend(fontsize=8)
    for name, title in [('medium-balanced-frequency-head', 'Old objective + dropout'),
                        ('coherent-global-seed42', 'Repaired, global readout'),
                        ('coherent-frequency-seed42', 'Repaired, frequency readout'),
                        ('coherent-frequency-seed7', 'Repaired frequency, seed7')]:
        history = results[name]['training']['history']
        epochs, values = [], []
        for row in history:
            if 'development_auc' in row:
                epochs.append(row['epoch'])
                values.append(row['development_auc'])
        axes[1].plot(epochs, values, marker='.', label=title)
    axes[1].set(xlabel='Epoch', ylabel='Development max-context ROC AUC', ylim=(.45, 1.01),
                title='Deterministic scores, same 210 development recordings')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(REPORT / 'objective-and-learning.png', dpi=150)
    plt.close(fig)


def main():
    results = {name: read_json(REPORT / f'{name}-evaluation.json') for name in REQUIRED}
    for directory in ['trained', 'frozen']:
        for source in sorted((ROOT / directory).rglob('*')):
            if source.is_file() and source.parent.name.endswith('-audit'):
                continue
            if source.is_file():
                preserve(source, ARTIFACT / directory / source.relative_to(ROOT / directory))
    for source in sorted(ROOT.glob('*.tar.gz')):
        preserve(source, ARTIFACT / 'sources' / source.name)
    with tarfile.open(REPORT / 'development-score-traces.tar.gz', 'w:gz') as archive:
        for source in sorted((ROOT / 'traces').glob('*.jsonl.gz')):
            archive.add(source, arcname=source.name)
    with tarfile.open(REPORT / 'instance-selection-audits.tar.gz', 'w:gz') as archive:
        for source in sorted((ROOT / 'trained').glob('*-audit/positive-selection.jsonl.gz')):
            archive.add(source, arcname=source.parent.name+'/'+source.name)
    catalog = read_json('reports/spectral-lab-v1/catalog.json')
    added = ['coverage8', 'medium-balanced-global-head', 'medium-balanced-frequency-head',
             'coherent-global-seed42', 'coherent-frequency-seed42', 'coherent-frequency-seed7']
    for name in added:
        model = load_model(ARTIFACT / 'frozen' / name, True)
        assert model.model_id == results[name]['model_id']
        catalog['models'].append({'name': name, 'bundle': os.path.relpath(ARTIFACT / 'frozen' / name, REPORT)})
    preferred = 'coherent-frequency-seed42'
    catalog.update({'preferred_for_inspection': preferred,
        'defaults': [preferred, 'svm-linear-fft4096', 'yamnet-default'],
        'scope': 'Development model lab. New CNN controls repair input coverage and a demonstrated weak-supervision/dropout loss flaw. Repeated development use does not establish field reliability. The previously assessed YAMNet product default is preserved.',
        'data_summary': 'Same 405 real training files (55 positive, 350 negative; 6 conservative positive groups) and 210 development files (22 positive, 188 negative; 4 positive groups). 4,648 available training contexts, no added independent audio. New models use 256×172 relative log-power inputs; training recipes and seeds are disclosed.'})
    write_json(REPORT / 'catalog.json', catalog)
    portable = {**catalog, 'models': [{**row, 'bundle': os.path.relpath((REPORT / row['bundle']).resolve(), ARTIFACT.resolve())} for row in catalog['models']]}
    write_json(ARTIFACT / 'catalog.json', portable)
    write_json(REPORT / 'experiment-ledger.json', {'scope': catalog['scope'],
        'manifest_sha256': file_hash('runs/real-v2/corpus/manifest-v2.jsonl'),
        'models': {name: {k: row[k] for k in ['model_id', 'parent_model_id', 'architecture', 'training',
            'development', 'source_breakdown', 'batch_sequential_parity', 'raw_trace_sha256']} for name, row in results.items()}})
    lines = ['# CNN foundations: a real training defect and controlled repairs', '',
        'The old positive-max / negative-mean objective rewards input-independent dropout noise. '
        'This is a demonstrated defect in this project’s scratch-CNN implementation. It is not a '
        'diagnosis of every historical EfficientNet result or proof that this is the only remaining problem.', '',
        '## Why the old objective can fail', '',
        'Give both classes exactly the same constant input. A dropout0.1 feature can produce logit−1 '
        'when retained and+1.5 when dropped. A positive bag of8 gets eight chances to receive the high '
        'logit, while the negative loss averages those spikes. Expected balanced loss is0.566, better '
        'than the uninformative0.693, despite no acoustic/class information. With60 contexts it falls '
        'to0.328. Turning dropout off yields0.762. Applying the same maximum-score BCE to both '
        'classes removes this counterexample: identical class distributions cannot beat0.693.', '',
        'This is not just a toy image observation. On405 fixed real training bags, the frozen '
        'balanced model’s old loss is0.367±0.017 across20 dropout draws but0.788 at inference, '
        'without changing its inputs or weights. The larger unbalanced model’s last checkpoint is '
        '0.574±0.081 versus1.559. The [exact witness and checkpoint audit](mil-objective-audit.json) '
        'is separate from accuracy evidence. These fixed-checkpoint comparisons establish mode '
        'sensitivity, not sole attribution of the learned optimization trajectory.', '',
        '## What was trained', '',
        'Data is unchanged:405 real training recordings,55 publisher-positive and350 negative, '
        '1.912 hours in total. Those positives cover only6 conservative provenance groups. '
        'The same210 development files contain22 positives and188 negatives,1.908 hours and '
        '4 positive groups. The final assessment audio was not opened. Publisher recording/flight '
        'presence is weak supervision; individual windows have not acquired verified audible labels.', '',
        'All new arms consume exactly the same cached input: mono16kHz,3seconds,4096-sample Hann '
        'STFT,256-sample hop,256 uniform-Hz triangular bands from50–7600Hz, relative power with '
        'a per-context peak reference, log intensity with an80dB floor. Tensor[256,172,1]. '
        'There is no shared waveform RMS normalization. Fresh extraction matched all7,299 '
        'train/development cached tensors exactly;4,648 are training contexts. Reusing them does '
        'not add independent data.', '',
        'Coverage control changes frequency strides from2/4/4 to2/3/3, preserving1,781 parameters '
        'and identical initial weights. Every input row now has a computation path. The wider '
        'encoder uses32/64/128 filters with5×5,3×3,3×3 kernels, ReLU and frequency/time '
        'strides(2,2),(3,2),(3,2). It has93,313 parameters with a linear global-average head.', '',
        'The matched readout pair shares that encoder and its exact initial convolution weights. '
        'Global readout averages frequency/time then applies Dense473-ReLU→logit:154,675 parameters. '
        'Frequency readout averages time, retains15 frequency positions, flattens1920 features '
        'then Dense32-ReLU→logit:154,689 parameters. They differ by14 parameters, with different '
        'readout function classes. This is one spectrogram input; no3D or multi-view input was added.', '',
        'The repaired pair uses binary cross entropy of the SAME maximum context logit for either '
        'recording class and disables dropout. Both repairs are bundled; the before/after training '
        'comparison does not isolate their separate effects. Per-step sampling is the same cyclic '
        'up-to8 contexts for either class. Each batch has4 recordings per class, no additional '
        'class weights,51 steps per epoch,24 epochs, Adam0.001/clipnorm1 and convolution L2=1e-4. '
        'Validation is deterministic over all cached contexts every2 epochs; select minimum '
        'consistent-max balanced development BCE. Best-at8, selected-best and last checkpoints '
        'are all retained. Fixed seeds reproduce initialization; seed7 is an explicitly '
        '[interim development-triggered replication](replication-trigger.json), not a second test set.', '',
        '## Full development replay', '',
        'All210 development recordings are replayed densely with the same3second/.5second '
        'causal contexts, including disclosed tails/silence. Thresholds and1/2-hit confirmation '
        'are selected under the shared5% negative-recording budget: at most9/188 alarms. '
        'Matching FP counts are calibration constraints, not independent false-alarm measurements. '
        'These recordings have been used repeatedly for development; no fresh assessment is claimed.', '',
        '| Model | Parameters | Selected epoch | TP /22 | FP /188 | Max-score AUC |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, row in results.items():
        metric = row['development']['metrics']
        lines.append(f"| {name} | {row['architecture']['parameters']:,} | {row['training']['selected_epoch']} | {metric['confusion']['tp']} | {metric['confusion']['fp']} | {metric['roc_auc_max_window_score']:.3f} |")
    lines += ['', '## Checkpoint stability on uniform cached contexts', '',
        'This table uses all cached train/development contexts at the declared8-epoch '
        'snapshot and24-epoch horizon, evaluated without dropout. It is a different context '
        'sampling from dense product replay. The last checkpoint is reported even if worse '
        'than the development-selected checkpoint.', '',
        '| Model | Best-at8 selected epoch / dev AUC | Best-at24 selected epoch / dev AUC | Last24 train AUC | Last24 dev AUC |',
        '| --- | --- | --- | ---: | ---: |']
    for name in REQUIRED[2:]:
        at8 = load_model(ROOT / 'trained' / (name+'-at8'))
        final = load_model(ROOT / 'trained' / name)
        history = final.metadata['training']['history']
        short_epoch, best_epoch = at8.metadata['training']['selected_epoch'], final.metadata['training']['selected_epoch']
        short_row = next(r for r in history if r['epoch'] == short_epoch)
        best_row = next(r for r in history if r['epoch'] == best_epoch)
        last = history[-1]
        lines.append(f"| {name} | {short_epoch} / {short_row['development_auc']:.3f} | {best_epoch} / {best_row['development_auc']:.3f} | {last['train_auc']:.3f} | {last['development_auc']:.3f} |")
    lines += ['', 'Preserved controls on the same development files: original linear256 CNN12/22, '
        'non-neural SVM15/22, relative-logistic18/22, YAMNet+logistic20/22; all9/188 negative '
        'files alarmed. The new compact8 repeat reaches11/22; it is a separately identified '
        'training run, not a bitwise recreation of the preserved original12/22 checkpoint. '
        'These are correlated recording counts, not independent flight trials.', '',
        '![Fixed-checkpoint dropout audit and deterministic learning curves](objective-and-learning.png)', '',
        '## Remaining errors by source', '',
        '| Model / source | Positive files detected | Negative files alarmed |',
        '| --- | ---: | ---: |']
    for name in ['medium-balanced-frequency-head', 'coherent-global-seed42', 'coherent-frequency-seed42', 'coherent-frequency-seed7']:
        for source, metric in results[name]['source_breakdown'].items():
            lines.append(f"| {name} / {source} | {metric['confusion']['tp']}/{metric['positive_recordings']} | {metric['confusion']['fp']}/{metric['negative_recordings']} |")
    gain_path = REPORT / 'level-transition/diagnosis.json'
    if gain_path.exists():
        gain = read_json(gain_path)
        lines += ['', '## Return to the quieter recording and specific contexts', '',
            'The same Svanstrom DRONE_001 waveform from4–7seconds was uniformly attenuated '
            'or attenuated only in its second half, with a10ms taper. These are controlled '
            'diagnostic changes to one existing training recording, not extra training audio '
            'or independent accuracy trials. Compare each model to itself; scores are not '
            'calibrated probabilities.', '',
            '| Model | Original score | Uniform×0.25 | Second-half×0.25 |',
            '| --- | ---: | ---: | ---: |']
        for name, item in gain['models'].items():
            controls = item['controls']
            values = [controls[key] for key in ['unchanged_4_to_7s', 'uniform_minus12dB', 'internal_minus12dB_step_at1_5s']]
            lines.append('| '+name+' | '+' | '.join(f'{value:.4f}' for value in values)+' |')
        lines += ['', 'The new frequency models remain stable on this particular internal-step '
            'control, while YAMNet and relative-logistic change sharply. This is one useful '
            'case, not evidence of general gain robustness. The real transition also changes '
            'frequency bands unequally, so it is not solely a uniform volume change.', '',
            '![Actual recording levels and old/new model timelines](level-transition/recording-level-and-scores.png)', '',
            '![Same five real contexts, inputs and final-layer positive Grad-CAM](specific-context-attributions.png)', '',
            'The five-context comparison includes the Svanstrom recording before/across '
            'the drop, a Batear FPV recording, helicopter and rural background. The two '
            'frequency models separate these particular examples more strongly than the '
            'tiny CNN, whose scores remain near0.42. Seed7 often concentrates around '
            'the800Hz region in the drone examples, while seed42 has much broader maps. '
            'Seed42 also shows a bright normalized positive map for the helicopter despite '
            'a drone score near0.007: positive-only, independently normalized Grad-CAM is '
            'not the model score. Spatial gradient averaging can obscure a frequency-sensitive '
            'head. These plots do not establish that harmonic relationships were learned '
            'or that highlighted regions are causal. Audible event timing remains unverified.', '']
    lines += ['', '## What this does and does not establish', '',
        'The repaired frequency readout detects16/22 positives at seed42 and17/22 at seed7, '
        'versus11/22 for its matched repaired global readout. Under the older loss, the '
        'frequency/global pair reaches18/22 versus12/22. Preserving frequency positions '
        'helps in both comparisons. The objective repair does NOT improve this operating '
        'point over the older frequency model: its value is removing a proven training '
        'loophole, not a demonstrated standalone alarm-accuracy gain. Both new seeds improve '
        'ranking over the tiny baseline, but two seeds cannot characterize run-to-run uncertainty.', '',
        'The objective defect is concrete; correcting missing frequency paths alone was insufficient. '
        'Capacity, readout structure, weak target quality and collection shortcuts remain distinct '
        'questions. A source-only rule already predicts199/210 development labels. Better CNN '
        'scores here can still reflect acquisition conditions or other cues. Grad-CAM is a '
        'sensitivity diagnostic, not an audible-label or causality check.', '',
        'The next decisive data task is matched, independently reviewed audible-on/off intervals '
        'with entire capture sessions held out, followed by a fresh grouped assessment. Increasing '
        'crop counts or repeatedly tuning this development set cannot supply that evidence. '
        'The old YAMNet product default and its consumed assessment remain separately identified. '
        'The browser lab now offers the new controls beside it, with identical-context Grad-CAM, '
        'training histories, retained multi-recording results and original audio playback levels.', '',
        '## Reproduce and inspect', '',
        'Plans, training histories, selected/last weights, exact execution source snapshots, '
        'instance-selection audits, per-recording source outcomes and compressed full score '
        'traces are retained. Raw traces identify trained parents and use placeholder0.5 '
        'decisions; evaluation reports reapply each frozen calibrated decision. No raw audio '
        'or feature cache is copied into the artifact package.', '',
        '`scripts/experiment_cnn_foundations.py`, `scripts/experiment_balanced_cnn.py`, '
        '`scripts/experiment_cnn_followup.py` and `scripts/experiment_coherent_cnn.py` execute '
        'the declared controls. `scripts/evaluate_cnn_foundations.py` shares dense development '
        'frontends and verifies batch/sequential score parity before calibration. '
        '`scripts/package_cnn_foundations.py` creates this report and the portable catalog.', '']
    (REPORT / 'RESULTS.md').write_text(readable_prose(lines))
    make_figures(results)
    write_json(ARTIFACT / 'source.json', revision())
    with tarfile.open(ARTIFACT / 'product-source.tar.gz', 'w:gz') as archive:
        for directory in ['src', 'scripts', 'tests', 'docs', '.github', 'web/src']:
            for source in sorted(Path(directory).rglob('*')):
                if source.is_file() and '__pycache__' not in source.parts:
                    archive.add(source, arcname=str(source))
        for name in ['pyproject.toml', 'uv.lock', 'README.md', '.python-version',
                     'web/package.json', 'web/package-lock.json', 'web/build.mjs',
                     'web/serve.cjs', 'web/hash-assets.cjs']:
            archive.add(name, arcname=name)
    sources = sorted(path for path in ARTIFACT.rglob('*') if path.is_file() and path.name != 'SHA256SUMS')
    (ARTIFACT / 'SHA256SUMS').write_text(''.join(f'{file_hash(path)}  {path.relative_to(ARTIFACT)}\n' for path in sources))
    print('Preserved controls,11 full development replays,source snapshots and17-model browser catalog.')


if __name__ == '__main__':
    main()
