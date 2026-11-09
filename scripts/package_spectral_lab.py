#!/usr/bin/env python3
"""Package the complete spectral sweep, diagnostic calibration and local lab."""

import os
import shutil
import tarfile
from pathlib import Path

from drone_detector.common import file_hash, read_json, write_json
from drone_detector.model import load_model


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ValueError(f'Refusing to replace a different packaged file: {destination}')
    else:
        shutil.copyfile(source, destination)


def copy_tree(source, destination):
    for file in sorted(source.rglob('*')):
        if file.is_file():
            copy_file(file, destination / file.relative_to(source))


def main():
    root = Path('runs/spectral-lab-v1')
    artifacts, reports = Path('artifacts/spectral-lab-v1'), Path('reports/spectral-lab-v1')
    coarse = read_json(root / 'results.json')
    calibrated = read_json(root / 'calibrated-results.json')
    expected = set(coarse['models']) | {'frequency-context-fft4096', 'linear256-fft4096'}
    if len(coarse['models']) != 7 or expected != set(calibrated['models']):
        raise ValueError('Finish the six-arm sweep, SVM, both follow-up controls and their calibration first')
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    for name in ['plan.json', 'results.json', 'calibration-plan.json', 'calibrated-results.json',
                 'sparse-projection-verification.json', 'frequency-context-plan.json', 'matched-frequency-axis-plan.json', 'freeze-status-correction.json']:
        copy_file(root / name, reports / name)
    copy_file(root / 'training-source.tar.gz', artifacts / 'sources' / 'initial-training-source.tar.gz')
    copy_file(root / 'calibration-source.tar.gz', artifacts / 'sources' / 'calibration-source.tar.gz')
    copy_file(root / 'frequency-context-source.tar.gz', artifacts / 'sources' / 'frequency-context-source.tar.gz')
    copy_file(root / 'matched-frequency-axis-source.tar.gz', artifacts / 'sources' / 'matched-frequency-axis-source.tar.gz')
    copy_file(root / 'freeze-status-source.tar.gz', artifacts / 'sources' / 'freeze-status-source.tar.gz')
    copy_tree(root / 'source-revisions', artifacts / 'sources' / 'revisions')
    copy_tree(root / 'calibration-revisions', artifacts / 'sources' / 'calibration-revisions')
    copy_tree(root / 'models', artifacts / 'coarse-grid')
    copy_tree(root / 'calibrated', artifacts / 'models')
    copy_tree(root / 'followup-models', artifacts / 'followup-trained')
    copy_file(Path('artifacts/real-v2/YAMNet-LICENSE.txt'), artifacts / 'YAMNet-LICENSE.txt')
    history = {}
    for name, row in calibrated['models'].items():
        model = load_model(artifacts / 'models' / name, True)
        assert model.model_id == row['model_id']
        history[name] = {k: v for k, v in model.metadata['training'].items() if k != 'records'}
    write_json(reports / 'training-history.json', history)
    with tarfile.open(reports / 'development-score-traces.tar.gz', 'w:gz') as archive:
        for file in sorted((root / 'calibrated-traces').glob('*.gz')):
            archive.add(file, arcname=file.name)
    neural = {name: row for name, row in calibrated['models'].items() if row['family'] == 'spectrogram_cnn'}
    preferred = max(neural, key=lambda name: (
        neural[name]['development']['metrics']['recording_recall'],
        -neural[name]['development']['metrics']['negative_recording_false_positive_fraction'],
        neural[name]['development']['metrics']['roc_auc_max_window_score']))
    catalog = {'scope': 'Development model lab. New spectrogram CNNs/SVM have no fresh independent assessment. The previously assessed YAMNet default is preserved.',
        'data_summary': '405 real training files (55 positive, 350 negative), 210 development files (22 positive, 188 negative). 4,648 sampled contexts available; the selected checkpoint can have seen fewer. Four positive development provenance groups; counts are not independent trials.',
        'preferred_for_inspection': preferred,
        'defaults': [preferred, 'svm-linear-fft4096', 'yamnet-default'],
        'models': []}
    order = [preferred] + [name for name in calibrated['models'] if name != preferred]
    for name in order:
        catalog['models'].append({'name': name, 'bundle': os.path.relpath(artifacts / 'models' / name, reports)})
    catalog['models'] += [{'name': 'yamnet-default', 'bundle': '../../artifacts/real-v2/candidate'},
                          {'name': 'relative-logistic', 'bundle': '../../artifacts/real-v2/baseline'}]
    write_json(reports / 'catalog.json', catalog)
    portable_catalog = {**catalog, 'models': [{**row, 'bundle': os.path.relpath(
        (reports / row['bundle']).resolve(), artifacts.resolve())} for row in catalog['models']]}
    write_json(artifacts / 'catalog.json', portable_catalog)
    lines = ['# Spectrogram and non-neural development experiments', '',
        'All results below use the same 405 real training files and 210 development files. '
        'The 22 positive development files cover four provenance groups. These are development results; '
        'no new final assessment or replacement of the assessed default is claimed.', '',
        'The [versioned manifest and splits](../real-v2/manifest-snapshot.jsonl) and '
        '[primary source versions](../real-v2/SOURCES.md) are shared with the preserved v2 protocol.', '',
        '## What changed', '',
        'Six CNNs share the same 1,781-parameter architecture: 6-filter 5×5 convolution, '
        '8-filter 3×3 convolution, 16-filter 3×3 convolution, ReLU, global average pooling, '
        'dropout 0.1 and a scalar linear logit. Frequency/time strides are (2,2), (4,2), (4,2). '
        'All ran eight epochs with seed 42, Adam 0.001, L2 1e-4, clipped gradients and the same '
        'balanced recording-bag objective. Positive bags use their maximum sampled logit; '
        'negative bags penalize all sampled windows. The best balanced development bag-loss '
        'checkpoint was retained. Only one seed was explored.', '',
        'At most 60 uniform contexts per training file supply 4,648 candidate inputs, including '
        '1,262 contexts from positive bags and 3,386 from negatives. All were visited by epoch 8; '
        'an earlier selected checkpoint saw fewer. Per-model history records that exact count. '
        'Publisher-positive windows are not all assigned positive labels.', '',
        '| Arm | Actual frequency × time × channel tensor | Hann window | Intensity |',
        '| --- | --- | ---: | --- |']
    plan = read_json(root / 'plan.json')
    for name, case in plan['cases'].items():
        lines.append(f"| {name} | {case['shape']} | {case['features']['n_fft']/16:.0f} ms | {case['features']['intensity']} |")
    lines.append('| linear256-fft4096 (matched-axis follow-up) | [256, 172, 1] | 256 ms | log_power |')
    lines += ['', 'FFT spacing is 15.625 Hz for 1,024 samples and 3.90625 Hz for 4,096 samples at 16 kHz. '
        'The actual Hann window is lengthened; this is not zero-padding or image enlargement. '
        'The 256-sample STFT hop is 16 ms. Every model sees 50–7,600 Hz in the same 3-second context '
        'with a 0.5-second detection step. The longer analysis window trades temporal discrimination '
        'for frequency discrimination. Mel filtering averages bins; logarithmic intensity is a separate '
        'operation. All inputs use per-context relative power, with an 80 dB floor for log inputs. '
        'Absolute capture level and frequencies above 8 kHz are not studied here.', '',
        'The non-neural RBF SVM uses the high-resolution linear-frequency/log-power input, '
        'then temporal mean/std at each of 1,933 frequency bins: 3,866 descriptors. '
        'Weighted standardization is fitted on training only. Four multiple-instance updates '
        'select positive contexts and retain all sampled negatives. C=1, gamma=scale. '
        'Its support vectors, dual coefficients, intercept and normalization are numeric NPZ arrays, '
        'and its portable scores were checked against sklearn.', '',
        'One subsequent architecture probe adds a frequency-only 16-filter 5×1 convolution '
        'to the high-resolution linear/log-power CNN: 3,077 parameters. The frequency receptive '
        'field expands from 25 bins (about 98 Hz) to 153 bins (about 598 Hz), while temporal '
        'receptive field stays fixed. This addresses the fact that denser inputs shrink the '
        'physical span of fixed pixel kernels. Its separate follow-up plan, training history '
        'and source snapshot are retained; it was motivated after the original sweep, not '
        'part of its initial planned comparison.', '',
        'A separate matched-axis control uses 256 area-normalized triangular bands uniformly '
        'spaced in Hz. It matches the mel256 tensor, initialization, CNN capacity, FFT, amplitude '
        'rule and training recipe. This avoids attributing the full 1,933-bin FFT advantage '
        'solely to frequency warping when it also retained more bins. Its own follow-up plan '
        'and source snapshot distinguish this diagnostic extension from the original sweep.', '',
        'Package validation caught a trained status inherited by the two calibrated follow-ups. '
        'The corrected packages explicitly mark frozen decisions; numerical weights, decisions '
        'and results did not change. The [correction record](freeze-status-correction.json) retains '
        'old/new identities and equality checks.', '',
        '## Coarse grid and additional calibration', '',
        'The initial predefined threshold grid produced zero detections for these small CNNs at '
        'the 5% negative-file budget, despite score ranking above chance. Those frozen packages '
        'and results are retained under `coarse-grid/`. A separate, explicitly additional '
        'development calibration used negative-recording activation statistics to set a threshold '
        'for one- and two-hit confirmation, with a 1e-5 numerical margin. Neural weights did not change. '
        'Its development FPR is constrained by construction, and does not establish future FPR.', '',
        '| Model | Coarse-grid TP / 22 | Coarse-grid FP / 188 | Calibrated TP / 22 | Calibrated FP / 188 | Max-score ROC AUC |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, row in calibrated['models'].items():
        old = coarse['models'].get(name, {}).get('development', {}).get('metrics')
        new = row['development']['metrics']
        old_tp = old['confusion']['tp'] if old else 'n/a'
        old_fp = old['confusion']['fp'] if old else 'n/a'
        lines.append(f"| {name} | {old_tp} | {old_fp} | {new['confusion']['tp']} | {new['confusion']['fp']} | {new['roc_auc_max_window_score']:.3f} |")
    lines += ['', 'Existing controls on the same development recordings: frozen YAMNet + logistic regression '
        '**20/22 positive files detected, 9/188 negative files alarmed**; non-neural relative-mel summaries '
        '+ logistic regression **18/22, 9/188**. Those controls have separate previously frozen thresholds. '
        'These comparisons do not isolate neural networks as the sole cause: representations, pretraining, '
        'capacity and optimization differ.', '',
        'Linear high-resolution log-power improved ranking over the coarse 64-mel CNN in this one-seed '
        'pilot. Linear power had worse ranking than log power at the same high-resolution frequency grid. '
        'This supports preserving frequency detail for further study while retaining log intensity. '
        'It does not establish that every high-resolution tensor or non-neural method is better. '
        'At matched 256×172 dimensions, uniform-Hz bands detect 12/22 versus 10/22 for mel, '
        'both at 9/188 development false positives; their ROC AUCs are 0.835 and 0.848. '
        'The full 1,933-bin linear FFT detects 11/22 with higher ROC AUC 0.867. '
        'More bins therefore did not monotonically improve recall at this operating point. '
        'The frequency-context extension still detects 11/22. '
        'The small CNN budget, weak labels, few positive groups and changed physical receptive fields '
        'limit the conclusion. No expensive architecture expansion or default promotion follows.', '',
        '## Source results and failures', '']
    for name in [preferred, 'svm-linear-fft4096']:
        lines += [f'### {name}', '', '| Source | Positive files detected | Negative files alarmed |', '| --- | ---: | ---: |']
        for source, metric in calibrated['models'][name]['source_breakdown'].items():
            lines.append(f"| {source} | {metric['confusion']['tp']}/{metric['positive_recordings']} | {metric['confusion']['fp']}/{metric['negative_recordings']} |")
        lines += ['', f"Group-bootstrap interval: `{calibrated['models'][name]['group_uncertainty']}`", '']
    lines += ['## Interface and reproducibility', '',
        'Run `bash scripts/start_lab.sh`. The local lab accepts audio, compares selected models, '
        'plays/seeks score timelines, and shows the selected CNN input beside genuine Grad-CAM. '
        'Choose the logit class, convolution layer, context and frequency zoom. Training histories, '
        'selected epochs, actual tensor dimensions and development errors are visible. '
        'Full comparison and explanation data can be downloaded as JSON. Grad-CAM is available for '
        'the trained spectrogram CNNs; it is not fabricated for the SVM or opaque YAMNet SavedModel.', '',
        'Attribution uses gradients of the selected CNN logit, not the input amplitude as a proxy. '
        'Its native spatial grid is disclosed; interpolating that grid adds no explanatory detail. '
        'A heatmap alone does not prove a causal or useful acoustic cue. Zero maps remain zero. '
        'The browser demonstration uses the fixed existing Batear development recording '
        '`batear-20260415_192501` (9.078 seconds); it supplies software evidence, not an additional trial.', '',
        'Complete coarse and calibrated models, source snapshots, plans, data identity, per-source '
        'outcomes, every compressed development score trace and batch/sequential parity checks are retained. '
        'Raw audio and multi-gigabyte feature caches remain in ignored `runs/`. '
        'The initial slow dense mel projection was replaced before remaining arms with an equivalent '
        'sparse multiplication; 16 checked contexts matched exactly. Completed arms were preserved.', '',
        'Archived raw score traces retain their parent model identity and original/placeholder '
        'alarm state. The calibrated decisions and resulting recording alarm intervals are in '
        '`calibrated-results.json`; apply those decisions to the raw scores to reproduce outcomes.', '',
        'The original v2 assessment remains 100/133 publisher-positive files detected and '
        '126/1,608 negative files alarmed for the unchanged YAMNet default. All later spectral '
        'experiments are development work. Event recall, true false alarms/hour, acoustic range, '
        'physical capture delay and live microphone operation remain unvalidated.', '']
    (reports / 'RESULTS.md').write_text('\n'.join(lines))
    ledger = {'scope': plan['scope'], 'manifest_sha256': plan['manifest_sha256'],
              'original_plan': 'plan.json', 'additional_calibration_plan': 'calibration-plan.json',
              'models': {name: {'training': history[name], 'development': row['development'],
                               'parent_model_id': row['parent_model_id'], 'model_id': row['model_id'],
                               'source_breakdown': row['source_breakdown']} for name, row in calibrated['models'].items()}}
    write_json(reports / 'experiment-ledger.json', ledger)
    print('Packaged the seven-arm sweep, both follow-ups, calibration stages and model lab.', flush=True)


if __name__ == '__main__':
    main()
