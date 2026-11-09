#!/usr/bin/env python3
"""Same-context attribution comparisons on existing licensed train/dev examples."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, write_json
from drone_detector.model import load_model
from drone_detector.spectrogram import axes


def main():
    report = Path('reports/cnn-foundations-v1')
    root = Path('runs/cnn-foundations-v1/frozen')
    names = ['linear256-fft4096', 'coherent-global-seed42', 'coherent-frequency-seed42', 'coherent-frequency-seed7']
    models = {name: load_model(Path('artifacts/spectral-lab-v1/models')/name if name.startswith('linear') else root/name, True) for name in names}
    cases = [('svanstrom-DRONE_001', 4.), ('svanstrom-DRONE_001', 6.5),
             ('batear-20260415_192501', 0.), ('svanstrom-HELICOPTER_001', 3.),
             ('batear-20230701_054200', 0.)]
    examples = {row['id']: row for row in read_json('web/public/examples/catalog.json')['examples']}
    fig, panels = plt.subplots(len(cases), 5, figsize=(18, 12), constrained_layout=True)
    output = {'scope': 'Five deliberately selected contexts from existing licensed TRAIN/DEVELOPMENT examples. Visual/software diagnostics, no new independent acoustic trials or verified audible labels.', 'cases': []}
    titles = ['Shared model input', 'Old tiny CNN', 'Repaired global readout', 'Repaired frequency, seed42', 'Repaired frequency, seed7']
    for row_number, (identity, start) in enumerate(cases):
        source = Path('runs/real-v2/corpus/audio') / (identity+'.wav')
        wave, _ = decode(source)
        excerpt = wave[round(start*16000):round((start+3)*16000)]
        assert len(excerpt) == 48000
        case = {'id': identity, 'start_seconds': start, 'end_seconds': start+3,
            'original_sha256': file_hash(source), 'publisher_label': examples[identity]['publisher_label'],
            'license': examples[identity]['license'], 'models': {}}
        for column, (name, model) in enumerate(models.items(), 1):
            values = model.gradcam(excerpt, 'drone', 'features')
            frequency, times, _ = axes(model.config)
            extent = [start+times[0], start+times[-1], frequency[0], frequency[-1]]
            case['models'][name] = {'model_id': model.model_id, 'score': values['classifier_score'],
                'coarse_shape': values['coarse_shape'], 'positive_map': values['has_positive_attribution']}
            if column == 1:
                panels[row_number, 0].imshow(values['spectrogram'], origin='lower', extent=extent,
                    aspect='auto', cmap='gray', vmin=0, vmax=1)
                panels[row_number, 0].set_ylabel(identity+'\nHz', fontsize=9)
            panels[row_number, column].imshow(values['heatmap'], origin='lower', extent=extent,
                aspect='auto', cmap='inferno', vmin=0, vmax=1)
            panels[row_number, column].text(.02, .92, f"drone score {values['classifier_score']:.3f}",
                transform=panels[row_number, column].transAxes, color='cyan', fontsize=10,
                bbox={'facecolor': 'black', 'alpha': .6, 'edgecolor': 'none'})
        for column, panel in enumerate(panels[row_number]):
            panel.set_ylim(50, 3000)
            panel.set_xlabel('Recording time (s)')
            if row_number == 0:
                panel.set_title(titles[column], fontsize=11)
        output['cases'].append(case)
    fig.suptitle('Same input and interval; positive drone-logit Grad-CAM, final convolution\nEach map normalized independently; interpolation adds no detail. Existing training/development examples only.', fontsize=13)
    fig.savefig(report / 'specific-context-attributions.png', dpi=145)
    plt.close(fig)
    write_json(report / 'specific-context-attributions.json', output)
    print('Compared20 model/context decisions on5 existing contexts; no independent accuracy claim.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
