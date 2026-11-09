#!/usr/bin/env python3
"""Public case notes and diagnosis, using only redistributable example plots."""

import html
import re
import shutil
from pathlib import Path

from drone_detector.common import read_json


def markdown_table(text):
    lines = text.strip().splitlines()
    header = [html.escape(cell.strip()) for cell in lines[0].strip('|').split('|')]
    rows = [[html.escape(cell.strip()) for cell in line.strip('|').split('|')] for line in lines[2:]]
    return '<div class="scroll"><table><tr>' + ''.join(f'<th>{cell}</th>' for cell in header) + '</tr>' + ''.join(
        '<tr>'+''.join(f'<td>{cell}</td>' for cell in row)+'</tr>' for row in rows) + '</table></div>'


def render_markdown(text):
    # Deliberately small renderer for local, trusted generated reports only.
    blocks = []
    for block in text.split('\n\n'):
        if block.startswith('!['):
            continue  # Overview includes non-redistributed D3 figures; use licensed cases below.
        if block.startswith('|'):
            blocks.append(markdown_table(block))
            continue
        block = html.escape(block)
        block = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', block)
        block = re.sub(r'`([^`]+)`', r'<code>\1</code>', block)
        block = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', block)
        if block.startswith('# '):
            blocks.append('<h1>'+block[2:]+'</h1>')
        elif block.startswith('## '):
            blocks.append('<h2>'+block[3:]+'</h2>')
        elif block.startswith('- '):
            blocks.append('<ul>'+''.join('<li>'+line[2:]+'</li>' for line in block.splitlines())+'</ul>')
        else:
            blocks.append('<p>'+block+'</p>')
    return '\n'.join(blocks)


root = Path('web/public')
cases = read_json('reports/real-observations-v1/cases.json')
body = '<p><a href="/">← Open the model lab and playable examples</a></p>'
body += '<p><a href="#cnn-foundations">CNN training repairs</a> · <a href="#level-transition">Quieter-transition experiment</a> · <a href="#next-steps">Competing explanations and next steps</a></p>'
foundations = Path('reports/cnn-foundations-v1/RESULTS.md')
if foundations.exists():
    body += '<div id="cnn-foundations"></div>'+render_markdown(foundations.read_text())
    shutil.copyfile(foundations.parent / 'objective-and-learning.png', root / 'examples/objective-and-learning.png')
    shutil.copyfile(foundations.parent / 'mil-objective-audit.json', root / 'mil-objective-audit.json')
    body += '<img loading="lazy" src="examples/objective-and-learning.png" alt="Fixed-checkpoint dropout loss audit and deterministic CNN learning curves"><p><a href="mil-objective-audit.json">Exact objective counterexample and real-input checkpoint measurements</a></p>'
    attribution = foundations.parent / 'specific-context-attributions.png'
    if attribution.exists():
        shutil.copyfile(attribution, root / 'examples/foundation-attributions.png')
        body += '<h2>Specific existing contexts: input and new model attributions</h2><p>Same input and interval in each row. Final-convolution positive drone-logit Grad-CAM is normalized within each map; map brightness is not comparable between models. These are selected training/development examples, not additional independent trials or verified audible labels.</p><img loading="lazy" src="examples/foundation-attributions.png" alt="Five real contexts, identical inputs, old CNN and new global/frequency readout attributions">'
    gain = foundations.parent / 'level-transition/diagnosis.json'
    if gain.exists():
        measured = read_json(gain)
        body += '<h2>Return to the quieter transition with the new CNNs</h2><p>These controls alter the same real waveform, not its label. The internal step attenuates only the second half of the context; it changes temporal energy distribution. Uniform attenuation preserves the relative spectral input.</p><div class="scroll"><table><tr><th>Model</th><th>Original4–7s score</th><th>Uniform×0.25 score</th><th>Internal half-context×0.25 score</th></tr>'
        for name, row in measured['models'].items():
            controls = row['controls']
            body += '<tr><td>'+html.escape(name)+'</td>'+''.join('<td>'+f'{controls[key]:.4f}'+'</td>' for key in ['unchanged_4_to_7s', 'uniform_minus12dB', 'internal_minus12dB_step_at1_5s'])+'</tr>'
        body += '</table></div>'
        shutil.copyfile(gain.parent / 'recording-level-and-scores.png', root / 'examples/foundation-level-transition.png')
        shutil.copyfile(gain, root / 'foundation-level-transition.json')
        body += '<img loading="lazy" src="examples/foundation-level-transition.png" alt="Recording level and new CNN score responses across the actual quieter transition"><p><a href="foundation-level-transition.json">Download measurements, counterfactual controls and original timelines</a></p>'
body += '<h2>Earlier baseline studies and source observations</h2>'
body += render_markdown(Path('reports/real-observations-v1/OBSERVATIONS.md').read_text())
body += '<h2>Inspect the same input and model attribution</h2>'
for case in cases['cases']:
    if 'public_example' not in case:
        continue
    image = root / 'examples' / case['figure']
    shutil.copyfile(Path('reports/real-observations-v1') / case['figure'], image)
    body += '<h3>'+html.escape(case['id'])+'</h3>'
    body += '<audio controls preload="none" src="'+html.escape(case['public_example'])+'"></audio>'
    body += '<p>'+html.escape(case['license'])+' · <a href="'+html.escape(case['provenance']['url'])+'">Original source</a></p>'
    body += '<img loading="lazy" alt="Actual spectrogram, model input and Grad-CAM for '+html.escape(case['id'])+'" src="examples/'+html.escape(case['figure'])+'">'
body += render_markdown(Path('reports/cnn-diagnosis-v1/RESULTS.md').read_text())
body += '<div id="level-transition"></div>' + render_markdown(Path('reports/level-transition-v1/RESULTS.md').read_text())
shutil.copyfile('reports/level-transition-v1/recording-level-and-scores.png', root / 'examples/level-transition.png')
shutil.copyfile('reports/level-transition-v1/diagnosis.json', root / 'level-transition-diagnosis.json')
body += '<img loading="lazy" src="examples/level-transition.png" alt="Actual recording RMS and model score timelines"><p><a href="level-transition-diagnosis.json">Download exact level-transition measurements and all model scores</a></p>'
body += '<div id="next-steps"></div>' + render_markdown(Path('docs/cnn-next-steps.md').read_text())
body += '<p><a href="/THIRD_PARTY.md">Source and license notices</a></p>'
style = '''body{font:16px system-ui;max-width:1150px;margin:auto;padding:24px;background:#0c171d;color:#e8f2f3;line-height:1.6}
a{color:#8de4c5}table{border-collapse:collapse;font-size:14px}td,th{padding:10px;border:1px solid #38515c;vertical-align:top}
.scroll{overflow:auto}img{max-width:100%;background:white}audio{width:100%}code{overflow-wrap:anywhere}h1,h2{line-height:1.25;margin-top:40px}'''
(root / 'diagnosis.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Real examples and CNN diagnosis</title><style>'+style+'</style>'+body+'</html>')
