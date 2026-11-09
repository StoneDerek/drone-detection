#!/usr/bin/env python3
"""Reuse the tested local comparison UI with browser-local inference."""
from pathlib import Path

from drone_detector.lab import PAGE


root = Path('web')
before, script = PAGE.split('<script>', 1)
script, after = script.split('</script>', 1)
before = before.replace('accept="audio/*,.wav,.flac,.ogg"', 'accept=".wav,audio/wav"')
before = before.replace('WAV, FLAC or OGG', 'PCM or float WAV')
before = before.replace('WAV / FLAC / OGG · up to 64 MB and 2 minutes. Audio is processed locally; server temporary files are removed after each request.',
    'PCM or float WAV · up to 64 MB and 2 minutes. Inference runs on your device; your audio is never uploaded. Processing speed depends on your device.')
before = before.replace('<th>Quality flags</th>', '<th>Quality flags (Python app only)</th>')
before = before.replace('Local model comparison', 'Audio model comparison')
before = before.replace('<section class="card">', '''<section class="card">
<h2>Work through real examples</h2>
<div class="bar"><select id="example"><option>Loading examples…</option></select><button id="openexample">Open example</button></div>
<p id="exampleinfo" class="mini">Publisher classes describe recordings; audible event boundaries have not been independently verified.</p>
</section><section class="card">''', 1)
before += '''<section class="card"><h2>Listening notes</h2>
<p>Listen and inspect a specific interval. Save an audible-drone, background, or uncertain judgment with your reasons. These draft notes stay on your device and are not used for training.</p>
<div class="bar"><label>Start (s) <input id="reviewstart" type="number" value="0" min="0" step="0.01"></label>
<label>End (s) <input id="reviewend" type="number" value="3" min="0" step="0.01"></label>
<select id="reviewlabel"><option value="uncertain">Uncertain</option><option value="audible_drone">Audible drone</option><option value="background">Background / other sound</option></select></div>
<p><textarea id="reviewnote" rows="2" style="width:95%;font:inherit" placeholder="What did you hear or see? What else could explain it?"></textarea></p>
<div class="bar"><button id="savereview">Save interval note</button><button id="downloadreview" disabled>Download notes</button><span id="reviewstatus"></span></div>
<ol id="reviewitems"></ol></section>'''
notice = '''<div class="notice">Runs on your device. Audio stays in your browser.
Choose PCM or float WAV, up to 2 minutes / 64 MB. The first comparison downloads model weights.
Per-window clipping and low-frequency flags are available in the Python app; this browser comparison does not measure them.
These are experimental detectors; development results do not establish field reliability.
Grad-CAM shows sensitivity of a selected CNN logit, not proof of a useful acoustic cue.
<a href="diagnosis.html">Read the formulation and training diagnosis</a>.</div>'''
before = before.replace('<body>', '<body>'+notice) if '<body>' in before else before.replace('<h1>',notice+'<h1>',1)
(root / 'public').mkdir(exist_ok=True)
(root / 'public/index.html').write_text(before+'<script type="module" src="/app.js"></script>'+after)
(root / 'src/ui.js').write_text(script)
(root / 'public/_headers').write_text('''/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
  Permissions-Policy: microphone=(), camera=(), geolocation=()
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src blob: 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'
''')
