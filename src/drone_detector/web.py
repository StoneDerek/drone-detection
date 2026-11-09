"""Local file-analysis product. No external service or notebook state required."""

import io
import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import soundfile as sf
from threadpoolctl import threadpool_limits

from .common import file_hash, read_json
from .data import load_manifest
from .model import load_model
from .stream import alarm_intervals, infer_file

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Drone audio · Inspect a recording</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0c171d;color:#edf6f5}
body{max-width:1040px;margin:0 auto;padding:32px}header{display:flex;align-items:center;justify-content:space-between;gap:18px}
h1{font-size:32px;font-weight:650;letter-spacing:-1px;margin:8px 0}p{line-height:1.55;color:#b4c8ce}
.tag{font-size:12px;text-transform:uppercase;letter-spacing:2px;color:#71d7bb}.card{background:#15272f;border:1px solid #2a414a;border-radius:16px;padding:24px;margin:22px 0}
.notice{border-left:3px solid #e8b569;padding:10px 18px;color:#f6d9a7;background:#322a20;font-size:14px}
.upload{display:block;padding:28px;border:1px dashed #69918f;border-radius:10px;cursor:pointer;text-align:center;color:#9eead2}
input{display:block;margin:16px auto;max-width:100%}button,select{font:inherit;background:#264a4e;color:white;border:1px solid #4f7a7a;border-radius:8px;padding:11px 18px;cursor:pointer}
button:disabled{opacity:.45;cursor:wait}.bar{display:flex;flex-wrap:wrap;align-items:center;gap:18px;margin:18px 0}.bar p{margin:0}
.state{font-size:25px;font-weight:650;color:#80e5c1}.state.alarm{color:#ffb280}.state.review{color:#e8c88d}audio{width:100%;margin:12px 0}
canvas{width:100%;height:260px;background:#0f2028;border-radius:8px}code{font-size:12px;overflow-wrap:anywhere;color:#b7d5d8}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:9px;border-bottom:1px solid #2a414a}
.scroll{max-height:240px;overflow:auto}#error{color:#ffb7a2}footer{font-size:13px;color:#8ca7af}
@media(max-width:650px){body{padding:16px}.card{padding:16px}h1{font-size:26px}header{display:block}}
</style><header><div><span class="tag">Acoustic presence detection</span><h1>Inspect a recording</h1>
<p>Upload audio. Review the score timeline. Replay every decision.</p></div><span class="tag">Local analysis</span></header>
<p class="notice" id="scope">Loading model details…</p>
<section class="card"><label class="upload" for="file">Choose a recording<input id="file" type="file" accept=".wav,.flac,.ogg,.mp3,audio/*"></label>
<div class="bar"><button id="analyze" disabled>Analyze recording</button><p id="filename">WAV / FLAC / OGG · up to 64 MB and 10 minutes</p></div>
<label id="examplelabel" for="examples">Or inspect a disclosed benchmark input</label> <select id="examples"><option value="">Choose an example…</option></select>
<p id="error" role="alert"></p><p id="progress" role="status"></p></section>
<section class="card" id="result" hidden><div class="bar"><span class="state" id="state">Waiting for audio context</span><span id="score"></span><span id="time"></span></div>
<audio id="player" controls></audio><canvas id="chart" width="1000" height="280" aria-label="Time and classifier score with threshold" role="img"></canvas>
<p id="summary"></p><p>White: classifier score. Orange: alarm threshold. A score is available at the end of its audio context.
The alarm uses consecutive scores and hysteresis. Scores are not calibrated probabilities.</p>
<p id="qualityinfo"></p>
<button id="download">Download scores (JSON)</button><div class="scroll"><table><thead><tr><th>Time (s)</th><th>Score</th><th>Alarm</th><th>Input handling</th><th>Audio quality</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="card"><span class="tag">Model & operating limits</span><p id="details"></p><code id="identity"></code></section>
<footer>Recorded replay. This interface does not claim live microphone capture, distance coverage or a validated field false-alarm rate.</footer>
<script>
const el=id=>document.getElementById(id);let model,selected,result,objectUrl;
async function load(){try{model=await(await fetch('/api/model')).json();
el('identity').textContent=model.model_id;el('scope').textContent=model.scope;el('examples').hidden=!model.examples.length;el('examplelabel').hidden=!model.examples.length;
el('details').textContent='Mono '+model.features.sample_rate+' Hz · '+model.features.window_seconds+' s rolling context · '+model.features.hop_seconds+' s step. '+model.limitations;
el('qualityinfo').textContent=model.quality_advisories+' Quality flags advise review and do not suppress an alarm.';
for(const [i,x] of model.examples.entries()){const option=document.createElement('option');option.value=i;option.textContent=(x.title||x.id)+' · publisher: '+x.label+' · '+x.selection_role;el('examples').append(option)}}catch(e){el('error').textContent='Unable to load the model: '+e.message}}
function pick(blob,name){selected=blob;el('filename').textContent=name;el('analyze').disabled=false;el('result').hidden=true;el('error').textContent='';el('progress').textContent='Ready to analyze.';result=null;
if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=URL.createObjectURL(blob);el('player').src=objectUrl}
el('file').addEventListener('change',e=>{if(e.target.files[0]){el('examples').value='';pick(e.target.files[0],e.target.files[0].name)}});
el('examples').addEventListener('change',async e=>{if(e.target.value==='')return;el('progress').textContent='Loading example…';try{const response=await fetch('/api/example/'+e.target.value);if(!response.ok)throw Error(await response.text());pick(await response.blob(),model.examples[Number(e.target.value)].id);el('progress').textContent='Example loaded. Its publisher label is separate from the prediction.'}catch(error){el('error').textContent=error.message}});
el('analyze').addEventListener('click',async()=>{if(!selected)return;el('analyze').disabled=true;el('progress').textContent='Analyzing full recording…';el('error').textContent='';
try{const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:selected});const payload=await response.json();if(!response.ok)throw Error(payload.error);result=payload;
el('result').hidden=false;el('progress').textContent='Analysis complete. Play or seek the recording to inspect decisions.';
el('summary').textContent=result.alarms.length+' alarm interval(s) · '+result.timing.duration_seconds.toFixed(2)+' s recording · '+result.timing.end_to_end_seconds.toFixed(3)+' s processing. '+(result.warnings.join(' ')||'No input-quality flags.');
el('rows').replaceChildren();for(const row of result.trace){const tr=document.createElement('tr');const names={low_frequency_dominance:'Low-frequency energy',clipping:'Clipping'};for(const value of [row.timestamp_seconds.toFixed(3),row.score===null?'insufficient audio':row.score.toFixed(4),row.alarm?'ALARM':'off',row.reason+(row.padded?' · padded tail':''),(row.quality_flags||[]).map(x=>names[x]||x).join(', ')||'—']){const td=document.createElement('td');td.textContent=value;tr.append(td)}el('rows').append(tr)}update();
}catch(error){el('error').textContent=error.message;el('progress').textContent='Analysis did not complete.'}finally{el('analyze').disabled=false}});
function update(){if(!result)return;const t=el('player').currentTime;let row;for(const item of result.trace)if(item.timestamp_seconds<=t+.001)row=item;
el('time').textContent=t.toFixed(2)+' s';const state=!row?'Waiting for audio context':row.alarm?'ALARM':row.reason==='insufficient_audio'?'Insufficient audio':row.reason==='silence'?'Silent input':(row.quality_flags||[]).length?'Review audio quality':'No alarm';el('state').textContent=state;el('state').className='state'+(row&&row.alarm?' alarm':state!=='No alarm'?' review':'');el('score').textContent=row&&row.score!==null?'Score '+row.score.toFixed(3):'';
const c=el('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height,p=35,d=result.timing.duration_seconds,x=s=>p+(w-2*p)*s/d,y=s=>h-p-(h-2*p)*s;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#3d5d66';ctx.beginPath();ctx.moveTo(p,p);ctx.lineTo(p,h-p);ctx.lineTo(w-p,h-p);ctx.stroke();ctx.fillStyle='#c1d2d5';ctx.font='14px system-ui';ctx.fillText('1',8,p);ctx.fillText('0',8,h-p);ctx.fillText('0 s',p,h-8);ctx.fillText(d.toFixed(1)+' s',w-90,h-8);
ctx.setLineDash([8,6]);ctx.strokeStyle='#e6b573';ctx.beginPath();ctx.moveTo(p,y(model.decision.on_threshold));ctx.lineTo(w-p,y(model.decision.on_threshold));ctx.stroke();ctx.setLineDash([]);ctx.strokeStyle='#edf6f5';ctx.beginPath();let first=true;for(const r of result.trace){if(r.score===null)continue;if(first){ctx.moveTo(x(r.timestamp_seconds),y(r.score));first=false}else ctx.lineTo(x(r.timestamp_seconds),y(r.score))}ctx.stroke();ctx.strokeStyle='#80e5c1';ctx.beginPath();ctx.moveTo(x(t),p);ctx.lineTo(x(t),h-p);ctx.stroke()}
el('player').addEventListener('timeupdate',update);el('player').addEventListener('seeked',update);
el('download').addEventListener('click',()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='drone-scores.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});load();
</script></html>"""


def make_server(bundle, host="127.0.0.1", port=8000, manifest=None, selection=None):
    model = load_model(bundle, require_frozen=True)
    if hasattr(model, 'warmup'):
        model.warmup()
    elif model.encoder is not None:
        import numpy as np
        model.score(np.zeros(model.config.window_samples, dtype=np.float32))
    examples = []
    if bool(manifest) != bool(selection):
        raise ValueError("Examples need both a manifest and a fixed selection")
    if manifest:
        choice = read_json(selection)
        if choice["manifest_sha256"] != file_hash(manifest):
            raise ValueError("Example selection manifest mismatch")
        by_id = {r["id"]: r for r in load_manifest(manifest, True)}
        for identity in choice['recording_ids']:
            row = dict(by_id[identity])
            row['selection_role'] = 'Observed failure appendix' if identity in choice.get('posthoc_failure_ids', []) else 'Fixed before assessment'
            row['title'] = choice.get('titles', {}).get(identity, identity)
            examples.append(row)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, payload, status=200, content_type="application/json"):
            if content_type == "application/json":
                payload = json.dumps(payload, allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self.respond(PAGE.encode(), content_type="text/html; charset=utf-8")
            if path == "/api/model":
                return self.respond({"model_id": model.model_id, "features": model.config.to_dict(),
                    "decision": model.metadata["decision"],
                    "scope": model.metadata['limitations'] if model.metadata.get('experimental') else
                    "Real-audio research prototype. Publisher-level labels; field reliability and event timing are not independently validated."
                    if model.metadata["training"]["origins"] == ["authentic"] else
                    "Synthetic software-test model; not evidence of real drone detection.",
                    'quality_advisories': 'Signal measurements only: low-frequency energy and clipping do not estimate wind speed, SNR, distance or detection reliability.',
                    "limitations": model.metadata["limitations"],
                    "examples": [{"id": r["id"], "label": r["label"], "origin": r["origin"],
                                  'title': r['title'], 'selection_role': r['selection_role']} for r in examples]})
            if path.startswith("/api/example/"):
                try:
                    index = int(path.rsplit("/", 1)[-1])
                    if index < 0:
                        raise ValueError("Negative index")
                    row = examples[index]
                    file = Path(manifest).resolve().parent / row["path"]
                    if file_hash(file) != row["sha256"]:
                        raise ValueError("Example audio content changed")
                    return self.respond(file.read_bytes(), content_type="audio/wav")
                except (ValueError, IndexError, OSError):
                    return self.respond({"error": "Example unavailable"}, 404)
            return self.respond({"error": "Not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/analyze":
                return self.respond({"error": "Not found"}, 404)
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.respond({"error": "Cross-origin upload is not supported"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64 * 1024 * 1024:
                    return self.respond({"error": "Choose an audio file up to 64 MB"}, 413)
                data = self.rfile.read(length)
                info = sf.info(io.BytesIO(data))
                if not 0 < info.duration <= 600 or info.channels > 16:
                    return self.respond({"error": "Audio must be nonempty, at most 10 minutes, and at most 16 channels"}, 422)
                with tempfile.TemporaryDirectory(prefix="drone-upload-") as temporary:
                    path = Path(temporary) / "recording.audio"
                    path.write_bytes(data)
                    with threadpool_limits(limits=1):
                        trace, timing = infer_file(model, path)
                warnings = []
                if timing["clipped_fraction"] > .001:
                    warnings.append("Input clipping detected; predictions may be unreliable.")
                if timing["rms"] < model.config.silence_rms:
                    warnings.append("Silent input.")
                if timing["duration_seconds"] < model.config.window_seconds:
                    warnings.append("Short recording uses a padded context.")
                if timing['rms'] >= model.config.silence_rms and sum(row['score'] is not None for row in trace) < model.metadata['decision']['on_hits']:
                    warnings.append('Not enough scored windows to confirm an alarm; use a longer recording.')
                if trace and sum('low_frequency_dominance' in row['quality_flags'] for row in trace) / len(trace) >= .2:
                    warnings.append('Strong low-frequency energy affects part of this recording. Review the audio; this is not a wind-speed or reliability estimate.')
                return self.respond({"model_id": model.model_id, "trace": trace,
                    "timing": timing, "warnings": warnings,
                    "alarms": alarm_intervals(trace, timing["duration_seconds"], merge_gap=1.0)})
            except (ValueError, RuntimeError, OSError):
                return self.respond({"error": "Unable to decode this audio. Try WAV, FLAC or OGG with finite samples."}, 422)

    return ThreadingHTTPServer((host, port), Handler)


def serve(bundle, host="127.0.0.1", port=8000, manifest=None, selection=None):
    server = make_server(bundle, host, port, manifest, selection)
    print(f"Drone audio product running at http://{host}:{server.server_port} (local analysis)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
