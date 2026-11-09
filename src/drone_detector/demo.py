"""Self-contained, disclosed recorded-replay demonstration (no web service)."""

import base64
import html
import json
from pathlib import Path

from .common import file_hash, read_json
from .data import checked_audio, load_manifest
from .model import load_model
from .stream import infer_file


def build_demo(bundle, manifest, fixed_ids, output):
    model = load_model(bundle, require_frozen=True)
    wanted = read_json(fixed_ids)
    ids = wanted["recording_ids"]
    if wanted.get("manifest_sha256") != file_hash(manifest):
        raise ValueError("Demo selection manifest mismatch")
    if model.metadata["training"]["manifest_sha256"] != file_hash(manifest):
        raise ValueError("Demo manifest differs from frozen model")
    by_id = {r["id"]: r for r in load_manifest(manifest, require_split=True)}
    if len(set(ids)) != len(ids) or set(ids) - by_id.keys():
        raise ValueError("Demo selection contains duplicate or missing IDs")
    records = []
    for identity in ids:
        record = by_id[identity]
        path = Path(manifest).resolve().parent / record["path"]
        checked_audio(record, manifest, model.config.sample_rate)
        trace, timing = infer_file(model, path)
        mime = "audio/" + ("wav" if path.suffix.lower() == ".wav" else path.suffix[1:])
        records.append({"id": identity, "origin": record["origin"], "label": record["label"],
                        "source": record["source"], "events": record["events"],
                        "demo_role": "Post-hoc failure appendix" if identity in wanted.get("posthoc_failure_ids", []) else "Fixed before training",
                        "negative_kind": record.get("negative_kind"),
                        "audio": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode(),
                        "trace": trace, "timing": timing})
    payload = json.dumps({"records": records, "model_id": model.model_id,
                          "decision": model.metadata["decision"], "selection": wanted}, allow_nan=False)
    # Data is never interpreted as markup/script (IDs and paths may be user input).
    payload = payload.replace("<", "\\u003c").replace("&", "\\u0026")
    title = "Acoustic detector · recorded replay"
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>
body{font:16px system-ui,sans-serif;background:#10181d;color:#ecf0ef;max-width:1000px;margin:auto;padding:28px}
h1{font-size:30px}small,p{line-height:1.5}a{color:#7bdfbd}.notice{border:1px solid #d89d44;padding:18px;color:#ffe0aa}
select,button{font:inherit;padding:10px;border-radius:6px;background:#203339;color:white;border:1px solid #6c858d}
audio{display:block;width:100%;margin:18px 0}canvas{width:100%;height:260px;background:#14262d;border-radius:8px}
code{overflow-wrap:anywhere}.status{display:flex;gap:30px;font-size:23px;padding:20px 0}.on{color:#ffa678}.off{color:#7bdfbd}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:7px;border-bottom:1px solid #30434a}
.scroll{max-height:270px;overflow:auto}label{display:block;margin-top:18px}
</style><h1>__TITLE__</h1>
<p class="notice">Experimental. This page is recorded-stream replay, not microphone capture or speaker playback.
Synthetic signals are software tests, not evidence that a drone will be detected. Misses and false alarms remain visible.</p>
<p>Model <code id="model"></code></p><label for="record">Fixed, disclosed input set</label>
<select id="record"></select><p id="description"></p><audio id="player" controls preload="metadata"></audio>
<div class="status"><span id="time">0.00 s</span><span id="score">Waiting for context</span><strong id="alarm">NO ALARM</strong></div>
<canvas id="chart" width="1000" height="280" role="img" aria-label="Drone score by time with ground-truth event shading and decision threshold"></canvas>
<p>Green shading: annotated events. White: model score. Orange: on threshold. Scores appear only after context arrives;
no predictions are backdated. End-of-file padding is disclosed below. Scores are classifier outputs, not calibrated probabilities.</p>
<p id="protocol"></p><div class="scroll"><table><thead><tr><th>Audio time (s)</th><th>Score</th><th>Alarm</th><th>Input handling</th></tr></thead><tbody id="rows"></tbody></table></div>
<script type="application/json" id="payload">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent);
const get=id=>document.getElementById(id), player=get('player'), select=get('record');let current;
get('model').textContent=data.model_id;
get('protocol').textContent='Frozen decision: '+JSON.stringify(data.decision)+'. Selection: '+data.selection.reason;
data.records.forEach((r,i)=>{const option=document.createElement('option');option.value=i;option.textContent=r.id+' · '+r.demo_role;select.append(option)});
function draw(){const c=get('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height,p=35,d=current.timing.duration_seconds;
ctx.clearRect(0,0,w,h);const x=t=>p+(w-2*p)*t/d,y=s=>h-p-(h-2*p)*s;
ctx.fillStyle='#214d41';for(const [a,b] of current.events)ctx.fillRect(x(a),p,x(b)-x(a),h-2*p);
ctx.strokeStyle='#557079';ctx.beginPath();ctx.moveTo(p,p);ctx.lineTo(p,h-p);ctx.lineTo(w-p,h-p);ctx.stroke();
ctx.fillStyle='#c7dbdd';ctx.font='14px sans-serif';ctx.fillText('1',8,p);ctx.fillText('0',8,h-p);ctx.fillText('0 s',p,h-8);ctx.fillText(d.toFixed(1)+' s',w-90,h-8);
ctx.setLineDash([8,6]);ctx.strokeStyle='#ffc06e';ctx.beginPath();ctx.moveTo(p,y(data.decision.on_threshold));ctx.lineTo(w-p,y(data.decision.on_threshold));ctx.stroke();ctx.setLineDash([]);
ctx.strokeStyle='#ffffff';ctx.beginPath();let first=true;for(const r of current.trace){if(r.score===null)continue;if(first){ctx.moveTo(x(r.timestamp_seconds),y(r.score));first=false}else ctx.lineTo(x(r.timestamp_seconds),y(r.score))}ctx.stroke();
ctx.strokeStyle='#7bdfbd';ctx.beginPath();ctx.moveTo(x(player.currentTime),p);ctx.lineTo(x(player.currentTime),h-p);ctx.stroke();}
function update(){const t=player.currentTime;let row;for(const r of current.trace)if(r.timestamp_seconds<=t+.001)row=r;
get('time').textContent=t.toFixed(2)+' s';get('score').textContent=row&&row.score!==null?'Score '+row.score.toFixed(3):'Waiting / insufficient audio';
get('alarm').textContent=row&&row.alarm?'ALARM':'NO ALARM';get('alarm').className=row&&row.alarm?'on':'off';draw()}
function change(){current=data.records[Number(select.value)];player.src=current.audio;
get('description').textContent=current.demo_role+' | Origin: '+current.origin+' | Label: '+current.label+' | Source: '+current.source+(current.negative_kind?' | '+current.negative_kind:'');
get('rows').replaceChildren();for(const r of current.trace){const tr=document.createElement('tr');for(const v of [r.timestamp_seconds.toFixed(3),r.score===null?'—':r.score.toFixed(5),r.alarm?'ALARM':'off',r.reason+(r.padded?' · zero-padded tail':'')]){const td=document.createElement('td');td.textContent=v;tr.append(td)}get('rows').append(tr)}update()}
select.addEventListener('change',change);player.addEventListener('timeupdate',update);player.addEventListener('seeked',update);change();
</script></html>"""
    page = page.replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", payload)
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite demonstration")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    return {"output": str(output), "records": len(records), "model_id": model.model_id}
