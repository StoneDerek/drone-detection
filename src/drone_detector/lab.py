"""Local model comparison and genuine convolutional Grad-CAM inspection."""

import base64
import io
import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits

from .audio import decode
from .common import read_json
from .model import load_model
from .spectrogram import axes
from .stream import alarm_intervals, replay_audio

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audio model lab</title><style>
:root{color-scheme:dark;font:15px system-ui;background:#0c171d;color:#e8f2f3}body{max-width:1240px;margin:auto;padding:28px}h1{font-size:32px;margin-bottom:6px}p{line-height:1.5;color:#b4c8ce}.tag{color:#77dcbc;text-transform:uppercase;letter-spacing:2px;font-size:12px}.card{background:#15272f;border:1px solid #2c444d;border-radius:14px;padding:22px;margin:20px 0}.notice{border-left:3px solid #e8b569;padding:12px 18px;background:#31291f;color:#f3d7aa}button,select,input{font:inherit}button,select,input[type=number]{background:#26464d;color:#eef8f8;border:1px solid #527778;border-radius:6px;padding:9px}button{cursor:pointer}button:disabled{opacity:.4;cursor:wait}.bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.models{display:flex;gap:12px;flex-wrap:wrap}.models label{background:#213b43;padding:9px;border-radius:6px}audio{width:100%;margin:18px 0}canvas{width:100%;background:#0d1c24;border-radius:6px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:20px}.plots canvas{height:360px}.plots h3{font-size:16px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:9px;text-align:left;border-bottom:1px solid #2e464e}code{font-size:12px;overflow-wrap:anywhere}.scroll{overflow:auto}.error{color:#ffac9b}.status{color:#8ce2c5}.mini{font-size:13px}input[type=number]{width:90px}#chart{height:220px}.legend{display:flex;gap:14px;flex-wrap:wrap}@media(max-width:750px){body{padding:15px}.plots{grid-template-columns:1fr}.card{padding:16px}}
</style><span class="tag">Recorded audio · model comparison</span><h1>Audio model lab</h1>
<p>Compare neural and non-neural scores. Inspect the spectrogram and the CNN's Grad-CAM for the same audio context.</p>
<p class="notice" id="scope">Loading models…</p>
<section class="card"><div class="bar"><label>Add recordings <input id="file" type="file" multiple accept="audio/*,.wav,.flac,.ogg"></label><button id="analyze" disabled>Compare all loaded recordings</button><button id="analyzecurrent" disabled>Compare current recording</button></div>
<div class="bar"><label>Current recording <select id="recording"></select></label><span id="loadedcount"></span></div>
<p>WAV / FLAC / OGG · up to 64 MB and 2 minutes. Audio is processed locally; server temporary files are removed after each request.</p>
<p class="mini">Playback keeps the recording's original level. Model preprocessing differs: the spectrogram CNNs/SVM use a per-context peak reference; the YAMNet control has no shared waveform RMS target. A level change inside a context can remain visible after peak normalization.</p>
<div id="models" class="models"></div><p id="progress" role="status"></p><p id="error" class="error" role="alert"></p></section>
<section class="card" id="batchoverview" hidden><h2>Compare recordings</h2><p>Results remain loaded when switching recordings. Each cell shows whether an alarm occurred, the highest score and the score timeline; dashed lines show that model's threshold. Scores are not calibrated probabilities.</p><p id="batchstatus"></p><div class="scroll"><table><thead id="batchhead"></thead><tbody id="batchrows"></tbody></table></div><button id="batchdownload" disabled>Download all recording results</button></section>
<section class="card" id="result" hidden><audio id="player" controls></audio><div id="legend" class="legend"></div><canvas id="chart" width="1200" height="240"></canvas>
<p>Solid lines: scores. Dashed lines: each model’s alarm-on threshold. Scores are uncalibrated and cannot be compared as probabilities. Each alarm uses its own development-selected threshold and confirmation rule.</p>
<div class="scroll"><table><thead><tr><th>Model</th><th>Score at playback time</th><th>Alarm / input</th><th>Quality flags</th></tr></thead><tbody id="decisions"></tbody></table></div>
<div class="bar"><button id="export">Download comparison JSON</button><span id="duration"></span></div></section>
<section class="card" id="explanation" hidden><h2>Inspect a CNN decision</h2><div class="bar">
<label>Model <select id="cnn"></select></label><label>Context <select id="window"></select></label>
<label>Target <select id="target"><option value="drone">Drone logit</option><option value="non_drone">Non-drone logit</option></select></label>
<label>Layer <select id="layer"><option value="conv1">First convolution</option><option value="conv2">Middle convolution</option><option value="features">Final convolution</option></select></label>
<button id="explain">Compute Grad-CAM</button></div><p id="camstatus" role="status"></p>
<p class="mini">Choose any trained spectrogram CNN here, independently of the score comparison above. <span id="cnncount"></span> <button id="playbackcontext">Use playback time</button></p>
<div class="bar"><label>View <select id="camview"><option value="heatmap">Heatmap only</option><option value="overlay">Overlay on spectrogram</option></select></label><label>Overlay opacity <input id="opacity" type="range" min="0" max="1" step="0.05" value="0.65"><output id="opacityvalue">65%</output></label><label><input id="nativecells" type="checkbox" checked> Show native attribution cells</label></div>
<div class="bar"><label>Frequency from <input id="fmin" type="number" min="0" max="8000" value="50"> Hz</label><label>to <input id="fmax" type="number" min="1" max="8000" value="3000"> Hz</label><label>Time from <input id="tmin" type="number" step="0.001" value="0"> s</label><label>to <input id="tmax" type="number" step="0.001" value="3"> s</label><button id="zoom">Apply zoom</button><button id="lowzoom">Below 3 kHz</button><button id="resetzoom">Reset zoom</button><button id="camdownload" disabled>Download explanation JSON</button></div>
<p id="zoomstatus" class="mini">Drag a rectangle in either plot to zoom both. Mouse wheel zooms at the cursor. Shift-drag pans. Double-click resets.</p>
<div class="plots"><div><h3>Model input spectrogram</h3><canvas id="spec" width="580" height="400" style="touch-action:none;cursor:crosshair"></canvas></div><div><h3 id="camtitle">Grad-CAM heatmap</h3><canvas id="cam" width="580" height="400" style="touch-action:none;cursor:crosshair"></canvas></div></div>
<div class="bar mini"><span>Relative positive attribution</span><span>0</span><span style="display:inline-block;width:240px;height:16px;background:linear-gradient(to right,#05091c,#1e3c87,#00aab4,#f5be2d,#fffae1)"></span><span>1</span><span id="camrange"></span><span>Scaled within each map; brightness is not comparable across models or contexts.</span></div><p id="plotreadout" class="mini">Hover over either plot to inspect time, frequency and values.</p>
<p id="tensor"></p><code id="identity"></code><p class="mini">The horizontal axis is time within the selected context; low frequencies are at the bottom. Both plots use a shared linear Hz display; values from the model's original linear or mel bins are projected onto that axis without adding spectral detail. Warm overlay colours mark positive attribution to the selected logit. Maps are normalized independently; brightness is relative within a map. A zero map is reported rather than filled with invented detail. Grad-CAM is computed from gradients of the trained CNN, at the selected layer's native resolution; interpolation adds no explanatory detail. Attribution can reveal shortcuts but does not prove a useful or causal acoustic cue.</p>
<h3>Training history for selected CNN</h3><canvas id="learning" width="1100" height="190" style="height:190px"></canvas><p class="mini">Green: sampled training objective including regularization. Orange: the deterministic development loss used to select this model, checked every two epochs. White dashed line: selected checkpoint. <span id="learningmetric"></span> Values are losses, not accuracies; sampling and regularization differ.</p></section>
<section class="card"><h2>Training and development evidence</h2><p id="data"></p><p class="notice">Thresholds were selected on this development set under a 5% negative-file budget: at most nine alarmed files out of 188. The matching false-positive counts largely reflect calibration, not independent validation or identical mistakes. Recall is compared at this budget; false-alarm rates on new recordings can differ.</p><div class="scroll"><table><thead><tr><th>Model</th><th>Method</th><th>Input tensor / descriptor</th><th>Development positives detected</th><th>Development negatives alarmed</th><th>Selected epoch</th><th>Contexts seen at checkpoint</th></tr></thead><tbody id="inventory"></tbody></table></div>
<p>Frequency-axis warping, log intensity and tensor shape are separate choices. Models share real training/development splits; training recipes differ by experiment. Counts describe correlated publisher-labeled files. The previously assessed default is preserved; this lab supplies development evidence.</p></section>
<script>
const el=id=>document.getElementById(id);
let catalog,file,objectUrl,result,explanation,pixels,view,visible,gesture,working=false;
const recordings=[],analysisByFile=new Map();
const colors=['#8de4c5','#f5c376','#97baff','#e9a5e5','#f09886','#c8d68d','#aeb5ed','#78d1dc','#dcae8e','#ffffff','#54e355','#f474ba','#f5e442','#598bff','#d05b35','#b970e0','#6dc1a0','#daae3b','#99c8ee','#e7c4b0'];
const modelColor=name=>colors[Math.max(0,catalog.models.findIndex(m=>m.name===name))%colors.length];
const plot={left:52,top:18,right:18,bottom:52};
const heatStops=[[5,9,28],[30,60,135],[0,170,180],[245,190,45],[255,250,225]];
const heatColors=Array.from({length:256},(_,value)=>{
  const at=value/255*4,lo=Math.min(3,Math.floor(at)),fraction=at-lo;
  return heatStops[lo].map((v,i)=>Math.round(v*(1-fraction)+heatStops[lo+1][i]*fraction));
});

function selectedCNN(){return catalog?.models.find(m=>m.name===el('cnn').value&&m.gradcam);}
function contextRows(){return result?.models.find(m=>m.name===el('cnn').value)?.trace||result?.models[0]?.trace||[];}
function busy(value){
  working=value;
  for(const id of ['analyze','analyzecurrent','explain','file','cnn','window','target','layer','playbackcontext','openexample','example'])if(el(id))el(id).disabled=value;
  if(!value){el('analyze').disabled=el('analyzecurrent').disabled=!file;el('explain').disabled=!file||!result||!selectedCNN();}
}
async function load(){
  try{
    catalog=await(await fetch('/api/models')).json();
    el('scope').textContent=catalog.scope;el('data').textContent=catalog.data_summary;
    el('cnncount').textContent='Grad-CAM is available for '+catalog.models.filter(m=>m.gradcam).length+' CNNs.';
    for(const m of catalog.models){
      const label=document.createElement('label'),input=document.createElement('input');
      input.type='checkbox';input.value=m.name;input.checked=catalog.defaults.includes(m.name);
      label.append(input,document.createTextNode(' '+m.name));el('models').append(label);
      const row=document.createElement('tr');
      for(const value of [m.name,m.method,m.shape,m.development,m.false_positives,m.selected_epoch||'—',m.training_windows_seen||'—']){
        const td=document.createElement('td');td.textContent=value;row.append(td);
      }
      el('inventory').append(row);
      const option=document.createElement('option');option.value=m.name;
      option.disabled=!m.gradcam;option.textContent=m.name+(m.gradcam?'':' · Grad-CAM unavailable');el('cnn').append(option);
    }
    const preferred=catalog.defaults.find(name=>catalog.models.some(m=>m.name===name&&m.gradcam));
    el('cnn').value=preferred||catalog.models.find(m=>m.gradcam)?.name||'';
  }catch(e){el('error').textContent=e.message;}
}
function selectFile(chosen){
  const index=recordings.findIndex(entry=>entry.file===chosen);if(index<0)return;
  file=chosen;el('recording').value=String(index);
  const transfer=new DataTransfer();transfer.items.add(file);el('file').files=transfer.files;
  if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=URL.createObjectURL(file);el('player').src=objectUrl;
  result=analysisByFile.get(file)||null;invalidateExplanation();el('error').textContent='';
  el('result').hidden=false;el('chart').hidden=!result;el('legend').hidden=!result;
  el('decisions').closest('.scroll').hidden=!result;el('export').disabled=!result;
  el('explanation').hidden=!result;el('duration').textContent=result?'':'Choose Compare current or Compare all to compute scores.';
  if(result)showResult(result);else el('decisions').replaceChildren();
  el('analyze').disabled=el('analyzecurrent').disabled=working;
  el('file').dispatchEvent(new Event('labrecordingchange'));renderBatch();
}
function addFiles(files,metadata=[],selectNew=true){
  const first=files[0];
  for(const [i,added] of files.entries())if(!recordings.some(entry=>entry.file===added))recordings.push({file:added,metadata:metadata[i]||null});
  el('recording').replaceChildren();
  for(const [index,entry] of recordings.entries()){
    const option=document.createElement('option');option.value=index;
    option.textContent=entry.file.name+(entry.metadata?' · example':' · upload');el('recording').append(option);
  }
  el('loadedcount').textContent=recordings.length+' recordings loaded';
  if(first)selectFile(selectNew||!file?first:file);
}
globalThis.audioLabBatch={addFiles,selectFile,get currentResult(){return result;},get recordings(){return recordings;}};
el('file').onchange=e=>addFiles([...e.target.files]);
el('recording').onchange=()=>selectFile(recordings[Number(el('recording').value)].file);
function showResult(data){
  result=data;el('result').hidden=false;el('chart').hidden=el('legend').hidden=false;
  el('decisions').closest('.scroll').hidden=false;el('export').disabled=false;
  el('duration').textContent=data.audio.duration_seconds.toFixed(2)+' s recorded audio · '+data.processing_seconds.toFixed(2)+' s total processing';
  el('legend').replaceChildren();
  for(const m of data.models){
    const name=document.createElement('span');name.style.color=modelColor(m.name);name.textContent=m.name;el('legend').append(name);
  }
  el('explanation').hidden=!catalog.models.some(m=>m.gradcam);populateWindows();update();
}
function renderBatch(){
  el('batchoverview').hidden=!recordings.length;
  const names=[...new Set([...analysisByFile.values()].flatMap(value=>value.models.map(model=>model.name)))];
  el('batchhead').replaceChildren();const heading=document.createElement('tr');
  for(const name of ['Recording',...names]){const cell=document.createElement('th');cell.textContent=name;heading.append(cell);}el('batchhead').append(heading);
  el('batchrows').replaceChildren();
  for(const entry of recordings){
    const row=document.createElement('tr'),label=document.createElement('td'),button=document.createElement('button');
    button.textContent=entry.file.name;button.onclick=()=>selectFile(entry.file);label.append(button);
    if(entry.metadata){const note=document.createElement('div');note.className='mini';note.textContent='Publisher: '+entry.metadata.publisher_label;label.append(note);}
    if(entry.file===file)label.style.borderLeft='3px solid #8de4c5';row.append(label);
    const analysis=analysisByFile.get(entry.file);
    for(const name of names){
      const cell=document.createElement('td'),model=analysis?.models.find(m=>m.name===name);
      if(!model){cell.textContent='Not computed';row.append(cell);continue;}
      const scores=model.trace.filter(r=>r.score!==null),maximum=Math.max(0,...scores.map(r=>r.score));
      const status=document.createElement('div');status.textContent=(model.trace.some(r=>r.alarm)?'ALARM':'No alarm')+' · max '+maximum.toFixed(4);cell.append(status);
      const canvas=document.createElement('canvas');canvas.width=160;canvas.height=45;canvas.style.cssText='height:45px;min-width:140px';
      const ctx=canvas.getContext('2d');ctx.strokeStyle='#617780';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(0,44-43*model.decision.on_threshold);ctx.lineTo(160,44-43*model.decision.on_threshold);ctx.stroke();ctx.setLineDash([]);
      ctx.strokeStyle='#8de4c5';ctx.beginPath();for(const [i,r] of scores.entries()){const x=160*r.timestamp_seconds/analysis.audio.duration_seconds,y=44-43*r.score;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);}ctx.stroke();cell.append(canvas);row.append(cell);
    }
    el('batchrows').append(row);
  }
  el('batchstatus').textContent=analysisByFile.size+' of '+recordings.length+' recordings have results. Select a recording to inspect it; previous results stay loaded.';
  el('batchdownload').disabled=!analysisByFile.size;
}
async function compareRecordings(all){
  const names=[...el('models').querySelectorAll('input:checked')].map(x=>x.value);
  if(!file||!names.length){el('error').textContent='Choose audio and at least one model.';return;}
  const targets=all?[...recordings]:recordings.filter(entry=>entry.file===file);
  busy(true);el('error').textContent='';el('progress').textContent='Analyzing recording with selected models…';
  try{
    for(const [index,entry] of targets.entries()){
      el('progress').textContent='Recording '+(index+1)+'/'+targets.length+' · '+entry.file.name;
      const response=await fetch('/api/analyze?models='+encodeURIComponent(names.join(',')),{method:'POST',body:entry.file});
      const data=await response.json();if(!response.ok)throw Error(data.error);
      data.filename=entry.file.name;analysisByFile.set(entry.file,data);
      if(file===entry.file)showResult(data);renderBatch();
    }
    el('progress').textContent='Comparison complete for '+targets.length+' recording(s). Switch recordings to compare retained results.';
  }catch(e){el('error').textContent=e.message;el('progress').textContent='Comparison did not complete.';if(!result)el('result').hidden=true;}
  finally{busy(false);}
}
el('analyze').onclick=()=>compareRecordings(true);el('analyzecurrent').onclick=()=>compareRecordings(false);
el('batchdownload').onclick=()=>download({scope:catalog.scope,recordings:recordings.filter(entry=>analysisByFile.has(entry.file)).map(entry=>({filename:entry.file.name,publisher:entry.metadata,comparison:analysisByFile.get(entry.file)}))},'all-recording-comparisons.json');

function invalidateExplanation(){
  explanation=pixels=view=visible=null;el('camdownload').disabled=true;
  el('camstatus').textContent='Compute Grad-CAM for the selected model, context and target.';
  for(const id of ['tensor','identity','camrange'])el(id).textContent='';
  for(const id of ['spec','cam']){const c=el(id);c.getContext('2d').clearRect(0,0,c.width,c.height);}
}
function populateWindows(){
  const previous=el('window').value;el('window').replaceChildren();
  for(const row of contextRows()){
    const option=document.createElement('option');option.value=row.window_start_seconds;
    option.textContent=row.window_start_seconds.toFixed(2)+'–'+row.timestamp_seconds.toFixed(2)+' s'+(row.padded?' · padded':'');
    el('window').append(option);
  }
  if([...el('window').options].some(option=>option.value===previous))el('window').value=previous;
  drawLearning();invalidateExplanation();
}
el('cnn').onchange=populateWindows;
for(const id of ['window','target','layer'])el(id).onchange=invalidateExplanation;
el('playbackcontext').onclick=()=>{
  const row=contextRows().filter(r=>r.timestamp_seconds<=el('player').currentTime+.001).at(-1)||contextRows()[0];
  if(row){el('window').value=String(row.window_start_seconds);invalidateExplanation();}
};

function drawLearning(){
  const model=selectedCNN(),history=model?.history||[],c=el('learning'),ctx=c.getContext('2d'),w=c.width,h=c.height,p=30;
  ctx.clearRect(0,0,w,h);if(!history.length)return;
  const metric=model.selection_loss||'development_balanced_bag_loss';
  el('learningmetric').textContent=metric==='development_consistent_max_balanced_loss'?'This model uses the same maximum-context score for both recording classes.':'This older experiment uses a positive maximum and a negative window mean; dropout can exploit this asymmetric objective.';
  const maximum=Math.max(.01,...history.flatMap(r=>[r.training_loss,r[metric]||0]))*1.1;
  const n=history.at(-1).epoch,x=e=>p+(w-2*p)*(e-1)/Math.max(1,n-1),y=v=>h-p-(h-2*p)*v/maximum;
  ctx.strokeStyle='#435e68';ctx.strokeRect(p,p,w-2*p,h-2*p);
  for(const [field,color] of [['training_loss','#8de4c5'],[metric,'#f5c376']]){
    ctx.strokeStyle=color;ctx.beginPath();let first=true;
    for(const r of history){if(r[field]===undefined)continue;if(first){ctx.moveTo(x(r.epoch),y(r[field]));first=false;}else ctx.lineTo(x(r.epoch),y(r[field]));}
    ctx.stroke();
  }
  ctx.strokeStyle='white';ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(x(model.selected_epoch),p);ctx.lineTo(x(model.selected_epoch),h-p);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#c5d8dc';ctx.font='13px system-ui';ctx.fillText('Epoch 1',p,h-8);ctx.fillText('Epoch '+n,w-85,h-8);ctx.fillText(maximum.toFixed(2),0,p);ctx.fillText('0',6,h-p);
}
function update(){
  if(!result)return;
  const time=el('player').currentTime;el('decisions').replaceChildren();
  for(const m of result.models){
    const row=m.trace.filter(r=>r.timestamp_seconds<=time+.001).at(-1),tr=document.createElement('tr');
    for(const text of [m.name,row?(row.score===null?'—':row.score.toFixed(4)):'Waiting for context',row?(row.alarm?'ALARM':row.reason==='silence'?'Silent input':row.reason==='insufficient_audio'?'Insufficient audio':'No alarm'):'Waiting',(row?.quality_flags||[]).join(', ')||'—']){
      const td=document.createElement('td');td.textContent=text;tr.append(td);
    }
    el('decisions').append(tr);
  }
  const c=el('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height,p=24;
  const x=t=>p+(w-2*p)*t/result.audio.duration_seconds,y=s=>h-p-(h-2*p)*s;
  ctx.clearRect(0,0,w,h);ctx.strokeStyle='#405d68';ctx.strokeRect(p,p,w-2*p,h-2*p);
  ctx.fillStyle='#c6d9df';ctx.font='13px system-ui';ctx.fillText('1',5,p);ctx.fillText('0',5,h-p);ctx.fillText('0 s',p,h-6);ctx.fillText(result.audio.duration_seconds.toFixed(1)+' s',w-70,h-6);
  for(const m of result.models){
    ctx.strokeStyle=modelColor(m.name);ctx.globalAlpha=.5;ctx.setLineDash([6,6]);ctx.beginPath();ctx.moveTo(p,y(m.decision.on_threshold));ctx.lineTo(w-p,y(m.decision.on_threshold));ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
    ctx.beginPath();let first=true;
    for(const r of m.trace){if(r.score===null)continue;if(first){ctx.moveTo(x(r.timestamp_seconds),y(r.score));first=false;}else ctx.lineTo(x(r.timestamp_seconds),y(r.score));}ctx.stroke();
  }
  ctx.strokeStyle='#ffffff';ctx.beginPath();ctx.moveTo(x(time),p);ctx.lineTo(x(time),h-p);ctx.stroke();
}
el('player').addEventListener('timeupdate',update);el('player').addEventListener('seeked',update);
function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
el('export').onclick=()=>download(result,'audio-model-comparison.json');el('camdownload').onclick=()=>download(explanation,'gradcam-explanation.json');
function unpack(value){return Uint8Array.from(atob(value),c=>c.charCodeAt(0));}
el('explain').onclick=async()=>{
  if(!file||!result||!selectedCNN())return;
  busy(true);invalidateExplanation();el('camstatus').textContent='Computing gradients for the selected context…';
  try{
    const query=new URLSearchParams({model:el('cnn').value,start:el('window').value,target:el('target').value,layer:el('layer').value});
    const response=await fetch('/api/explain?'+query,{method:'POST',body:file});const data=await response.json();
    if(!response.ok)throw Error(data.error);
    explanation=data;pixels={spec:unpack(data.spectrogram_u8),cam:unpack(data.heatmap_u8),native:data.native_heatmap_u8?unpack(data.native_heatmap_u8):null};
    el('nativecells').disabled=!pixels.native;
    el('identity').textContent=data.name+' · model '+data.model_id;el('camdownload').disabled=false;
    el('camstatus').textContent='Context '+data.start_seconds.toFixed(2)+'–'+data.end_seconds.toFixed(2)+' s'+(data.padded?' · zero-padded context':'')+' · classifier score (drone) '+data.classifier_score.toFixed(4)+' · explain '+data.target+' logit · '+(data.has_positive_attribution?'positive attribution shown':'zero positive-attribution map');
    el('tensor').textContent=data.axis+' frequency axis · '+data.intensity+' intensity · input ['+data.shape.join(', ')+', 1] · Hann window '+(1000*data.n_fft/16000).toFixed(0)+' ms · FFT spacing '+(16000/data.n_fft).toFixed(2)+' Hz · Grad-CAM native grid '+data.coarse_shape.join(' × ')+'. Display arrays are quantized to 8 bits; model inputs use float32.';
    const map=pixels.native||pixels.cam;let minimum=255,maximum=0;
    for(const v of map){minimum=Math.min(minimum,v);maximum=Math.max(maximum,v);}
    el('camrange').textContent='Native grid '+data.coarse_shape.join(' × ')+' · map range '+(minimum/255).toFixed(2)+'–'+(maximum/255).toFixed(2)+(data.has_positive_attribution?'':' · zero attribution');
    setView({fMin:Number(el('fmin').value),fMax:Number(el('fmax').value),tMin:data.frame_times_seconds[0],tMax:data.frame_times_seconds.at(-1)});
  }catch(e){el('camstatus').textContent=e.message;}
  finally{busy(false);}
};

function domain(){return {fMin:explanation.frequencies_hz[0],fMax:explanation.frequencies_hz.at(-1),tMin:explanation.frame_times_seconds[0],tMax:explanation.frame_times_seconds.at(-1)};}
function clampRange(lo,hi,minimum,maximum){
  const span=Math.min(hi-lo,maximum-minimum);lo=Math.max(minimum,Math.min(lo,maximum-span));return [lo,lo+span];
}
function setView(next){
  if(!explanation)return;
  if(!Object.values(next).every(Number.isFinite)||next.fMax<=next.fMin||next.tMax<=next.tMin){el('zoomstatus').textContent='Choose increasing finite time and frequency limits.';return;}
  const d=domain(),[fMin,fMax]=clampRange(next.fMin,next.fMax,d.fMin,d.fMax),[tMin,tMax]=clampRange(next.tMin,next.tMax,d.tMin,d.tMax);
  const bins=explanation.frequencies_hz.map((v,i)=>v>=fMin-.001&&v<=fMax+.001?i:-1).filter(i=>i>=0);
  const frames=explanation.frame_times_seconds.map((v,i)=>v>=tMin-.00001&&v<=tMax+.00001?i:-1).filter(i=>i>=0);
  if(!bins.length||!frames.length){el('zoomstatus').textContent='This selection contains no model bins or frames. Zoom out or reset.';return;}
  view={fMin,fMax,tMin,tMax};for(const [id,key] of [['fmin','fMin'],['fmax','fMax'],['tmin','tMin'],['tmax','tMax']])el(id).value=view[key].toFixed(3);
  drawExplanation();
}
function nearestIndex(values,value){
  let lo=0,hi=values.length-1;
  while(hi-lo>1){const middle=Math.floor((lo+hi)/2);if(values[middle]<value)lo=middle;else hi=middle;}
  return Math.abs(values[lo]-value)<=Math.abs(values[hi]-value)?lo:hi;
}
function drawExplanation(){
  if(!explanation||!view)return;
  const data=explanation,[frequencyCount,timeCount]=data.shape;
  const bins=data.frequencies_hz.map((v,i)=>v>=view.fMin-.001&&v<=view.fMax+.001?i:-1).filter(i=>i>=0);
  const frames=data.frame_times_seconds.map((v,i)=>v>=view.tMin-.00001&&v<=view.tMax+.00001?i:-1).filter(i=>i>=0);
  if(!bins.length||!frames.length)return;
  const mode=el('camview').value,opacity=Number(el('opacity').value),native=el('nativecells').checked&&pixels.native;
  const [nf,nt]=data.coarse_shape;
  visible={...view};el('camtitle').textContent=mode==='heatmap'?'Grad-CAM heatmap':'Grad-CAM overlay';
  el('opacity').disabled=mode!=='overlay';el('opacityvalue').textContent=Math.round(100*opacity)+'%';
  el('zoomstatus').textContent=view.tMin.toFixed(2)+'–'+view.tMax.toFixed(2)+' s · '+Math.round(view.fMin)+'–'+Math.round(view.fMax)+' Hz. Drag to zoom; wheel zooms at cursor; Shift-drag pans; double-click resets.';
  for(const id of ['spec','cam']){
    const target=el(id),ctx=target.getContext('2d'),buffer=document.createElement('canvas');
    const width=target.width-plot.left-plot.right,height=target.height-plot.top-plot.bottom;
    buffer.width=width;buffer.height=height;
    const fIndices=Array.from({length:height},(_,y)=>nearestIndex(data.frequencies_hz,view.fMax-y/Math.max(1,height-1)*(view.fMax-view.fMin)));
    const tIndices=Array.from({length:width},(_,x)=>nearestIndex(data.frame_times_seconds,view.tMin+x/Math.max(1,width-1)*(view.tMax-view.tMin)));
    const bctx=buffer.getContext('2d'),image=bctx.createImageData(width,height);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const f=fIndices[y],t=tIndices[x],source=f*timeCount+t,p=(y*width+x)*4,v=pixels.spec[source];
      const cell=Math.min(nf-1,Math.floor((f+.5)*nf/frequencyCount))*nt+Math.min(nt-1,Math.floor((t+.5)*nt/timeCount));
      const attribution=native?pixels.native[cell]:pixels.cam[source],color=heatColors[attribution];
      const alpha=id==='cam'&&mode==='overlay'?opacity*attribution/255:0;
      for(let channel=0;channel<3;channel++)image.data[p+channel]=id==='cam'&&mode==='heatmap'?color[channel]:v*(1-alpha)+color[channel]*alpha;
      image.data[p+3]=255;
    }
    bctx.putImageData(image,0,0);ctx.clearRect(0,0,target.width,target.height);ctx.imageSmoothingEnabled=false;
    ctx.drawImage(buffer,plot.left,plot.top,width,height);ctx.fillStyle='#d3e4e6';ctx.font='12px system-ui';
    ctx.fillText(Math.round(view.fMax)+' Hz',3,14);ctx.fillText(Math.round(view.fMin)+' Hz',3,target.height-36);
    ctx.fillText(view.tMin.toFixed(2)+' s',plot.left,target.height-12);ctx.fillText(view.tMax.toFixed(2)+' s',target.width-65,target.height-12);
    if(gesture?.mode==='zoom'){
      ctx.strokeStyle='#8fffe0';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);ctx.strokeRect(gesture.start.x,gesture.start.y,gesture.current.x-gesture.start.x,gesture.current.y-gesture.start.y);ctx.setLineDash([]);
    }
  }
}
el('zoom').onclick=()=>setView({fMin:Number(el('fmin').value),fMax:Number(el('fmax').value),tMin:Number(el('tmin').value),tMax:Number(el('tmax').value)});
el('resetzoom').onclick=()=>{if(explanation)setView(domain());};
el('lowzoom').onclick=()=>{if(explanation)setView({...view,fMin:domain().fMin,fMax:Math.min(3000,domain().fMax)});};
for(const id of ['camview','nativecells'])el(id).onchange=drawExplanation;
el('opacity').oninput=drawExplanation;
function point(event,canvas){
  const box=canvas.getBoundingClientRect();return {x:(event.clientX-box.left)*canvas.width/box.width,y:(event.clientY-box.top)*canvas.height/box.height};
}
function within(p,canvas){return p.x>=plot.left&&p.x<=canvas.width-plot.right&&p.y>=plot.top&&p.y<=canvas.height-plot.bottom;}
function coordinate(p,canvas,bounds){
  return {time:bounds.tMin+(p.x-plot.left)/(canvas.width-plot.left-plot.right)*(bounds.tMax-bounds.tMin),
    frequency:bounds.fMax-(p.y-plot.top)/(canvas.height-plot.top-plot.bottom)*(bounds.fMax-bounds.fMin)};
}
for(const id of ['spec','cam']){
  const canvas=el(id);
  canvas.onpointerdown=event=>{
    if(!visible||!within(point(event,canvas),canvas))return;
    const p=point(event,canvas);gesture={mode:event.shiftKey?'pan':'zoom',start:p,current:p,bounds:{...visible}};canvas.setPointerCapture(event.pointerId);
  };
  canvas.onpointermove=event=>{
    if(!visible)return;const p=point(event,canvas);
    if(gesture){
      gesture.current={x:Math.max(plot.left,Math.min(canvas.width-plot.right,p.x)),y:Math.max(plot.top,Math.min(canvas.height-plot.bottom,p.y))};
      if(gesture.mode==='pan'){
        const dx=(p.x-gesture.start.x)/(canvas.width-plot.left-plot.right)*(gesture.bounds.tMax-gesture.bounds.tMin);
        const dy=(p.y-gesture.start.y)/(canvas.height-plot.top-plot.bottom)*(gesture.bounds.fMax-gesture.bounds.fMin);
        setView({tMin:gesture.bounds.tMin-dx,tMax:gesture.bounds.tMax-dx,fMin:gesture.bounds.fMin+dy,fMax:gesture.bounds.fMax+dy});
      }else drawExplanation();
    }else if(within(p,canvas)){
      const at=coordinate(p,canvas,visible);let f=0,t=0;
      for(let i=1;i<explanation.frequencies_hz.length;i++)if(Math.abs(explanation.frequencies_hz[i]-at.frequency)<Math.abs(explanation.frequencies_hz[f]-at.frequency))f=i;
      for(let i=1;i<explanation.frame_times_seconds.length;i++)if(Math.abs(explanation.frame_times_seconds[i]-at.time)<Math.abs(explanation.frame_times_seconds[t]-at.time))t=i;
      const index=f*explanation.shape[1]+t;
      el('plotreadout').textContent=at.time.toFixed(3)+' s · '+Math.round(at.frequency)+' Hz · input '+(pixels.spec[index]/255).toFixed(3)+' · interpolated relative attribution '+(pixels.cam[index]/255).toFixed(3);
    }
  };
  canvas.onpointerup=event=>{
    if(!gesture)return;const active=gesture;gesture=null;
    if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);
    if(active.mode==='zoom'&&Math.abs(active.current.x-active.start.x)>8&&Math.abs(active.current.y-active.start.y)>8){
      const a=coordinate(active.start,canvas,active.bounds),b=coordinate(active.current,canvas,active.bounds);
      setView({tMin:Math.min(a.time,b.time),tMax:Math.max(a.time,b.time),fMin:Math.min(a.frequency,b.frequency),fMax:Math.max(a.frequency,b.frequency)});
    }else drawExplanation();
  };
  canvas.onpointercancel=()=>{gesture=null;drawExplanation();};
  canvas.ondblclick=()=>{if(explanation)setView(domain());};
  canvas.addEventListener('wheel',event=>{
    if(!visible||!within(point(event,canvas),canvas))return;event.preventDefault();
    const p=point(event,canvas),at=coordinate(p,canvas,visible),factor=Math.max(.6,Math.min(1.6,Math.exp(event.deltaY*.0015)));
    setView({tMin:at.time+(visible.tMin-at.time)*factor,tMax:at.time+(visible.tMax-at.time)*factor,
      fMin:at.frequency+(visible.fMin-at.frequency)*factor,fMax:at.frequency+(visible.fMax-at.frequency)*factor});
  },{passive:false});
}
load();
</script></html>"""


def make_lab_server(catalog_path, host='127.0.0.1', port=8000):
    catalog_path = Path(catalog_path).resolve()
    catalog = read_json(catalog_path)
    models = {row['name']: load_model(catalog_path.parent / row['bundle'], True) for row in catalog['models']}
    if not models or len(models) != len(catalog['models']):
        raise ValueError('Catalog needs unique model names')
    for model in models.values():
        if model.config.sample_rate != 16000:
            raise ValueError('Lab comparison requires canonical 16kHz models')
        if hasattr(model, 'warmup'):
            model.warmup()
        elif model.encoder is not None:
            model.score(np.zeros(model.config.window_samples, dtype=np.float32))
    lock = threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, payload, status=200, html=False):
            data = payload.encode() if html else json.dumps(payload, allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def do_GET(self):
            if self.path == '/':
                return self.respond(PAGE, html=True)
            if self.path == '/api/models':
                inventory = []
                for name, model in models.items():
                    meta = model.metadata
                    chosen = meta['selection']['chosen']['metrics']
                    is_spectral = meta['schema_version'] == 2
                    shape = [len(axes(model.config)[0]), len(axes(model.config)[1]), 1] if is_spectral else meta['feature_dimension']
                    inventory.append({'name': name, 'model_id': model.model_id,
                        'method': {'spectrogram_cnn': f"2D CNN · {meta.get('architecture', {}).get('parameters', 1781)} parameters", 'rbf_svm': 'RBF SVM · no neural encoder',
                                   'yamnet': 'Frozen YAMNet + logistic regression',
                                   'relative_mel': 'Spectral summaries + logistic regression · no neural encoder'}.get(
                            meta.get('family', meta['features'].get('representation')), 'Experimental classifier'),
                        'shape': str(shape) + (' → per-frequency mean/std' if meta.get('family') == 'rbf_svm' else ''),
                        'gradcam': meta.get('family') == 'spectrogram_cnn',
                        'development': f"{chosen['confusion']['tp']}/{chosen['positive_recordings']}" if 'confusion' in chosen else 'Not recording-level',
                        'false_positives': f"{chosen['confusion']['fp']}/{chosen['negative_recordings']}" if 'confusion' in chosen else 'Not recording-level',
                        'selected_epoch': meta['training'].get('selected_epoch'),
                        'selection_loss': 'development_consistent_max_balanced_loss' if meta['training'].get('negative_loss', '').startswith('softplus(max') else 'development_balanced_bag_loss',
                        'history': meta['training'].get('history', []),
                        'training_windows_seen': meta['training'].get('unique_windows_seen_at_selected_epoch') or sum(r.get('sampled_windows', 0) for r in meta['training'].get('records', [])),
                        'decision': meta['decision'], 'features': meta['features']})
                return self.respond({'models': inventory, 'scope': catalog['scope'],
                    'data_summary': catalog['data_summary'], 'defaults': catalog['defaults']})
            return self.respond({'error': 'Not found'}, 404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path not in {'/api/analyze', '/api/explain'}:
                return self.respond({'error': 'Not found'}, 404)
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                return self.respond({'error': 'Cross-origin upload is not supported'}, 403)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 64 * 1024 * 1024:
                    return self.respond({'error': 'Choose audio up to 64 MB'}, 413)
                raw = self.rfile.read(length)
                info = sf.info(io.BytesIO(raw))
                if not 0 < info.duration <= 120 or info.channels > 16:
                    return self.respond({'error': 'Use nonempty audio up to 2 minutes and 16 channels'}, 422)
                begin = time.perf_counter()
                with tempfile.TemporaryDirectory(prefix='audio-lab-') as directory:
                    path = Path(directory) / 'recording.audio'
                    path.write_bytes(raw)
                    audio, details = decode(path)
                query = parse_qs(parsed.query)
                with lock, threadpool_limits(limits=1):
                    if parsed.path == '/api/analyze':
                        names = query.get('models', [''])[0].split(',')
                        if len(names) > 12 or not all(name in models for name in names):
                            raise ValueError('Select models from the catalog')
                        result = []
                        for name in names:
                            model = models[name]
                            trace = model.batch_trace(audio) if hasattr(model, 'batch_trace') else replay_audio(model, audio)
                            result.append({'name': name, 'model_id': model.model_id, 'trace': trace,
                                'decision': model.metadata['decision'], 'gradcam': model.metadata.get('family') == 'spectrogram_cnn',
                                'alarms': alarm_intervals(trace, details['duration_seconds'], merge_gap=1)})
                        return self.respond({'models': result, 'audio': details,
                            'processing_seconds': time.perf_counter() - begin, 'scope': catalog['scope']})
                    name = query.get('model', [''])[0]
                    if name not in models or models[name].metadata.get('family') != 'spectrogram_cnn':
                        raise ValueError('Select a trained convolutional model for Grad-CAM')
                    model = models[name]
                    start = float(query.get('start', ['0'])[0])
                    if not np.isfinite(start) or not 0 <= start < details['duration_seconds']:
                        raise ValueError('Context start is outside the recording')
                    offset = round(start * model.config.sample_rate)
                    wave = audio[offset:offset + model.config.window_samples]
                    if len(wave) < model.config.min_audio_seconds * model.config.sample_rate:
                        raise ValueError('No classifier score is used for insufficient audio. Choose a longer context.')
                    if np.sqrt(np.mean(wave.astype(np.float64) ** 2)) < model.config.silence_rms:
                        raise ValueError('No classifier score is used for silent input. Choose a non-silent context.')
                    values = model.gradcam(wave, query.get('target', ['drone'])[0], query.get('layer', ['features'])[0])
                    frequencies, times, _ = axes(model.config)
                    response = {k: v for k, v in values.items() if k not in {'spectrogram', 'heatmap', 'native_heatmap'}}
                    for key in ['spectrogram', 'heatmap', 'native_heatmap']:
                        pixels = np.clip(np.rint(values[key] * 255), 0, 255).astype(np.uint8)
                        response[key + '_u8'] = base64.b64encode(pixels.tobytes()).decode()
                    response.update({'name': name, 'model_id': model.model_id, 'shape': list(values['spectrogram'].shape),
                        'frequencies_hz': frequencies.tolist(), 'frame_times_seconds': (times + start).tolist(),
                        'start_seconds': start, 'end_seconds': (offset + len(wave)) / model.config.sample_rate,
                        'padded': len(wave) < model.config.window_samples, 'axis': model.config.axis,
                        'intensity': model.config.intensity, 'n_fft': model.config.n_fft,
                        'encoding': 'row-major frequency × time uint8 display arrays, base64; model input is float32',
                        'scope': 'Grad-CAM attribution to a classifier logit, not a causal explanation or reliability evidence'})
                    return self.respond(response)
            except (ValueError, OSError, RuntimeError, KeyError) as error:
                return self.respond({'error': str(error) if isinstance(error, ValueError) else 'Unable to analyze this audio. Use WAV, FLAC or OGG.'}, 422)

    return ThreadingHTTPServer((host, port), Handler)


def serve_lab(catalog, host='127.0.0.1', port=8000):
    server = make_lab_server(catalog, host, port)
    print(f'Audio model lab running at http://{host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
