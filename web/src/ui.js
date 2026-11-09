
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
