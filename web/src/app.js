import * as engine from './inference.js';

const fetchAsset=globalThis.fetch.bind(globalThis);
const catalog=await (await fetchAsset('models/catalog.json')).json();
const metadata=new Map();
for(const item of catalog.models) metadata.set(item.name,await (await fetchAsset(item.path)).json());
await engine.initialize();
const audioCache=new WeakMap();

function inventory() {
  return {scope:catalog.scope,data_summary:catalog.data_summary,defaults:catalog.defaults,
    models:catalog.models.map(item=>{
      const m=metadata.get(item.name),metrics=m.development;
      return {...m,method:{spectrogram_cnn:'2D CNN · '+m.architecture?.parameters+' parameters',
        rbf_svm:'RBF SVM · no neural encoder',yamnet:'Frozen YAMNet + logistic regression',
        relative_mel:'Spectral summaries + logistic regression · no neural encoder'}[m.family],
        shape:m.shape?JSON.stringify([...m.shape,1]):m.family==='yamnet'?'1024 embeddings':'96 summary features',
        gradcam:m.family==='spectrogram_cnn',development:metrics.confusion.tp+'/'+metrics.positive_recordings,
        false_positives:metrics.confusion.fp+'/'+metrics.negative_recordings};
    })};
}

async function audio(file) {
  if(!file || file.size>64*1024*1024)throw Error('Choose a WAV file up to 64 MB.');
  if(audioCache.has(file))return audioCache.get(file);
  const bytes=await file.arrayBuffer();
  const audio_sha256=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
  const decoded={file,audio_sha256,...await engine.decodeWav(bytes)};
  audioCache.set(file,decoded);
  return decoded;
}

function bytesBase64(values) {
  const bytes=Uint8Array.from(values,x=>Math.max(0,Math.min(255,Math.round(x*255))));
  let output='';
  for(let i=0;i<bytes.length;i+=16384)output+=String.fromCharCode(...bytes.subarray(i,i+16384));
  return btoa(output);
}

async function request(path,options) {
  const url=new URL(path,location.href),params=url.searchParams;
  if(url.pathname==='/api/models')return inventory();
  const decoded=await audio(options?.body),started=performance.now();
  if(url.pathname==='/api/analyze') {
    const names=params.get('models').split(',');
    if(!names.length||names.some(name=>!metadata.has(name)))throw Error('Select catalog models.');
    const models=[];
    for(const name of names) {
      document.getElementById('progress').textContent='Loading '+name+'…';
      models.push(await engine.loadModel(catalog.models.find(m=>m.name===name)));
    }
    const results=await engine.analyze(decoded.samples,models,(fraction,name)=>{
      document.getElementById('progress').textContent=options.body.name+' · '+name+' · '+Math.round(100*fraction)+'%';
    });
    const result={models:results,audio:decoded.info,processing_seconds:(performance.now()-started)/1000,
      filename:options.body.name,audio_sha256:decoded.audio_sha256,scope:catalog.scope,execution:'In-browser CPU inference; audio stays on this device. Offline comparison, not live capture.'};
    globalThis.droneLab.lastAnalysis=result;
    return result;
  }
  if(url.pathname==='/api/explain') {
    const name=params.get('model'),start=Number(params.get('start'));
    const item=catalog.models.find(m=>m.name===name);
    if(!item||metadata.get(name).family!=='spectrogram_cnn')throw Error('Select a CNN.');
    if(!Number.isFinite(start)||start<0||start>=decoded.info.duration_seconds)throw Error('Invalid context start.');
    const model=await engine.loadModel(item),offset=Math.round(start*16000),wave=decoded.samples.subarray(offset,offset+48000);
    if(wave.length<4000||engine.rms(wave)<1e-5)throw Error('Choose a non-silent context with at least 0.25 seconds of audio.');
    const result=engine.gradcam(wave,model,params.get('target'),params.get('layer'));
    const explanation={name,model_id:model.model_id,shape:model.shape,coarse_shape:result.coarse_shape,
      spectrogram_u8:bytesBase64(result.spectrogram),heatmap_u8:bytesBase64(result.heatmap),
      native_heatmap_u8:bytesBase64(result.native_heatmap),
      classifier_score:result.classifier_score,has_positive_attribution:result.has_positive_attribution,
      target:result.target,layer:result.layer,frequencies_hz:model.frequencies,
      frame_times_seconds:model.frame_times.map(t=>t+start),start_seconds:start,
      end_seconds:(offset+wave.length)/16000,padded:wave.length<48000,
      axis:model.features.axis,intensity:model.features.intensity,n_fft:model.features.n_fft,
      scope:'Grad-CAM for selected CNN logit. Attribution is not evidence of causal acoustic cues or field reliability.'};
    globalThis.droneLab.lastExplanation=explanation;
    return explanation;
  }
  throw Error('Unknown local request.');
}

// Preserve the tested comparison UI; these API-shaped calls execute locally.
// No audio request leaves the browser.
globalThis.fetch=async function(path,options) {
  if(typeof path!=='string'||!path.startsWith('/api/'))return fetchAsset(path,options);
  try{return new Response(JSON.stringify(await request(path,options)),{headers:{'Content-Type':'application/json'}});}
  catch(error){return new Response(JSON.stringify({error:error.message}),{status:422,headers:{'Content-Type':'application/json'}});}
};

// Exposed for reproducible numerical browser tests, without user audio or keys.
globalThis.droneLab={engine,catalog,metadata};
await import('./ui.js');

const examples=await (await fetchAsset('examples/catalog.json')).json();
const exampleFiles=await Promise.all(examples.examples.map(async entry=>{
  const response=await fetchAsset(entry.public_example);
  if(!response.ok)throw Error('Unable to load example '+entry.id);
  return {file:new File([await response.blob()],entry.id+'.wav',{type:'audio/wav'}),entry};
}));
globalThis.audioLabBatch.addFiles(exampleFiles.map(item=>item.file),exampleFiles.map(item=>item.entry),false);
const element=id=>document.getElementById(id);
element('example').replaceChildren();
for(const entry of examples.examples) {
  const option=document.createElement('option');option.value=entry.id;
  option.textContent=entry.id+' · publisher '+entry.publisher_label;
  element('example').append(option);
}
element('openexample').onclick=async()=>{
  const button=element('openexample');button.disabled=true;
  try {
    const entry=examples.examples.find(item=>item.id===element('example').value);
    globalThis.audioLabBatch.selectFile(exampleFiles.find(item=>item.entry.id===entry.id).file);
    element('exampleinfo').textContent=entry.id+' · '+entry.license+' · excerpt '+entry.excerpt_start_seconds.toFixed(2)+'–'+
      (entry.excerpt_start_seconds+entry.excerpt_duration_seconds).toFixed(2)+' s. Publisher class: '+entry.publisher_label+'. Listen before treating this as an audible label.';
  }catch(error){element('exampleinfo').textContent=error.message;}
  finally{button.disabled=false;}
};

let notes=[];
const notesByFile=new WeakMap();
element('file').addEventListener('labrecordingchange',()=>{
  notes=notesByFile.get(element('file').files[0])||[];
  element('reviewitems').replaceChildren();element('reviewstatus').textContent=notes.length?notes.length+' draft interval note(s)':'';
  for(const note of notes){const row=document.createElement('li');row.textContent=note.start_seconds.toFixed(2)+'–'+note.end_seconds.toFixed(2)+' s · '+note.label+' · '+note.note;element('reviewitems').append(row);}
  element('downloadreview').disabled=!notes.length;element('reviewnote').value='';
});
element('savereview').onclick=async()=>{
  try {
    const file=element('file').files[0],decoded=await audio(file);
    const start=Number(element('reviewstart').value),end=Number(element('reviewend').value);
    if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>decoded.info.duration_seconds)
      throw Error('Choose an interval within the recording.');
    notes.push({start_seconds:start,end_seconds:end,label:element('reviewlabel').value,note:element('reviewnote').value});
    notesByFile.set(file,notes);
    const row=document.createElement('li');row.textContent=start.toFixed(2)+'–'+end.toFixed(2)+' s · '+notes.at(-1).label+' · '+notes.at(-1).note;
    element('reviewitems').append(row);element('reviewstatus').textContent=notes.length+' draft interval note(s)';
    element('downloadreview').disabled=false;
  }catch(error){element('reviewstatus').textContent=error.message;}
};
element('downloadreview').onclick=async()=>{
  const file=element('file').files[0];
  if(!file||!notes.length)return;
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer()))]
    .map(x=>x.toString(16).padStart(2,'0')).join('');
  const payload={schema_version:1,status:'human_draft',filename:file.name,audio_sha256:hash,
    label_scope:'User listening notes; not independently verified and not used for training',intervals:notes};
  const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='audio-listening-notes.json';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
