import * as tf from '@tensorflow/tfjs';
import FFT from 'fft.js';

const loaded = new Map();
let yamnet;
let resamplers;
const sigmoid = x => 1 / (1 + Math.exp(-x));
export const rms = wave => Math.sqrt(wave.reduce((a,b) => a+b*b,0)/wave.length);

export async function initialize() {
  // CPU gives a reproducible float32 baseline and supports the Grad-CAM path.
  await tf.setBackend('cpu');
  await tf.ready();
}

export async function loadModel(item) {
  if (loaded.has(item.name)) return loaded.get(item.name);
  const info = await (await fetch(item.path)).json();
  const base = item.path.slice(0,item.path.lastIndexOf('/')+1);
  info.values = {};
  for (const [key,spec] of Object.entries(info.arrays)) {
    const shards=[];
    for(const path of spec.paths) shards.push(new Uint8Array(await (await fetch(base+path)).arrayBuffer()));
    const combined=new Uint8Array(shards.reduce((sum,shard)=>sum+shard.length,0));
    let offset=0;for(const shard of shards){combined.set(shard,offset);offset+=shard.length;}
    const bytes=combined.buffer;
    const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))]
      .map(x=>x.toString(16).padStart(2,'0')).join('');
    if(hash!==spec.sha256) throw Error('Model asset checksum mismatch: '+item.name);
    info.values[key] = spec.dtype==='float32' ? new Float32Array(bytes) : new Float64Array(bytes);
  }
  if(info.family==='spectrogram_cnn') info.weights = Object.keys(info.arrays)
    .sort((a,b)=>Number(a.slice(1))-Number(b.slice(1)))
    .map(key=>tf.tensor(info.values[key],info.arrays[key].shape,'float32'));
  loaded.set(item.name,info);
  return info;
}

export async function decodeWav(buffer) {
  const view = new DataView(buffer);
  const tag = offset => String.fromCharCode(...new Uint8Array(buffer,offset,4));
  if(buffer.byteLength<44 || tag(0)!=='RIFF' || tag(8)!=='WAVE')
    throw Error('Choose a PCM or float WAV recording.');
  let format, data;
  for(let offset=12;offset+8<=buffer.byteLength;) {
    const length=view.getUint32(offset+4,true), start=offset+8;
    if(start+length>buffer.byteLength) throw Error('Truncated WAV file.');
    if(tag(offset)==='fmt ') {
      if(length<16) throw Error('Invalid WAV format.');
      format={code:view.getUint16(start,true),channels:view.getUint16(start+2,true),
        rate:view.getUint32(start+4,true),block:view.getUint16(start+12,true),bits:view.getUint16(start+14,true)};
      if(format.code===65534 && length>=40) format.code=view.getUint16(start+24,true);
    }
    if(tag(offset)==='data') data={start,length};
    offset=start+length+(length%2);
  }
  if(!format||!data||![1,3].includes(format.code)||format.channels<1||format.channels>16||
     ![8,16,24,32,64].includes(format.bits)||format.block!==format.channels*format.bits/8||
     (format.code===3&&![32,64].includes(format.bits))||(format.code===1&&format.bits===64))
    throw Error('Unsupported WAV encoding. Use PCM 16/24/32 or float32 WAV.');
  const count=Math.floor(data.length/format.block);
  if(!count||count/format.rate>120) throw Error('Choose nonempty audio up to 2 minutes.');
  const mono=new Float32Array(count);
  let peak=0,clipped=0;
  for(let i=0;i<count;i++) {
    let sum=0;
    for(let c=0;c<format.channels;c++) {
      const at=data.start+i*format.block+c*format.bits/8;
      let value;
      if(format.code===3) value=format.bits===32?view.getFloat32(at,true):view.getFloat64(at,true);
      else if(format.bits===8) value=(view.getUint8(at)-128)/128;
      else if(format.bits===16) value=view.getInt16(at,true)/32768;
      else if(format.bits===24) value=(view.getUint8(at)|(view.getUint8(at+1)<<8)|(view.getInt8(at+2)<<16))/8388608;
      else value=view.getInt32(at,true)/2147483648;
      value=Math.fround(value);
      if(!Number.isFinite(value)) throw Error('Nonfinite audio samples.');
      sum+=value;peak=Math.max(peak,Math.abs(value));clipped+=Math.abs(value)>=.999;
    }
    mono[i]=sum/format.channels;
  }
  const samples=await resample(mono,format.rate);
  return {samples,info:{original_sample_rate:format.rate,channels:format.channels,
    canonical_sample_rate:16000,duration_seconds:count/format.rate,peak,
    clipped_fraction:clipped/(count*format.channels),rms:rms(samples)}};
}

export async function resample(wave,rate) {
  if(rate===16000) return wave;
  if(!resamplers) resamplers=await (await fetch('models/resamplers.json')).json();
  const filter=resamplers[rate];
  if(!filter) throw Error('Supported WAV sample rates: 8, 11.025, 16, 22.05, 24, 32, 44.1, 48, 88.2, 96 and 192 kHz.');
  const {up,down,prepad,remove,weights}=filter;
  const output=new Float32Array(Math.ceil(wave.length*up/down));
  for(let i=0;i<output.length;i++) {
    const center=(i+remove)*down-prepad;
    const first=Math.max(0,Math.ceil((center-weights.length+1)/up));
    const last=Math.min(wave.length-1,Math.floor(center/up));
    let sum=0;
    // upfirdn accumulates its FIR in float32, in input sample order.
    for(let k=first;k<=last;k++) sum=Math.fround(sum+Math.fround(wave[k]*weights[center-k*up]));
    output[i]=sum;
  }
  return output;
}

function rawPower(wave,config) {
  const n=config.n_fft, hop=config.stft_hop, frames=1+Math.floor((48000-n)/hop);
  const bins=n/2+1, fft=new FFT(n), input=new Float64Array(n), spectrum=fft.createComplexArray();
  const hann=Float64Array.from({length:n},(_,i)=>Math.fround(.5-.5*Math.cos(2*Math.PI*i/n)));
  const power=new Float64Array(bins*frames), factor=2/n;
  for(let t=0;t<frames;t++) {
    for(let i=0;i<n;i++) input[i]=Math.fround((wave[t*hop+i]||0)*hann[i]);
    fft.realTransform(spectrum,input);
    for(let f=0;f<bins;f++) {
      const re=Math.fround(spectrum[2*f]*factor), im=Math.fround(spectrum[2*f+1]*factor);
      // scipy abs(complex64) rounds the magnitude before the float64 square.
      const magnitude=Math.fround(Math.hypot(re,im));
      power[f*frames+t]=magnitude*magnitude;
    }
  }
  return {power,bins,frames};
}

function filterbank(config) {
  const mel=hz=>2595*Math.log10(1+hz/700), hz=m=>700*(10**(m/2595)-1);
  const count=config.mel_bins, low=mel(config.f_min), high=mel(config.f_max), bins=config.n_fft/2+1;
  const edges=Array.from({length:count+2},(_,i)=>hz(low+(high-low)*i/(count+1)));
  const indptr=[0],indices=[],data=[];
  for(let i=0;i<count;i++) {
    let sum=0;
    const weights=[];
    for(let f=0;f<bins;f++) {
      const frequency=f*16000/config.n_fft;
      const w=Math.max(0,Math.min((frequency-edges[i])/(edges[i+1]-edges[i]),(edges[i+2]-frequency)/(edges[i+2]-edges[i+1])));
      if(w>0) {weights.push([f,w]);sum+=w;}
    }
    for(const [f,w] of weights) {indices.push(f);data.push(w/Math.max(sum,1e-12));}
    indptr.push(indices.length);
  }
  return {indptr,indices,data};
}

function bands(raw,bank,count) {
  const output=new Float64Array(count*raw.frames);
  for(let f=0;f<count;f++) for(let k=bank.indptr[f];k<bank.indptr[f+1];k++) {
    const bin=bank.indices[k], weight=bank.data[k];
    for(let t=0;t<raw.frames;t++) output[f*raw.frames+t]+=weight*raw.power[bin*raw.frames+t];
  }
  return output;
}

export function spectrogram(wave,model) {
  const config=model.features, raw=rawPower(wave,config);
  let power;
  if(model.bank) power=bands(raw,model.bank,config.mel_bins);
  else {
    const first=Math.ceil(config.f_min*config.n_fft/16000),last=Math.floor(config.f_max*config.n_fft/16000);
    power=raw.power.subarray(first*raw.frames,(last+1)*raw.frames);
  }
  let peak=0;
  for(const x of power) peak=Math.max(peak,x);
  const image=new Float32Array(power.length);
  if(peak<=1e-30) return image;
  for(let i=0;i<power.length;i++) {
    const relative=power[i]/peak;
    image[i]=Math.max(0,Math.min(1,config.intensity==='log_power'?
      1+10*Math.log10(Math.max(relative,1e-30))/config.dynamic_range_db:relative));
  }
  return image;
}

function descriptors(image,[frequency,frames]) {
  const result=new Float64Array(2*frequency);
  for(let f=0;f<frequency;f++) {
    let mean=0,variance=0;
    for(let t=0;t<frames;t++) mean+=image[f*frames+t];
    mean=Math.fround(mean/frames);
    for(let t=0;t<frames;t++) variance+=Math.fround((image[f*frames+t]-mean)**2);
    result[f]=mean;result[f+frequency]=Math.fround(Math.sqrt(variance/frames));
  }
  return result;
}

function convolution(model,x,index) {
  return tf.conv2d(x,model.weights[2*index],model.architecture.convolutions[index].stride,'same')
    .add(model.weights[2*index+1]).relu();
}
function convCount(model) {return model.architecture.variant==='frequency_context'?4:3;}
function tail(model,x,next=0) {
  const count=convCount(model);
  for(let i=next;i<count;i++) x=convolution(model,x,i);
  x=model.architecture.variant==='frequency_aware'?x.mean(2).reshape([x.shape[0],-1]):x.mean([1,2]);
  x=x.matMul(model.weights[2*count]).add(model.weights[2*count+1]);
  if(['frequency_aware','global_readout'].includes(model.architecture.variant))
    x=x.relu().matMul(model.weights[2*count+2]).add(model.weights[2*count+3]);
  return x;
}

function linearHead(vector,model) {
  const {mean,scale,coef,intercept}=model.values;
  let logit=intercept[0];
  for(let i=0;i<vector.length;i++) logit+=(vector[i]-mean[i])/scale[i]*coef[i];
  return sigmoid(logit);
}

export async function score(wave,model) {
  if(model.family==='yamnet') {
    if(!yamnet) yamnet=await tf.loadGraphModel('models/yamnet/model.json');
    const input=tf.tensor1d(Float32Array.from({length:48000},(_,i)=>wave[i]||0));
    const embeddings=await yamnet.executeAsync(input,'Identity_1');
    const average=embeddings.mean(0),vector=await average.data();
    input.dispose();embeddings.dispose();average.dispose();
    return linearHead(vector,model);
  }
  if(model.family==='relative_mel') {
    const raw=rawPower(wave,model.features);
    if(!model.relativeBank) model.relativeBank=filterbank(model.features);
    const mel=bands(raw,model.relativeBank,model.features.mel_bins);
    let peak=0;for(const x of mel)peak=Math.max(peak,x);
    const floor=Math.max(peak*1e-10,Number.MIN_VALUE),log=Float64Array.from(mel,x=>10*Math.log10(Math.max(x,floor)));
    const average=log.reduce((a,b)=>a+b,0)/log.length,vector=new Float64Array(2*model.features.mel_bins);
    for(let f=0;f<model.features.mel_bins;f++) {
      let mean=0,variance=0;
      for(let t=0;t<raw.frames;t++)mean+=log[f*raw.frames+t]-average;
      mean/=raw.frames;
      for(let t=0;t<raw.frames;t++)variance+=(log[f*raw.frames+t]-average-mean)**2;
      vector[f]=mean;vector[f+model.features.mel_bins]=Math.sqrt(variance/raw.frames);
    }
    return linearHead(vector,model);
  }
  const image=spectrogram(wave,model);
  if(model.family==='spectrogram_cnn') return tf.tidy(()=>
    sigmoid(tail(model,tf.tensor4d(image,[1,...model.shape,1])).dataSync()[0]));
  const vector=descriptors(image,model.shape),{mean,scale,support_vectors,dual_coef,intercept}=model.values;
  for(let i=0;i<vector.length;i++) vector[i]=(vector[i]-mean[i])/scale[i];
  let margin=intercept[0];
  for(let row=0;row<dual_coef.length;row++) {
    let distance=0;
    for(let i=0;i<vector.length;i++) distance+=(vector[i]-support_vectors[row*vector.length+i])**2;
    margin+=Math.exp(-model.gamma*distance)*dual_coef[row];
  }
  return sigmoid(margin);
}

export function gradcam(wave,model,target='drone',layer='features') {
  if(!['drone','non_drone'].includes(target)||!['conv1','conv2','features'].includes(layer)) throw Error('Invalid explanation selection.');
  const image=spectrogram(wave,model),index=layer==='conv1'?0:layer==='conv2'?1:convCount(model)-1;
  return tf.tidy(()=>{
    let activation=tf.tensor4d(image,[1,...model.shape,1]);
    for(let i=0;i<=index;i++) activation=convolution(model,activation,i);
    const gradient=tf.grad(a=>tail(model,a,index+1).sum().mul(target==='drone'?1:-1))(activation);
    const coarse=activation.mul(gradient.mean([1,2],true)).sum(-1).relu().squeeze([0]);
    const maximum=coarse.max().dataSync()[0],positive=maximum>1e-15;
    const normalized=positive?coarse.div(maximum):tf.zerosLike(coarse);
    const heatmap=tf.image.resizeBilinear(normalized.expandDims(-1),model.shape,false,true).squeeze([-1]);
    return {spectrogram:image,heatmap:Float32Array.from(heatmap.dataSync()),
      native_heatmap:Float32Array.from(normalized.dataSync()),
      classifier_score:sigmoid(tail(model,activation,index+1).dataSync()[0]),
      coarse_shape:coarse.shape,has_positive_attribution:positive,target,layer};
  });
}

export async function analyze(samples,models,progress=()=>{}) {
  const starts=[];
  for(let s=0;s+48000<=samples.length;s+=8000)starts.push(s);
  if(!starts.length) starts.push(0);
  else if(starts.at(-1)+48000<samples.length) starts.push(starts.at(-1)+8000);
  const result=[];
  let finished=0;
  for(const model of models) {
    let active=false,hits=0,misses=0;
    const trace=[],d=model.decision;
    for(const start of starts) {
      const wave=samples.subarray(start,start+48000),level=rms(wave);
      const reason=wave.length<4000?'insufficient_audio':level<1e-5?'silence':'scored';
      const value=reason==='insufficient_audio'?null:reason==='silence'?0:await score(wave,model);
      if(value!==null) {
        if(!active) {hits=value>=d.on_threshold?hits+1:0;if(hits>=d.on_hits){active=true;hits=0;}}
        else {misses=value<d.off_threshold?misses+1:0;if(misses>=d.off_hits){active=false;misses=0;}}
      }
      trace.push({window_start_seconds:start/16000,timestamp_seconds:(start+wave.length)/16000,
        score:value,reason,alarm:active,rms:level,padded:wave.length<48000,
        end_of_file:wave.length<48000,model_id:model.model_id,
        compute_seconds:null,delivered_at_seconds:null,processing_mode:'offline browser comparison'});
      progress(++finished/(starts.length*models.length),model.name);
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    result.push({name:model.name,model_id:model.model_id,trace,decision:d,gradcam:model.family==='spectrogram_cnn'});
  }
  return result;
}

export const memory = ()=>tf.memory();
