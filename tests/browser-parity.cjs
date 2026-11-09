const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {chromium} = require('playwright');

(async()=>{
  const [url='http://127.0.0.1:8010', destination='reports/cloudflare-v1/parity.json', fixtures='runs/cloudflare-deployment/parity'] = process.argv.slice(2);
  const directory=path.resolve(fixtures);
  const expected=JSON.parse(fs.readFileSync(path.join(directory,'expected.json')));
  const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true});
  const failures=[];
  const errors=[];
  try {
    const page=await browser.newPage({viewport:{width:1400,height:1100}});
    page.setDefaultTimeout(180000);
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/__fixtures/*',async route=>{
      const name=path.basename(new URL(route.request().url()).pathname);
      await route.fulfill({body:fs.readFileSync(path.join(directory,name)),contentType:'application/octet-stream'});
    });
    await page.goto(url);
    await page.waitForFunction(()=>window.droneLab&&document.querySelectorAll('#models input').length===window.droneLab.catalog.models.length);
    assert.deepEqual((await page.evaluate(()=>window.droneLab.catalog.models.map(m=>m.name))).sort(),Object.keys(expected.cases[0].models).sort());
    const results=[];
    for(const [number,fixture] of expected.cases.entries()) {
      const raw=fs.readFileSync(fixture.source).toString('base64');
      const result=await page.evaluate(async ({fixture,raw,number})=>{
        const {engine,catalog}=window.droneLab;
        const original=Uint8Array.from(atob(raw),c=>c.charCodeAt(0));
        const decoded=await engine.decodeWav(original.buffer);
        const reference=new Float32Array(await (await fetch('/__fixtures/'+fixture.canonical)).arrayBuffer());
        function compare(a,b) {
          if(a.length!==b.length)return {length:[a.length,b.length],max_error:Infinity};
          let maximum=0,total=0;
          for(let i=0;i<a.length;i++){const difference=Math.abs(a[i]-b[i]);maximum=Math.max(maximum,difference);total+=difference;}
          return {length:a.length,max_error:maximum,mean_error:total/a.length};
        }
        const output={id:fixture.id,decode:compare(decoded.samples,reference),models:{},gradcam:{}};
        for(const item of catalog.models) {
          const model=await engine.loadModel(item),expected=fixture.models[item.name];
          let actual;
          if(number===0) actual=(await engine.analyze(decoded.samples,[model]))[0].trace;
          else actual=[{score:await engine.score(decoded.samples.subarray(0,48000),model)}];
          const scoreErrors=actual.map((row,i)=>Math.abs(row.score-expected.expected[i]?.score));
          output.models[item.name]={model_id:model.model_id,windows:actual.length,
            max_score_error:Math.max(...scoreErrors),scores:actual.map(r=>r.score),
            decisions_match:number!==0||actual.every((row,i)=>['alarm','reason','padded','timestamp_seconds','window_start_seconds'].every(k=>row[k]===expected.expected[i]?.[k]))};
          if(expected.image) {
            const image=new Float32Array(await (await fetch('/__fixtures/'+expected.image)).arrayBuffer());
            output.models[item.name].image=compare(engine.spectrogram(decoded.samples.subarray(0,48000),model),image);
          }
        }
        if(number===0) {
          for(const [key,expected] of Object.entries(fixture.gradcam)) {
            const split=key.lastIndexOf('-'),target=key.slice(0,split),layer=key.slice(split+1);
            const model=await engine.loadModel(catalog.models.find(m=>m.name===(expected.model||'linear256-fft4096')));
            const result=engine.gradcam(decoded.samples.subarray(0,48000),model,expected.target||target,expected.layer||layer);
            const heatmap=new Float32Array(await (await fetch('/__fixtures/'+expected.heatmap)).arrayBuffer());
            output.gradcam[key]={...compare(result.heatmap,heatmap),
              score_error:Math.abs(result.classifier_score-expected.score),
              positive_matches:result.has_positive_attribution===expected.positive,
              coarse_shape:result.coarse_shape};
          }
        }
        return output;
      },{fixture,raw,number});
      results.push(result);
      if(result.decode.max_error>1e-5)failures.push(result.id+': resampling '+result.decode.max_error);
      for(const [name,model] of Object.entries(result.models)) {
        const tolerance=name==='yamnet-default'?2e-4:1e-5;
        if(!Number.isFinite(model.max_score_error)||model.max_score_error>tolerance)failures.push(result.id+': '+name+' score '+model.max_score_error);
        if(!model.decisions_match)failures.push(result.id+': '+name+' decisions');
        if(model.image?.max_error>2e-3)failures.push(result.id+': '+name+' image '+model.image.max_error);
      }
      for(const [name,cam] of Object.entries(result.gradcam)) {
        if(cam.max_error>2e-3||!cam.positive_matches||cam.score_error>1e-5)failures.push('Grad-CAM '+name+' '+JSON.stringify(cam));
      }
      console.log(JSON.stringify(result));
    }
    if(errors.length) failures.push(...errors);
    const proof={passed:failures.length===0,scope:expected.scope,
      app_js_sha256:crypto.createHash('sha256').update(await (await page.request.get(new URL('/app.js',url).href)).body()).digest('hex'),
      tolerances:{decoded_audio_max_abs:1e-5,cnn_svm_logistic_score_abs:1e-5,yamnet_score_abs:2e-4,spectrogram_and_gradcam_max_abs:2e-3},
      browser:await browser.version(),results,errors,failures};
    fs.mkdirSync(path.dirname(destination),{recursive:true});
    fs.writeFileSync(destination,JSON.stringify(proof,null,2)+'\n');
    assert.deepEqual(failures,[]);
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
