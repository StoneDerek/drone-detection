const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {chromium}=require('playwright');

(async()=>{
  const [url='http://127.0.0.1:8012',output='reports/cloudflare-v2/interactions.json']=process.argv.slice(2);
  const proxy=!new URL(url).hostname.match(/^(localhost|127\.0\.0\.1)$/)&&process.env.HTTPS_PROXY?{server:process.env.HTTPS_PROXY}:undefined;
  const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,proxy});
  const errors=[],uploads=[];fs.mkdirSync(path.dirname(output),{recursive:true});
  try{
    const page=await browser.newPage({viewport:{width:1500,height:1100}});page.setDefaultTimeout(180000);
    page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()!=='GET')uploads.push(r.url());});
    await page.goto(url);await page.waitForFunction(()=>window.audioLabBatch?.recordings.length===5);
    const cnnCount=await page.evaluate(()=>window.droneLab.catalog.models.filter(m=>m.family==='spectrogram_cnn').length);
    assert(cnnCount>=8);assert.equal(await page.locator('#cnn option:not([disabled])').count(),cnnCount);
    await page.locator('#file').setInputFiles(['runs/real-v2/corpus/audio/batear-20260415_192501.wav','runs/real-v2/corpus/audio/svanstrom-HELICOPTER_001.wav']);
    await page.waitForFunction(()=>window.audioLabBatch.recordings.length===7);
    await page.locator('#models input').evaluateAll(inputs=>inputs.forEach(input=>{input.checked=input.value==='linear256-fft4096';}));
    await page.locator('#analyze').click();
    await page.waitForFunction(()=>document.querySelector('#progress').textContent.startsWith('Comparison complete for 7')&&!document.querySelector('#analyze').disabled);
    assert.equal(await page.locator('#batchrows tr').count(),7);
    assert.equal(await page.locator('#batchrows canvas').count(),7);
    const original=await page.evaluate(()=>window.audioLabBatch.currentResult);
    await page.locator('#recording').selectOption('0');
    assert(await page.locator('#explanation').isVisible());
    await page.locator('#recording').selectOption('5');
    const restored=await page.evaluate(()=>window.audioLabBatch.currentResult);
    assert.deepEqual(restored,original);
    await page.locator('#window').selectOption('1.5');
    await page.locator('#target').selectOption('non_drone');
    await page.locator('#layer').selectOption('features');
    async function explain(name){
      await page.locator('#cnn').selectOption(name);
      assert.equal(await page.locator('#window').inputValue(),'1.5');
      await page.locator('#explain').click();
      await page.waitForFunction(name=>window.droneLab.lastExplanation?.name===name&&window.droneLab.lastExplanation?.start_seconds===1.5&&!document.querySelector('#explain').disabled,name);
      return page.evaluate(()=>window.droneLab.lastExplanation);
    }
    const first=await explain('linear-fft4096');
    const second=await explain('frequency-context-fft4096');
    assert.equal(first.spectrogram_u8,second.spectrogram_u8);
    assert.notEqual(first.model_id,second.model_id);
    const coarse=await explain('linear256-fft4096');
    assert.deepEqual(coarse.coarse_shape,[8,22]);
    assert.equal(Buffer.from(coarse.native_heatmap_u8,'base64').length,8*22);
    assert(coarse.has_positive_attribution);
    assert.equal(await page.locator('#camview').inputValue(),'heatmap');
    await page.locator('#camview').selectOption('overlay');
    await page.locator('#opacity').evaluate(input=>{input.value='0';input.dispatchEvent(new Event('input'));});
    const transparent=await page.evaluate(()=>['spec','cam'].map(id=>document.getElementById(id).toDataURL()));
    assert.equal(transparent[0],transparent[1]);
    await page.locator('#opacity').evaluate(input=>{input.value='.8';input.dispatchEvent(new Event('input'));});
    const overlaid=await page.evaluate(()=>['spec','cam'].map(id=>document.getElementById(id).toDataURL()));
    assert.notEqual(overlaid[0],overlaid[1]);
    await page.locator('#camview').selectOption('heatmap');
    const native=await page.locator('#cam').evaluate(canvas=>canvas.toDataURL());
    await page.locator('#nativecells').uncheck();
    assert.notEqual(await page.locator('#cam').evaluate(canvas=>canvas.toDataURL()),native);
    await page.locator('#nativecells').check();
    async function bounds(){return page.evaluate(()=>Object.fromEntries(['fmin','fmax','tmin','tmax'].map(id=>[id,Number(document.getElementById(id).value)])));}
    const before=await bounds();
    await page.locator('#spec').scrollIntoViewIfNeeded();const box=await page.locator('#spec').boundingBox();
    const at=(x,y)=>({x:box.x+box.width*x,y:box.y+box.height*y});
    let a=at(.3,.25),b=at(.75,.7);
    await page.mouse.move(a.x,a.y);await page.mouse.down();await page.mouse.move(b.x,b.y,{steps:5});await page.mouse.up();
    const dragged=await bounds();
    assert(dragged.fmax-dragged.fmin<(before.fmax-before.fmin)*.7);
    assert(dragged.tmax-dragged.tmin<(before.tmax-before.tmin)*.7);
    a=at(.55,.45);await page.mouse.move(a.x,a.y);await page.mouse.wheel(0,-180);
    await page.waitForFunction(span=>Number(document.querySelector('#tmax').value)-Number(document.querySelector('#tmin').value)<span,dragged.tmax-dragged.tmin);
    const wheeled=await bounds();
    await page.keyboard.down('Shift');await page.mouse.down();await page.mouse.move(a.x+25,a.y+10,{steps:4});await page.mouse.up();await page.keyboard.up('Shift');
    const panned=await bounds();assert.notEqual(panned.tmin,wheeled.tmin);
    assert(Math.abs((panned.tmax-panned.tmin)-(wheeled.tmax-wheeled.tmin))<.003);
    await page.locator('#explanation').screenshot({path:output.replace(/\.json$/,'.png')});
    await page.locator('#resetzoom').click();
    const reset=await bounds();assert(Math.abs(reset.tmin-coarse.frame_times_seconds[0])<.001);
    assert(Math.abs(reset.fmax-coarse.frequencies_hz.at(-1))<.001);
    const mel=await explain('mel64-fft4096');assert.equal(mel.shape[0],64);
    assert.equal(mel.start_seconds,1.5);
    const event=page.waitForEvent('download');await page.locator('#batchdownload').click();const downloaded=await event;
    const batch=JSON.parse(fs.readFileSync(await downloaded.path()));assert.equal(batch.recordings.length,7);
    assert(batch.recordings.every(entry=>entry.comparison.models.length===1&&entry.comparison.audio_sha256.length===64));
    const added=await page.evaluate(()=>window.droneLab.catalog.models.filter(m=>
      m.name.startsWith('coherent-')||m.name==='coverage8'||m.name.startsWith('medium-balanced-')).map(m=>m.name));
    const newExplanations=[];
    for(const name of added){
      const explanation=await explain(name);
      assert.deepEqual(explanation.shape,[256,172]);assert.deepEqual(explanation.coarse_shape,[15,22]);
      assert.equal(explanation.spectrogram_u8,coarse.spectrogram_u8);
      newExplanations.push({name,model_id:explanation.model_id,coarse_shape:explanation.coarse_shape,score:explanation.classifier_score});
    }
    if(added.includes('coherent-frequency-seed42')){
      await page.locator('#models input').evaluateAll(inputs=>inputs.forEach(input=>{input.checked=input.value==='coherent-frequency-seed42';}));
      await page.locator('#analyzecurrent').click();
      await page.waitForFunction(()=>window.audioLabBatch.currentResult?.models[0]?.name==='coherent-frequency-seed42'&&!document.querySelector('#analyzecurrent').disabled);
      await page.locator('#window').selectOption('1.5');
      const explanation=await explain('coherent-frequency-seed42');
      const trace=await page.evaluate(()=>window.audioLabBatch.currentResult.models[0].trace);
      assert(Math.abs(trace.find(r=>r.window_start_seconds===1.5).score-explanation.classifier_score)<1e-5);
      assert((await page.locator('#learningmetric').textContent()).includes('same maximum-context'));
      await page.locator('#explanation').screenshot({path:output.replace(/\.json$/,'-new-cnn.png')});
    }
    assert.deepEqual(errors,[]);assert.deepEqual(uploads,[]);
    const app=await (await page.request.get(new URL('/app.js',url).href)).body();
    const proof={passed:true,url,scope:'UI checks on existing real examples and one development upload; no new acoustic benchmark',
      app_js_sha256:crypto.createHash('sha256').update(app).digest('hex'),
      automaticExamplesLoaded:5,multipleFileUpload:true,batchRecordings:7,persistentRecordingResults:true,allCatalogCNNsSelectable:true,cnnCount,
      unscoredCNNExplanation:true,contextPreservedAcrossModels:true,identicalPreprocessingShowsIdenticalInput:true,
      nativeMapShape:coarse.coarse_shape,opacityZeroMatchesInput:true,heatmapAndOverlay:true,nativeAndInterpolatedMaps:true,
      linkedRectangleZoom:true,wheelZoom:true,shiftDragPan:true,resetZoom:true,hashedBatchExport:true,
      bounds:{before,dragged,wheeled,panned,reset},noAudioUploads:true,pageErrors:errors,browser:await browser.version()};
    proof.newModelExplanations=newExplanations;
    fs.writeFileSync(output,JSON.stringify(proof,null,2)+'\n');console.log(JSON.stringify(proof));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
