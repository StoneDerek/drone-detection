const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {chromium}=require('playwright');

(async()=>{
  const [url='http://127.0.0.1:8010',output='reports/cloudflare-v1/local-browser.json']=process.argv.slice(2);
  const proxy=!new URL(url).hostname.match(/^(localhost|127\.0\.0\.1)$/)&&process.env.HTTPS_PROXY?
    {server:process.env.HTTPS_PROXY}:undefined;
  const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,proxy});
  const errors=[],audioRequests=[];
  const screenshot=output.replace(/\.json$/,'.png');
  fs.mkdirSync(path.dirname(output),{recursive:true});
  try {
    const page=await browser.newPage({viewport:{width:1400,height:1100}});
    page.setDefaultTimeout(180000);
    page.on('pageerror',error=>errors.push(error.message));
    page.on('request',request=>{if(request.method()!=='GET')audioRequests.push({method:request.method(),url:request.url()});});
    await page.goto(url);
    await page.waitForFunction(()=>window.droneLab&&document.querySelectorAll('#models input').length===window.droneLab.catalog.models.length);
    await page.waitForFunction(()=>document.querySelectorAll('#example option').length===5);
    await page.locator('#file').setInputFiles('runs/real-v2/corpus/audio/batear-20260415_192501.wav');
    // Preserve the old known-zero-map regression independently of new defaults.
    await page.locator('#models input').evaluateAll(inputs=>inputs.forEach(input=>{input.checked=['linear256-fft4096','svm-linear-fft4096','yamnet-default'].includes(input.value);}));
    await page.locator('#analyzecurrent').click();
    await page.waitForFunction(()=>!document.querySelector('#analyzecurrent').disabled&&window.audioLabBatch.currentResult);
    const result=await page.evaluate(()=>window.audioLabBatch.currentResult);
    await page.locator('#cnn').selectOption('linear256-fft4096');
    assert.equal(result.models.length,3);
    assert(result.models.some(m=>m.name==='yamnet-default'));
    assert(result.models.some(m=>m.name==='svm-linear-fft4096'));
    await page.waitForFunction(()=>document.querySelector('audio').readyState>=1);
    await page.locator('#player').evaluate(audio=>{audio.currentTime=3.6;});
    await page.waitForFunction(()=>document.querySelector('#decisions').textContent.includes('ALARM'));
    await page.locator('#player').evaluate(audio=>audio.play());
    await page.waitForFunction(()=>document.querySelector('#player').currentTime>3.8);
    await page.locator('#player').evaluate(audio=>audio.pause());
    await page.locator('#layer').selectOption('conv1');
    await page.locator('#explain').click();
    await page.waitForFunction(()=>document.querySelector('#camstatus').textContent.includes('classifier score'));
    const cam=await page.evaluate(()=>window.droneLab.lastExplanation);
    const row=result.models.find(m=>m.name===cam.name).trace.find(r=>r.window_start_seconds===cam.start_seconds);
    assert(row&&Math.abs(row.score-cam.classifier_score)<1e-5);
    assert.equal(cam.has_positive_attribution,false);
    const map=Buffer.from(cam.heatmap_u8,'base64');
    assert.equal(Math.max(...map),0);
    await page.locator('#fmax').fill('2500');
    await page.locator('#zoom').click();
    assert((await page.locator('#spec, #cam').evaluateAll(canvases=>canvases.every(c=>c.toDataURL().length>2000))));
    await page.screenshot({path:screenshot,fullPage:true});
    const camDownload=page.waitForEvent('download');
    await page.locator('#camdownload').click();
    assert.equal((await camDownload).suggestedFilename(),'gradcam-explanation.json');
    const comparisonDownload=page.waitForEvent('download');
    await page.locator('#export').click();
    assert.equal((await comparisonDownload).suggestedFilename(),'audio-model-comparison.json');
    await page.locator('#target').selectOption('non_drone');
    assert(await page.locator('#camdownload').isDisabled());
    await page.locator('#explain').click();
    await page.waitForFunction(()=>window.droneLab.lastExplanation?.target==='non_drone'&&!document.querySelector('#explain').disabled);
    const negative=await page.evaluate(()=>window.droneLab.lastExplanation);
    assert.equal(negative.has_positive_attribution,true);
    assert(Math.abs(negative.classifier_score-cam.classifier_score)<1e-6);
    await page.screenshot({path:screenshot.replace(/\.png$/,'-non-drone.png'),fullPage:true});
    await page.locator('#reviewstart').fill('0');await page.locator('#reviewend').fill('3');
    await page.locator('#reviewlabel').selectOption('uncertain');
    await page.locator('#reviewnote').fill('Software test draft; no listening verification claimed.');
    await page.locator('#savereview').click();
    await page.waitForFunction(()=>document.querySelector('#reviewstatus').textContent.includes('1 draft'));
    const noteEvent=page.waitForEvent('download');await page.locator('#downloadreview').click();
    const noteDownload=await noteEvent;
    const notes=JSON.parse(fs.readFileSync(await noteDownload.path()));
    assert.equal(notes.status,'human_draft');assert.equal(notes.intervals[0].label,'uncertain');
    assert.equal(notes.audio_sha256,'18ef2eb6318f075bd1c978b2c0c16dcb01fd8a6575925703e0fca27f0206395c');
    await page.locator('#example').selectOption('svanstrom-DRONE_001');
    await page.locator('#openexample').click();
    await page.waitForFunction(()=>document.querySelector('#file').files[0]?.name==='svanstrom-DRONE_001.wav');
    assert(await page.locator('#downloadreview').isDisabled());
    await page.locator('#file').setInputFiles({name:'broken.wav',mimeType:'audio/wav',buffer:Buffer.from('broken')});
    await page.locator('#analyzecurrent').click();
    await page.waitForFunction(()=>document.querySelector('#error').textContent.length>0);
    assert(await page.locator('#result').isHidden());
    const diagnosis=await page.request.get(new URL('/diagnosis.html',url).href);
    assert.equal(diagnosis.status(),200);
    assert((await diagnosis.text()).includes('Specific recordings before a general representation'));
    assert.deepEqual(audioRequests,[]);assert.deepEqual(errors,[]);
    const proof={passed:true,url,scope:'Browser functionality on one existing real development file, not an acoustic field test',
      app_js_sha256:crypto.createHash('sha256').update(await (await page.request.get(new URL('/app.js',url).href)).body()).digest('hex'),
      models:result.models.map(m=>({name:m.name,model_id:m.model_id,windows:m.trace.length})),
      playbackAndSeek:true,scoreAndGradcamAgree:true,zeroMapPreserved:true,nonDroneTarget:true,
      frequencyZoom:true,comparisonAndExplanationDownloads:true,examples:true,
      intervalNotesWithAudioHash:true,invalidInputHandled:true,noAudioUploads:true,
      browser:await browser.version(),pageErrors:errors,screenshot};
    fs.writeFileSync(output,JSON.stringify(proof,null,2)+'\n');
    console.log(JSON.stringify(proof));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
