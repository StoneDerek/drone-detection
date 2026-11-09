const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');

(async () => {
  const [url, audioPath, screenshot, resultFile] = process.argv.slice(2);
  assert(url && audioPath && screenshot && resultFile, 'Provide URL, audio file, screenshot and output JSON');
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1400, height: 1100 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(() => document.querySelectorAll('#models input').length >= 2);
    const catalog = await page.evaluate(() => JSON.parse(JSON.stringify(catalog)));
    assert(catalog.models.some(m => m.gradcam));
    assert(catalog.models.some(m => /no neural encoder/.test(m.method)));
    await page.locator('#file').setInputFiles(audioPath);
    const response = page.waitForResponse(r => r.url().includes('/api/analyze?'), { timeout: 180000 });
    await page.locator('#analyze').click();
    assert.equal((await response).status(), 200);
    await page.waitForFunction(() => !document.querySelector('#result').hidden);
    const result = await page.evaluate(() => JSON.parse(JSON.stringify(result)));
    assert(result.models.length >= 2);
    assert(result.models.some(m => m.gradcam));
    assert(result.models.some(m => !m.gradcam));
    await page.waitForFunction(() => document.querySelector('audio').readyState >= 1);
    await page.locator('audio').evaluate(audio => { audio.currentTime = 3.6; });
    await page.waitForFunction(() => document.querySelector('#decisions').textContent.includes('No alarm') || document.querySelector('#decisions').textContent.includes('ALARM'));
    await page.locator('audio').evaluate(audio => audio.play());
    await page.waitForFunction(() => document.querySelector('audio').currentTime > 3.8);
    await page.locator('audio').evaluate(audio => audio.pause());
    await page.locator('#layer').selectOption('conv1');
    const explainResponse = page.waitForResponse(r => r.url().includes('/api/explain?'), { timeout: 60000 });
    await page.locator('#explain').click();
    assert.equal((await explainResponse).status(), 200);
    await page.waitForFunction(() => document.querySelector('#camstatus').textContent.includes('classifier score'));
    const explanation = await page.evaluate(() => JSON.parse(JSON.stringify(explanation)));
    assert.equal(explanation.layer, 'conv1');
    assert((await page.locator('#identity').textContent()).includes(explanation.model_id));
    assert(Buffer.from(explanation.heatmap_u8, 'base64').length === explanation.shape[0] * explanation.shape[1]);
    const model = result.models.find(m => m.name === explanation.name);
    const row = model.trace.find(r => Math.abs(r.window_start_seconds - explanation.start_seconds) < 1e-8);
    assert(row);
    assert(Math.abs(row.score - explanation.classifier_score) < 1e-5);
    await page.locator('#fmax').fill('2500');
    await page.locator('#zoom').click();
    const canvases = await page.locator('#spec, #cam').evaluateAll(items => items.map(canvas => canvas.toDataURL().length));
    assert(canvases.every(size => size > 2000));
    assert(await page.locator('#learning').evaluate(canvas => canvas.toDataURL().length > 2000));
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#camdownload').click();
    assert.equal((await downloadEvent).suggestedFilename(), 'gradcam-explanation.json');
    const comparisonDownload = page.waitForEvent('download');
    await page.locator('#export').click();
    assert.equal((await comparisonDownload).suggestedFilename(), 'audio-model-comparison.json');
    await page.screenshot({ path: screenshot, fullPage: true });
    await page.locator('#target').selectOption('non_drone');
    assert(await page.locator('#camdownload').isDisabled());
    assert.match(await page.locator('#camstatus').textContent(), /Compute Grad-CAM/);
    const negativeResponse = page.waitForResponse(r => r.url().includes('/api/explain?'), { timeout: 60000 });
    await page.locator('#explain').click();
    assert.equal((await negativeResponse).status(), 200);
    await page.waitForFunction(() => document.querySelector('#camstatus').textContent.includes('classifier score'));
    const negativeExplanation = await page.evaluate(() => JSON.parse(JSON.stringify(explanation)));
    assert.equal(negativeExplanation.target, 'non_drone');
    assert.equal(negativeExplanation.model_id, explanation.model_id);
    assert(Math.abs(negativeExplanation.classifier_score - explanation.classifier_score) < 1e-6);
    await page.screenshot({ path: screenshot.replace(/\.png$/, '-non-drone.png'), fullPage: true });
    await page.locator('#file').setInputFiles({ name: 'broken.wav', mimeType: 'audio/wav', buffer: Buffer.from('invalid') });
    await page.locator('#analyze').click();
    await page.waitForFunction(() => document.querySelector('#error').textContent.length > 0);
    assert(await page.locator('#result').isHidden());
    assert.deepEqual(errors, []);
    const proof = { passed: true, scope: 'Product software check on one existing real development input; not acoustic validation',
      models: result.models.map(m => ({ name: m.name, model_id: m.model_id, windows: m.trace.length })),
      uploadComparison: true, playbackAndSeek: true, genuineGradcamAPI: true, explanationMatchesScore: true,
      gradcam: { shape: explanation.shape, coarse_shape: explanation.coarse_shape, target: explanation.target,
        layer: explanation.layer, has_positive_attribution: explanation.has_positive_attribution },
      nonDroneTarget: { passed: true, has_positive_attribution: negativeExplanation.has_positive_attribution },
      frequencyZoom: true, trainingCurveVisible: true, changedTargetClearsOldExplanation: true,
      comparisonAndExplanationDownloads: true, invalidAudioActionable: true,
      browser: await browser.version(), pageErrors: errors };
    fs.writeFileSync(resultFile, JSON.stringify(proof, null, 2) + '\n');
    console.log(JSON.stringify(proof));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
