const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');

(async () => {
  const [url, fixture, screenshot, resultFile] = process.argv.slice(2);
  assert(url && fixture && screenshot && resultFile, 'Provide URL, fixture JSON, screenshot and result JSON');
  const inputs = JSON.parse(fs.readFileSync(fixture, 'utf8'));
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1160, height: 1100 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(() => document.querySelector('#identity').textContent.length === 64);
    assert.match(await page.locator('#scope').textContent(), /Real-audio research prototype/);
    const checks = [];
    for (const [kind, path] of [['positive', inputs.positive_path], ['negative', inputs.negative_path]]) {
      await page.locator('#file').setInputFiles(path);
      const response = page.waitForResponse(r => r.url().endsWith('/api/analyze') && r.request().method() === 'POST', { timeout: 120000 });
      await page.locator('#analyze').click();
      const api = await response;
      assert.equal(api.status(), 200);
      await page.waitForFunction(() => !document.querySelector('#result').hidden);
      const result = await page.evaluate(() => JSON.parse(JSON.stringify(result)));
      assert.equal(await page.locator('#rows tr').count(), result.trace.length);
      await page.waitForFunction(() => document.querySelector('audio').readyState >= 1);
      const target = kind === 'positive' ? result.alarms[0][0] + .02 : 4;
      await page.locator('audio').evaluate((audio, t) => { audio.currentTime = t; }, target);
      await page.waitForFunction(t => Math.abs(document.querySelector('audio').currentTime - t) < .01, target);
      const row = result.trace.filter(r => r.timestamp_seconds <= target + .001).at(-1);
      assert(row);
      assert.equal(row.alarm, kind === 'positive');
      const expectedState = row.alarm ? 'ALARM' : row.reason === 'silence' ? 'Silent input' :
        row.reason === 'insufficient_audio' ? 'Insufficient audio' : row.quality_flags.length ? 'Review audio quality' : 'No alarm';
      await page.waitForFunction(expected => document.querySelector('#state').textContent === expected, expectedState);
      assert.match(await page.locator('#qualityinfo').textContent(), /do not suppress an alarm/);
      if (kind === 'positive') {
        await page.locator('audio').evaluate(audio => audio.play());
        await page.waitForFunction(t => document.querySelector('audio').currentTime > t + .2, target);
        await page.locator('audio').evaluate(audio => audio.pause());
        const downloadPromise = page.waitForEvent('download');
        await page.locator('#download').click();
        const download = await downloadPromise;
        assert.equal(download.suggestedFilename(), 'drone-scores.json');
        await page.screenshot({ path: screenshot, fullPage: true });
      }
      checks.push({ kind, windows: result.trace.length, duration: result.timing.duration_seconds,
        alarms: result.alarms.length, model_id: result.model_id, uploadAndSeek: true, displayedState: expectedState });
    }
    await page.locator('#file').setInputFiles({ name: 'broken.wav', mimeType: 'audio/wav', buffer: Buffer.from('invalid audio') });
    await page.locator('#analyze').click();
    await page.waitForFunction(() => document.querySelector('#error').textContent.includes('Unable to decode'));
    assert(await page.locator('#result').isHidden());
    assert.deepEqual(errors, []);
    const result = { passed: true, checks, download: true, playbackAdvanced: true, invalidAudioActionable: true,
      pageErrors: errors, browser: await browser.version(),
      scope: 'Software product verification on development recordings; not additional benchmark trials' };
    fs.writeFileSync(resultFile, JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
