// Optional managed-cloud browser check. Core Python checks require no browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');

(async () => {
  const [url, screenshot, resultFile] = process.argv.slice(2);
  assert(url && screenshot && resultFile, 'Usage: node tests/browser-smoke.cjs URL SCREENSHOT RESULT_JSON');
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 1100 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(() => document.querySelector('#record').options.length === 7);
    await page.selectOption('#record', '6');
    await page.waitForFunction(() => document.querySelector('audio').readyState >= 1);
    await page.locator('audio').evaluate(audio => { audio.currentTime = 8; });
    await page.waitForFunction(() => document.querySelector('#time').textContent === '8.00 s');
    const failureState = await page.locator('#alarm').textContent();
    assert.equal(failureState, 'NO ALARM');
    assert.match(await page.locator('#description').textContent(), /Post-hoc failure/);
    assert.equal(await page.locator('#rows tr').count(), 19);
    await page.screenshot({ path: screenshot, fullPage: true });
    const positiveIndex = await page.locator('#record option').evaluateAll(options =>
      options.findIndex(option => option.textContent.startsWith('g2-s00-positive')));
    assert(positiveIndex >= 0);
    await page.selectOption('#record', String(positiveIndex));
    await page.waitForFunction(() => document.querySelector('audio').readyState >= 1);
    await page.locator('audio').evaluate(audio => { audio.currentTime = 8; });
    await page.waitForFunction(() => document.querySelector('#time').textContent === '8.00 s');
    const positiveState = await page.locator('#alarm').textContent();
    assert.equal(positiveState, 'ALARM');
    await page.locator('audio').evaluate(audio => audio.play());
    await page.waitForFunction(() => document.querySelector('audio').currentTime > 8.2);
    await page.locator('audio').evaluate(audio => audio.pause());
    assert.deepEqual(errors, []);
    const result = {
      passed: true, records: 7, scoreRowsPerRecording: 19, failureState, positiveState,
      seeking: true, playbackAdvanced: true, pageErrors: errors, browser: await browser.version(),
      node: process.version, playwright: require('playwright/package.json').version,
      delivery: 'Internal loopback HTTP; direct file navigation is administratively blocked in this cloud browser',
      scope: 'HTML/audio/score/selection demonstration only; not real-world detector validation'
    };
    fs.writeFileSync(resultFile, JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
