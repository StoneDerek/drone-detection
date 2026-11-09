const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');

function wav(frequency, amplitude, seconds) {
  const count = Math.round(16000 * seconds), bytes = count * 2;
  const data = Buffer.alloc(44 + bytes);
  data.write('RIFF'); data.writeUInt32LE(36 + bytes, 4); data.write('WAVEfmt ', 8);
  data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(1, 22);
  data.writeUInt32LE(16000, 24); data.writeUInt32LE(32000, 28);
  data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34);
  data.write('data', 36); data.writeUInt32LE(bytes, 40);
  for (let i = 0; i < count; i++) {
    const sample = Math.max(-1, Math.min(1, amplitude * Math.sin(2 * Math.PI * frequency * i / 16000)));
    data.writeInt16LE(Math.min(32767, Math.round(sample * 32768)), 44 + 2 * i);
  }
  return data;
}

(async () => {
  const [url, screenshot, resultFile] = process.argv.slice(2);
  assert(url && screenshot && resultFile, 'Provide URL, screenshot and result JSON');
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1160, height: 1100 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(() => document.querySelector('#identity').textContent.length === 64);
    const checks = [];
    for (const [name, frequency, amplitude, seconds, expected, flag] of [
      ['low-frequency', 30, .1, 4, 'Review audio quality', 'low_frequency_dominance'],
      ['clipping', 250, 2, 4, 'Review audio quality', 'clipping'],
      ['silence', 0, 0, 4, 'Silent input', null],
      ['short', 250, .1, .1, 'Insufficient audio', null]
    ]) {
      await page.locator('#file').setInputFiles({ name: name + '.wav', mimeType: 'audio/wav', buffer: wav(frequency, amplitude, seconds) });
      const response = page.waitForResponse(r => r.url().endsWith('/api/analyze') && r.request().method() === 'POST');
      await page.locator('#analyze').click();
      assert.equal((await response).status(), 200);
      await page.waitForFunction(() => !document.querySelector('#result').hidden);
      const payload = await page.evaluate(() => JSON.parse(JSON.stringify(result)));
      assert.equal(payload.alarms.length, 0);
      if (flag) assert(payload.trace.every(row => row.quality_flags.includes(flag)));
      await page.waitForFunction(() => document.querySelector('audio').readyState >= 1);
      const target = Math.min(3.6, seconds);
      await page.locator('audio').evaluate((audio, t) => { audio.currentTime = t; }, target);
      await page.waitForFunction(state => document.querySelector('#state').textContent === state, expected);
      if (name === 'low-frequency') {
        assert.match(await page.locator('#summary').textContent(), /not a wind-speed/);
        assert.match(await page.locator('#rows').textContent(), /Low-frequency energy/);
        await page.screenshot({ path: screenshot, fullPage: true });
      }
      checks.push({ name, displayedState: expected, windows: payload.trace.length, flag });
    }
    assert.deepEqual(errors, []);
    const result = { passed: true, checks, browser: await browser.version(), pageErrors: errors,
      scope: 'Generated signals verify software handling and display only. They are not acoustic detection evidence.' };
    fs.writeFileSync(resultFile, JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
