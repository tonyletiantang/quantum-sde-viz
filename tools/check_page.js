// Checks the page in a headless browser: every control can be reached at several window sizes,
// the simulation passes its own physics self-test, and one picture of each look is saved to tools/out/.
//
// Needs Node.js and Playwright:   npm install playwright   then   npx playwright install chromium
// Run from the project folder:    node tools/check_page.js
const { chromium } = require('playwright');
const path = require('path'); const fs = require('fs');

const PAGE = require('url').pathToFileURL(path.resolve(__dirname, '../index.html')).href;
const OUT = path.join(__dirname, 'out');
const ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];   // software drawing, so it runs anywhere

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ args: ARGS });
  let failed = false;
  const open = async (width, height) => {
    const page = await browser.newPage({ viewport: { width, height } });
    page.on('pageerror', (e) => { failed = true; console.log('PAGE ERROR:', String(e)); });
    page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) { failed = true; console.log('CONSOLE ERROR:', m.text().slice(0, 400)); } });
    await page.goto(PAGE, { waitUntil: 'load' });
    await page.waitForTimeout(800);
    await page.evaluate(() => { if (!window.__scene.P.paused) document.getElementById('pause').click(); });
    return page;
  };

  // 1. controls and captions at several window sizes
  for (const [w, h] of [[1440, 810], [1180, 720], [980, 700], [640, 1100], [390, 780]]) {
    const page = await open(w, h);
    const res = await page.evaluate(() => {
      const blocked = [], rect = (el) => el.getBoundingClientRect();
      const reach = (el, name) => { const r = rect(el), t = document.elementFromPoint(r.left + r.width/2, r.top + r.height/2); if (!(t === el || el.contains(t))) blocked.push(name); };
      ['notes-btn', 'g', 'lam', 'speed', 'throw', 'pause', 'auto'].forEach((id) => reach(document.getElementById(id), id));
      document.querySelectorAll('.looks button').forEach((b) => reach(b, 'look ' + b.dataset.look));
      const shown = (sel) => [...document.querySelectorAll(sel)].filter((c) => getComputedStyle(c).display !== 'none');
      const hits = (a, b) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
      const caps = shown('.cap'), ctr = shown('#controls > *'), labs = shown('.lab');
      return { blocked, captionOverControls: caps.some((c) => ctr.some((k) => hits(rect(c), rect(k)))), sideways: document.documentElement.scrollWidth > window.innerWidth,
        captionOverLabel: caps.some((c) => labs.some((l) => hits(rect(c), rect(l)))),
        layout: document.documentElement.dataset.layout };
    });
    const ok = !res.blocked.length && !res.captionOverControls && !res.captionOverLabel && !res.sideways;
    if (!ok) failed = true;
    console.log(`${w}x${h} (${res.layout}): ${ok ? 'ok' : 'PROBLEM ' + JSON.stringify(res)}`);
    await page.close();
  }

  // 2. physics self-test, 3. one picture per look
  const page = await open(1440, 810);
  const t = await page.evaluate(() => window.__scene.selfTest());
  const pass = t.normDrift < 1e-4 && t.coherentWorstZ < 4.5 && t.restAfterRun < 1.5*t.restExpectedMax && t.restAfterRunWide < 1.5*t.restExpectedMax && t.throwLandsWorstZ < 4.5;
  if (!pass) failed = true;
  console.log('self-test:', pass ? 'ok' : 'PROBLEM', JSON.stringify(t));
  await page.evaluate(() => { const S = window.__scene; S.throwNow(0); S.run(0.3); document.documentElement.classList.add('dragged'); });
  for (const [key, name] of [['1', 'a-luminous'], ['2', 'b-ink'], ['3', 'c-glass']]) {
    await page.keyboard.press(key); await page.waitForTimeout(1800);
    await page.screenshot({ path: path.join(OUT, name + '.png') });
  }
  console.log('pictures saved in', OUT);
  await browser.close();
  console.log(failed ? 'RESULT: problems found' : 'RESULT: all checks passed');
  process.exit(failed ? 1 : 0);
})();
