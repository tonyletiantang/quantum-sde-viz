// Runs the page's own simulation and writes the measured oscillator amplitudes to tools/amplitudes.json.
// spectral_check.py then compares them with an independent calculation.
//
// Run from the project folder:    node tools/amplitudes.js     (needs Playwright, see check_page.js)
const { chromium } = require('playwright');
const path = require('path'); const fs = require('fs');
(async () => {
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 640, height: 400 } });
  page.on('pageerror', (e) => console.log('PAGE ERROR:', String(e)));
  await page.goto(require('url').pathToFileURL(path.resolve(__dirname, '../index.html')).href, { waitUntil: 'load' });
  await page.waitForTimeout(800);
  await page.evaluate(() => document.getElementById('pause').click());
  // each case: start point x0, time t, damping rate lambda, noise strength q (default 2 lambda), force strength g, time step dt
  const cases = [
    { x0: [1.1, 0.6, 1.56], t: 0.5, lambda: 0.22, g: 1.4, dt: 0.01 },
    { x0: [-0.76, 1.56, 1.0], t: 1.5, lambda: 0.22, g: 1.4, dt: 0.01 },
    { x0: [1.1, 0.6, 1.56], t: 0.5, lambda: 0.22, g: 1.4, dt: 0.02 },
    { x0: [1.1, 0.6, 1.56], t: 0.8, lambda: 0.22, q: 0.8, g: 1.4, dt: 0.01 },     // q differs from 2 lambda: wider resting cloud
    { x0: [0.5, -0.4, 0.6], t: 1.0, lambda: 0.4, q: 0.4, g: 1.4, dt: 0.01 },      // narrower resting cloud
  ];
  const out = { sites: await page.evaluate(() => window.__scene.sites), cases: [] };
  for (const c of cases) out.cases.push({ ...c, batches: await page.evaluate((c) => window.__scene.amplitudes(c.x0, c.t, c.lambda, c.g, c.dt, c.q), c) });
  const file = path.join(__dirname, 'amplitudes.json');
  fs.writeFileSync(file, JSON.stringify(out));
  console.log('wrote', file);
  await browser.close();
})();
