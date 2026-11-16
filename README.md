# quantum-sde-viz

An interactive picture of the idea behind Bravyi, Byrne, Zayats and Zhuk,
*Quantum algorithms for stochastic nonlinear differential equations*
([arXiv:2606.08349](https://arxiv.org/abs/2606.08349)).

A noisy three-variable system is drawn twice: as a cloud of runs, and as light on the
levels of three quantum oscillators. The page has three looks (luminous, ink, glass).

## See it

Double-click `index.html`. It opens in your browser and runs from the files in this
folder. There is nothing to install and nothing to build. The only things it fetches from
the internet are its fonts and the math typesetter (KaTeX).

After you change a file, save it and reload the page.

## The files

| File | What is in it |
|---|---|
| `index.html` | Every word on the page. This is the file to edit for text. |
| `style.css` | How the words and controls look: colors and fonts of each look, sizes, spacing. |
| `scene.js` | The simulation and the drawing of the cloud and the levels. No words. |
| `tools/` | Optional checks used while building the page. Not part of the site. |
| `.nojekyll`, `.gitattributes`, `.gitignore` | Housekeeping for git and GitHub Pages. |

## Changing the words

Open `index.html` in VS Code. Comments like `<!-- === Title block, top left === -->`
mark each piece of text. Edit the text between the tags and leave the tags themselves.

- **Tab title and summary:** the `<title>` and `<meta name="description">` lines near the top.
- **Title block:** the small line above the title, the title, and the line under it.
- **Captions** under the cloud and under the levels.
- **Controls:** the words next to the sliders and on the buttons.
- **Notes panel:** one `<p>` per paragraph.
- **Labels on the scene:** the axis names and the two tags on the levels, near the bottom of the file.

Text that differs between looks is written three times, once per look, marked
`data-for="a"`, `data-for="b"` and `data-for="c"`.

Math in the notes panel is plain LaTeX: `$...$` inline and `$$...$$` on its own line.
It is typeset by KaTeX, loaded from the internet like the fonts. Inside math, write
`\lt` and `\gt` for `<` and `>`. Outside math, `<sub>1</sub>` is a subscript and
`<sup>2</sup>` a superscript.

Two things in the text are tied to the code. The scene draws levels with up to 6 excitations,
set by `const D = 6` in `scene.js`. The speed slider's range is the `min` and `max` on the
`<input id="speed">` line in `index.html`. If you change either, change the words that
mention them too.

## Changing the looks

- Colors and fonts of the text and controls: the three blocks at the top of `style.css`.
- How the scene is drawn: `scene.js`. The constants `LOOKA` and `LOOKB`, the color ramps
  `RAMP_A` and `RAMP_C`, and the functions `drawA`, `drawB`, `drawC` hold one look each.
- Starting values of the sliders: the `value="..."` on each `<input>` in `index.html`
  and the matching entries of `P` at the top of `scene.js` (`g`, `lambda`, `q`, `rate`).
- Length of the short tails on the dots: `TAU` and the `cap` entries of `LOOKA` and `LOOKB` in `scene.js`.

## Publishing

The page is live at <https://tonyletiantang.github.io/quantum-sde-viz/>.

## Checks

The scene was checked against the mathematics it shows. The checks can be run again
after changes to `scene.js`. They need Node.js with Playwright, and Python with numpy and scipy.

```
npm install playwright
npx playwright install chromium

node tools/check_page.js          # controls reachable, physics self-test, a picture of each look
node tools/amplitudes.js          # measure the level amplitudes with the page's own simulation
python tools/spectral_check.py tools/amplitudes.json   # compare with an independent calculation
```

What they establish:

- The force step keeps the distance from the origin, and the time step keeps the cloud at rest unchanged.
- A cloud moved away from rest has the amplitudes of a coherent state.
- The amplitudes measured from the runs agree, within sampling error, with evolving the
  oscillator state by the paper's operator: damping at one rate per quantum, plus an
  anti-Hermitian force term that links neighboring heights only.
