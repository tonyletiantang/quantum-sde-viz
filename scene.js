/* The scene: simulation and drawing. There are no words in this file;
   every word on the page is in index.html, and its appearance is in style.css.

   The system. Three variables X = (X1, X2, X3) obey
       dX = [ -lambda X + f(X) ] dt + sqrt(q) dW,
       f(X) = g (X2 X3, X3 X1, -2 X1 X2).
   Damping and noise alone settle into the resting Gaussian N(0, p I) with
   variance p = q / (2 lambda). The force keeps |X| and has zero divergence,
   so it leaves that Gaussian unchanged for every p: the assumption of
   Bravyi, Byrne, Zayats and Zhuk, arXiv:2606.08349.

   Two drawings of the same state.
   (1) The cloud: M runs of the system. Looks A and B draw a run as a streak
       whose tail is the path along which the force alone carried it.
   (2) The oscillator levels: phi_n = E[ h_n1(Y1) h_n2(Y2) h_n3(Y3) ], with
       Y = X / sqrt(p) and h_k = He_k / sqrt(k!) the normalized Hermite polynomials. These are the
       amplitudes of the state on the levels |n1 n2 n3> of three oscillators,
       and they are measured from the runs in every frame.

   The time step keeps the resting Gaussian exactly (see step()).

   Where to look when changing things:
     params ............ starting values, number of runs, highest level drawn
     simulation ........ the time step
     levels ............ measuring the amplitudes from the runs
     shaders, looks .... how each of the three looks is drawn (LOOKA, LOOKB, colors)
     camera & layout ... where the two objects sit on screen
     UI ................ sliders, buttons, keys                                  */
(function () {
'use strict';

const root = document.documentElement;
const canvas = document.getElementById('stage');
const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, depth: true, stencil: false, powerPreference: 'high-performance' });
if (!gl) { document.getElementById('nogl').hidden = false; return; }
const HALF = !!(gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float'));

/* ------------------------------------------------------------------ params */
const P = { g: 1.4, lambda: 0.22, q: 0.44, rate: 1, M: 90000, look: 'a', auto: true, paused: false };
const restWidth = () => Math.sqrt(P.q/(2*P.lambda));   // standard deviation sqrt(p) of the resting Gaussian
const SPEED = 0.9;         // scene time per second of real time when the speed control reads 1
const MMAX = 90000;        // particles simulated
const MS = 12000;          // particles used to measure the levels in one frame (the subset rotates)
const D = 6;               // highest level drawn (total number of excitations)
const R0 = 2.0;            // distance of a throw
const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* Ambient mode (index.html?ambient): the scene alone, as a decoration embedded in another page.
   No text or controls, fewer runs, a calmer pace. The embedding page can send messages:
     { type: 'qsv-theme', theme: 'dark' | 'light' }   dark shows look A, light shows look B
     { type: 'qsv-visible', visible: true | false }   pauses the scene while it is scrolled out of view
   ?look=a|b|c picks the starting look. */
const QS = new URLSearchParams(location.search);
const AMBIENT = QS.has('ambient');
let ambientVisible = true;
if (AMBIENT) {
  P.M = 24000; P.rate = 0.7;
  if (/^[abc]$/.test(QS.get('look') || '')) P.look = QS.get('look');
  root.classList.add('ambient');
}

/* -------------------------------------------------------------------- math */
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const dot3 = (a, b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/l, a[1]/l, a[2]/l]; };
function perspective(fovy, aspect, near, far) {
  const f = 1/Math.tan(fovy/2), nf = 1/(near-far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function lookAt(e, t, up) {
  const z = norm([e[0]-t[0], e[1]-t[1], e[2]-t[2]]), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot3(x,e),-dot3(y,e),-dot3(z,e),1]);
}

/* -------------------------------------------------------------- simulation */
const pos = new Float32Array(MMAX*3);
const GT = new Float32Array(65536);          // table of standard normal values: one per quantile, so its moments are accurate
(function () {
  const A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const Bc = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const Dc = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const icdf = (p) => {                         // inverse normal distribution function (Acklam's rational approximation)
    if (p < 0.02425) { const q = Math.sqrt(-2*Math.log(p)); return (((((C[0]*q+C[1])*q+C[2])*q+C[3])*q+C[4])*q+C[5])/((((Dc[0]*q+Dc[1])*q+Dc[2])*q+Dc[3])*q+1); }
    if (p > 1-0.02425) return -icdf(1-p);
    const q = p-0.5, r = q*q;
    return (((((A[0]*r+A[1])*r+A[2])*r+A[3])*r+A[4])*r+A[5])*q/(((((Bc[0]*r+Bc[1])*r+Bc[2])*r+Bc[3])*r+Bc[4])*r+1);
  };
  let v = 0;
  for (let i = 0; i < 65536; i++) { GT[i] = icdf((i+0.5)/65536); v += GT[i]*GT[i]; }
  v = Math.sqrt(v/65536); for (let i = 0; i < 65536; i++) GT[i] /= v;
})();
let rs = 0x2F6E2B1 | 0;
function rngIdx() { rs ^= rs << 13; rs ^= rs >>> 17; rs ^= rs << 5; return (rs >>> 5) & 65535; }

function fillGaussian(center, sd) {                      // the resting Gaussian (standard deviation sd) shifted to center
  const w = sd === undefined ? restWidth() : sd;
  for (let i = 0; i < MMAX*3; i += 3) {
    pos[i] = center[0] + w*GT[rngIdx()]; pos[i+1] = center[1] + w*GT[rngIdx()]; pos[i+2] = center[2] + w*GT[rngIdx()];
  }
}

/* One step of length dt for the first m particles.
   The force splits as f = f1 + f3 with
       f1 = 2g X1 (0, X3, -X2)   a rotation about axis 1 whose rate depends only on X1,
       f3 =  g X3 (X2, -X1, 0)   a rotation about axis 3 whose rate depends only on X3.
   Each piece is solved exactly (half step of f1, full step of f3, half step of f1).
   Each rotation angle depends only on the coordinate that the rotation leaves alone,
   so every piece keeps |X| and volume exactly, and so keeps the resting Gaussian
   exactly, for any dt. The rotations use the rational (Cayley) form of sin and cos.
   Damping and noise take one exact Ornstein-Uhlenbeck step,
       X -> a X + b xi,   a = exp(-lambda dt),   b^2 = q (1 - a^2) / (2 lambda),
   with xi standard normal; it keeps N(0, p I) exactly. Successive steps
   alternate the order (force then noise, noise then force), which makes every
   pair of steps symmetric and the scheme second-order accurate in dt. */
let stepParity = 0;
function step(dt, m, lambda, q, g) {
  const a = Math.exp(-lambda*dt);
  const b = Math.sqrt(Math.max(0, lambda > 0 ? q*(1 - a*a)/(2*lambda) : q*dt));
  const h = 0.5*g*dt, n = m*3, X = pos, G = GT;
  const noiseFirst = (stepParity++ & 1) === 1;
  let s = rs;
  for (let i = 0; i < n; i += 3) {
    let x1 = X[i], x2 = X[i+1], x3 = X[i+2];
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5; const i1 = (s >>> 5) & 65535;
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5; const i2 = (s >>> 5) & 65535;
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5; const i3 = (s >>> 5) & 65535;
    if (noiseFirst) { x1 = a*x1 + b*G[i1]; x2 = a*x2 + b*G[i2]; x3 = a*x3 + b*G[i3]; }
    let t = h*x1, q = 1/(1 + t*t), c = (1 - t*t)*q, sn = 2*t*q;
    const y2 = x2*c + x3*sn, y3 = x3*c - x2*sn;
    t = h*y3; q = 1/(1 + t*t); c = (1 - t*t)*q; sn = 2*t*q;
    const z1 = x1*c + y2*sn, z2 = y2*c - x1*sn;
    t = h*z1; q = 1/(1 + t*t); c = (1 - t*t)*q; sn = 2*t*q;
    x1 = z1; x2 = z2*c + y3*sn; x3 = y3*c - z2*sn;
    if (!noiseFirst) { x1 = a*x1 + b*G[i1]; x2 = a*x2 + b*G[i2]; x3 = a*x3 + b*G[i3]; }
    X[i] = x1; X[i+1] = x2; X[i+2] = x3;
  }
  rs = s;
}

/* ------------------------------------------------- levels of 3 oscillators */
const sites = [];
for (let i = 0; i <= D; i++) for (let j = 0; j <= D-i; j++) for (let k = 0; k <= D-i-j; k++) sites.push([i, j, k]);
const NS = sites.length;
const phi = new Float64Array(NS), raw = new Float64Array(NS), acc = new Float64Array(NS), mean = [0, 0, 0];
const ha = new Float64Array(D+1), hb = new Float64Array(D+1), hc = new Float64Array(D+1);
const SQ = new Float64Array(D+2); for (let k = 0; k < D+2; k++) SQ[k] = Math.sqrt(k);
let msRound = 0;
function measure(alpha, count, from, sd) {                // Hermite amplitudes in the basis of the resting Gaussian (standard deviation sd)
  const Ms = Math.min(count || MS, P.M);
  const first = count ? (from || 0) : (msRound++ % Math.max(1, Math.floor(P.M/Ms)))*Ms;
  const iw = 1/(sd === undefined ? restWidth() : sd);
  acc.fill(0);
  for (let p = first; p < first+Ms; p++) {
    const o = p*3, x = pos[o]*iw, y = pos[o+1]*iw, z = pos[o+2]*iw;
    ha[0] = 1; ha[1] = x; hb[0] = 1; hb[1] = y; hc[0] = 1; hc[1] = z;
    for (let k = 1; k < D; k++) {
      const s0 = SQ[k], s1 = 1/SQ[k+1];
      ha[k+1] = (x*ha[k] - s0*ha[k-1])*s1; hb[k+1] = (y*hb[k] - s0*hb[k-1])*s1; hc[k+1] = (z*hc[k] - s0*hc[k-1])*s1;
    }
    let idx = 0;
    for (let i = 0; i <= D; i++) { const A = ha[i];
      for (let j = 0; j <= D-i; j++) { const AB = A*hb[j], km = D-i-j;
        for (let k = 0; k <= km; k++) acc[idx++] += AB*hc[k]; } }
  }
  const inv = 1/Ms;
  for (let s = 0; s < NS; s++) { const r = acc[s]*inv; raw[s] = r; phi[s] += alpha*(r - phi[s]); }
}
function centroid() {                                    // the cloud's center, from every run
  let sx = 0, sy = 0, sz = 0; const n = P.M*3;
  for (let i = 0; i < n; i += 3) { sx += pos[i]; sy += pos[i+1]; sz += pos[i+2]; }
  mean[0] = sx/P.M; mean[1] = sy/P.M; mean[2] = sz/P.M;
}
function excitedLight() { let e = 0; for (let s = 1; s < NS; s++) e += phi[s]*phi[s]; return e; }

/* ------------------------------------------------------------- GL helpers */
function compile(type, src, name) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error('shader ' + name + ': ' + gl.getShaderInfoLog(s));
  return s;
}
function program(vs, fs, name) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs, name + '.vs')); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs, name + '.fs'));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) console.error('link ' + name + ': ' + gl.getProgramInfoLog(p));
  const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, info.name); }
  return { p, u };
}
function target(w, h, hdr, filter) {
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  if (hdr && HALF) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
  else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter || gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter || gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { tex, fbo, w, h };
}
function freeTarget(t) { if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); } }

/* -------------------------------------------- the frame shared by both objects
   Both objects are drawn in the same frame, so the cloud's axis x_i and the lattice's
   edge for oscillator i point the same way. One fixed rotation ROT turns that frame so
   that the diagonal (1,1,1) points straight up: the lattice then stands on its corner,
   and the height of a site counts its total number of excitations. */
const ROT = (function () {
  const by = norm([1, 1, 1]), bx = norm([2, -1, -1]), bz = cross(bx, by);   // rows of the rotation
  return { rows: [bx, by, bz], gl: new Float32Array([bx[0], by[0], bz[0], bx[1], by[1], bz[1], bx[2], by[2], bz[2]]) };
})();
const rot = (p) => ROT.rows.map((r) => r[0]*p[0] + r[1]*p[1] + r[2]*p[2]);          // data frame -> world
const unrot = (w) => [0, 1, 2].map((k) => ROT.rows[0][k]*w[0] + ROT.rows[1][k]*w[1] + ROT.rows[2][k]*w[2]);   // world -> data frame

/* ----------------------------------------------------------------- shaders */
const COMMON = `
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_group;      // xy: shift of this object on screen (clip units), z: zoom
uniform vec4 u_model;      // xyz: translation, w: scale
uniform mat3 u_rot;        // the shared frame (ROT)
uniform float u_pxPerUnit; // pixels per world unit at clip w = 1
uniform vec2 u_viewport;
vec3 toWorld(vec3 p){ return u_rot*(p*u_model.w) + u_model.xyz; }
vec4 worldToClip(vec3 w){ vec4 c = u_proj*(u_view*vec4(w,1.0)); c.xy = c.xy*u_group.z + u_group.xy*c.w; return c; }
vec4 toClip(vec3 p){ return worldToClip(toWorld(p)); }
`;
const RAMP = `
uniform vec3 u_ramp[5];
vec3 ramp(float t){
  t = clamp(t,0.0,1.0)*4.0;
  vec3 c = mix(u_ramp[0], u_ramp[1], smoothstep(0.0,1.0,t));
  c = mix(c, u_ramp[2], smoothstep(1.0,2.0,t));
  c = mix(c, u_ramp[3], smoothstep(2.0,3.0,t));
  c = mix(c, u_ramp[4], smoothstep(3.0,4.0,t));
  return c;
}
`;
const LIT = `
uniform float u_log;       // 1: brightness follows log(light) between u_lrange.x and u_lrange.y (natural logs); 0: saturating map
uniform vec2 u_lrange;
float logLit(float v){ return clamp((log(max(v, 1e-12)) - u_lrange.x)/(u_lrange.y - u_lrange.x), 0.0, 1.0); }
float litOf(float v, float k){ return mix(1.0-exp(-v*k), logLit(v), u_log); }
`;
const HASH = `
float hash(vec2 p){ p = fract(p*vec2(123.34,456.21)); p += dot(p,p+45.32); return fract(p.x*p.y); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1.0,0.0)),f.x), mix(hash(i+vec2(0.0,1.0)),hash(i+vec2(1.0,1.0)),f.x), f.y); }
`;
const SHADE = `
uniform vec3 u_light;      // unit vector toward the key light, world space
vec3 shadeBall(vec3 base, vec3 nV, mat3 v2w, float gloss, float sh, float emis, vec3 glow){
  vec3 nW = v2w*nV;
  vec3 vW = v2w*vec3(0.0,0.0,1.0);
  float ndl = dot(nW,u_light);
  float diff = max(ndl,0.0);
  vec3 sky = mix(vec3(0.46,0.50,0.60), vec3(1.0,1.0,1.0), 0.5+0.5*nW.y);
  vec3 hW = normalize(u_light+vW);
  float spec = pow(max(dot(nW,hW),0.0), gloss);
  float fres = pow(1.0-nV.z, 3.0);
  vec3 col = base*(sky*(0.44+0.24*sh) + vec3(1.0,0.97,0.92)*0.66*diff*sh);
  col += glow*emis;
  col += vec3(1.0)*spec*0.9*sh + vec3(0.90,0.95,1.0)*fres*0.20*(0.5+0.5*sh);
  return col;
}
`;

const VS_FULL = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_p;
out vec2 v_uv;
void main(){ v_uv = a_p*0.5+0.5; gl_Position = vec4(a_p,0.0,1.0); }`;

/* look A: additive light */
const VS_PT_A = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;
${COMMON}${RAMP}
uniform float u_size, u_rmax, u_dist;
out vec3 v_col; out float v_gain;
void main(){
  vec4 c = toClip(a_pos);
  gl_Position = c;
  float px = u_size*u_pxPerUnit/c.w;
  float ps = max(px, 2.0);
  gl_PointSize = ps;
  v_gain = (px*px)/(ps*ps)*mix(1.35, 0.45, smoothstep(u_dist-3.2, u_dist+3.2, c.w));
  v_col = ramp(length(a_pos)/u_rmax);
}`;
const FS_PT_A = `#version 300 es
precision highp float;
in vec3 v_col; in float v_gain;
uniform float u_intensity;
out vec4 o;
void main(){
  vec2 d = gl_PointCoord*2.0-1.0;
  float a = exp(-dot(d,d)*4.0)-0.0183;
  if(a<=0.0) discard;
  o = vec4(v_col*(a*u_intensity*v_gain),1.0);
}`;
const FS_DOWN = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform vec2 u_texel; in vec2 v_uv; out vec4 o;
void main(){
  vec4 s = texture(u_tex,v_uv)*4.0;
  s += texture(u_tex,v_uv+u_texel*vec2(-1.0,-1.0)); s += texture(u_tex,v_uv+u_texel*vec2(1.0,-1.0));
  s += texture(u_tex,v_uv+u_texel*vec2(-1.0,1.0));  s += texture(u_tex,v_uv+u_texel*vec2(1.0,1.0));
  o = s*0.125;
}`;
const FS_UP = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform vec2 u_texel; uniform float u_k; in vec2 v_uv; out vec4 o;
void main(){
  vec4 s = texture(u_tex,v_uv+u_texel*vec2(-2.0,0.0)) + texture(u_tex,v_uv+u_texel*vec2(2.0,0.0))
         + texture(u_tex,v_uv+u_texel*vec2(0.0,-2.0)) + texture(u_tex,v_uv+u_texel*vec2(0.0,2.0));
  s += 2.0*(texture(u_tex,v_uv+u_texel*vec2(-1.0,-1.0)) + texture(u_tex,v_uv+u_texel*vec2(1.0,-1.0))
          + texture(u_tex,v_uv+u_texel*vec2(-1.0,1.0)) + texture(u_tex,v_uv+u_texel*vec2(1.0,1.0)));
  o = s*(u_k/12.0);
}`;
const FS_COMP = `#version 300 es
precision highp float;
uniform sampler2D u_scene; uniform sampler2D u_bloom;
uniform float u_bloomK, u_exposure; uniform vec2 u_res; uniform vec2 u_focus;
in vec2 v_uv; out vec4 o;
${HASH}
void main(){
  vec3 hdr = texture(u_scene,v_uv).rgb + u_bloomK*texture(u_bloom,v_uv).rgb;
  vec3 col = 1.0-exp(-hdr*u_exposure);
  col = pow(col, vec3(0.4545));
  vec2 q = (v_uv-u_focus)*vec2(u_res.x/u_res.y,1.0);
  float v = exp(-dot(q,q)*1.5);
  vec3 bg = mix(vec3(0.015,0.016,0.027), vec3(0.043,0.050,0.090), v);
  col = bg + col*(1.0-bg);
  col += (hash(gl_FragCoord.xy)-0.5)/255.0;
  o = vec4(col,1.0);
}`;

/* look B: ink */
const FS_BG_B = `#version 300 es
precision highp float;
uniform vec2 u_res; uniform vec3 u_paper;
in vec2 v_uv; out vec4 o;
${HASH}
void main(){
  float grain = hash(gl_FragCoord.xy)*0.020-0.010;
  float mott = (vnoise(gl_FragCoord.xy/110.0)-0.5)*0.016 + (vnoise(gl_FragCoord.xy*vec2(0.5,0.045))-0.5)*0.010;
  vec2 q = v_uv-0.5;
  vec3 c = u_paper*(1.0+grain+mott) - dot(q,q)*0.07;
  o = vec4(c,1.0);
}`;

/* look C: beads, glass, shadows */
const SHADOWMAP = `
uniform vec3 u_Lx, u_Ly, u_L;
uniform float u_shE, u_shZ;
vec3 toLight(vec3 w){ return vec3(dot(w,u_Lx)/u_shE, dot(w,u_Ly)/u_shE, -dot(w,u_L)/u_shZ); }
`;
const VS_PT_C = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;
${COMMON}${RAMP}
uniform float u_size;
out vec3 v_col; out vec3 v_world; out float v_zv; flat out float v_px;
void main(){
  vec3 wp = toWorld(a_pos);
  vec4 vp = u_view*vec4(wp,1.0);
  vec4 c = u_proj*vp; c.xy = c.xy*u_group.z + u_group.xy*c.w;
  gl_Position = c;
  float px = u_size*u_pxPerUnit/c.w;
  gl_PointSize = px;
  v_px = px; v_zv = vp.z; v_world = wp;
  v_col = ramp(length(a_pos)/4.0);
}`;
const FS_PT_C = `#version 300 es
precision highp float;
precision highp sampler2DShadow;
uniform mat4 u_view; uniform mat4 u_proj;
uniform float u_radW, u_shTexel, u_bias, u_haze, u_dist;
uniform sampler2DShadow u_shadow;
${SHADE}${SHADOWMAP}
in vec3 v_col; in vec3 v_world; in float v_zv; flat in float v_px;
out vec4 o;
void main(){
  vec2 d = vec2(gl_PointCoord.x*2.0-1.0, 1.0-gl_PointCoord.y*2.0);
  float r2 = dot(d,d);
  if(r2>1.0) discard;
  float nz = sqrt(1.0-r2);
  mat3 v2w = transpose(mat3(u_view));
  vec3 nV = vec3(d,nz);
  vec3 wp = v_world + (v2w*nV)*u_radW;
  vec3 q = toLight(wp)*0.5+0.5;
  float t = u_shTexel, sh = 0.0;
  sh += texture(u_shadow, vec3(q.xy+vec2(-t,-t), q.z-u_bias));
  sh += texture(u_shadow, vec3(q.xy+vec2( t,-t), q.z-u_bias));
  sh += texture(u_shadow, vec3(q.xy+vec2(-t, t), q.z-u_bias));
  sh += texture(u_shadow, vec3(q.xy+vec2( t, t), q.z-u_bias));
  sh *= 0.25;
  vec3 col = shadeBall(v_col, nV, v2w, 42.0, sh, 0.0, vec3(0.0));
  col = mix(col, vec3(0.90,0.915,0.935), u_haze*smoothstep(u_dist-2.5, u_dist+5.0, -v_zv));   // far beads fade into the backdrop
  float zv = v_zv + u_radW*nz;
  gl_FragDepth = ((u_proj[2][2]*zv + u_proj[3][2])/(-zv))*0.5+0.5;
  o = vec4(col, clamp((1.0-sqrt(r2))*v_px*0.5+0.5, 0.0, 1.0));
}`;
const VS_DEPTH = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;
${SHADOWMAP}
uniform float u_ps; uniform mat3 u_rot;
void main(){ gl_Position = vec4(toLight(u_rot*a_pos),1.0); gl_PointSize = u_ps; }`;
const FS_DEPTH = `#version 300 es
precision highp float;
void main(){ vec2 d = gl_PointCoord*2.0-1.0; if(dot(d,d)>1.0) discard; }`;
const VS_DENS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;
uniform vec3 u_light; uniform float u_floorY, u_extent, u_ps; uniform mat3 u_rot;
void main(){
  vec3 w = u_rot*a_pos;
  vec3 f = w - u_light*((w.y-u_floorY)/u_light.y);
  gl_Position = vec4(f.x/u_extent, f.z/u_extent, 0.0, 1.0);
  gl_PointSize = u_ps;
}`;
const FS_DENS = `#version 300 es
precision highp float;
uniform float u_k; out vec4 o;
void main(){ vec2 d = gl_PointCoord*2.0-1.0; float a = exp(-dot(d,d)*3.0)-0.05; if(a<=0.0) discard; o = vec4(a*u_k,0.0,0.0,1.0); }`;
const FS_BLUR = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform vec2 u_dir; in vec2 v_uv; out vec4 o;
void main(){
  vec4 s = texture(u_tex,v_uv)*0.2270270;
  s += (texture(u_tex,v_uv+u_dir*1.3846154)+texture(u_tex,v_uv-u_dir*1.3846154))*0.3162162;
  s += (texture(u_tex,v_uv+u_dir*3.2307692)+texture(u_tex,v_uv-u_dir*3.2307692))*0.0702703;
  o = s;
}`;
const FS_BG_C = `#version 300 es
precision highp float;
uniform vec2 u_res;
in vec2 v_uv; out vec4 o;
${HASH}
void main(){
  vec3 top = vec3(0.800,0.832,0.872), mid = vec3(0.905,0.920,0.938), bot = vec3(0.955,0.960,0.966);
  vec3 c = mix(bot, mid, smoothstep(0.0,0.42,v_uv.y));
  c = mix(c, top, smoothstep(0.42,1.0,v_uv.y));
  vec2 q = (v_uv-vec2(0.5,0.52))*vec2(u_res.x/u_res.y,1.0);
  c -= dot(q,q)*0.045;
  c += (hash(gl_FragCoord.xy)-0.5)/255.0;
  o = vec4(c,1.0);
}`;
const VS_FLOOR = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_p;
${COMMON}
uniform float u_floorY, u_extent;
out vec2 v_xz;
void main(){ v_xz = a_p*u_extent; gl_Position = worldToClip(vec3(v_xz.x,u_floorY,v_xz.y)); }`;
const FS_FLOOR = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform float u_extent, u_strength; uniform vec3 u_tint; uniform vec3 u_shell;
in vec2 v_xz; out vec4 o;
void main(){
  float dns = texture(u_tex, v_xz/u_extent*0.5+0.5).r;
  float sh = 1.0-exp(-dns);
  float ds = length(v_xz-u_shell.xy)/u_shell.z;
  float glass = (1.0-smoothstep(0.80,1.06,ds))*0.035 + smoothstep(0.78,0.97,ds)*(1.0-smoothstep(0.97,1.07,ds))*0.045;
  float fade = 1.0-smoothstep(u_extent*0.6, u_extent*0.98, length(v_xz));
  o = vec4(u_tint, clamp(sh*u_strength+glass,0.0,0.9)*fade);
}`;
const VS_GLASS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;
${COMMON}
uniform float u_R;
out vec3 v_n; out vec3 v_w;
void main(){ v_n = u_rot*a_pos; v_w = toWorld(a_pos*u_R); gl_Position = worldToClip(v_w); }`;
const FS_GLASS = `#version 300 es
precision highp float;
uniform vec3 u_eye; uniform vec3 u_light; uniform float u_face;
in vec3 v_n; in vec3 v_w; out vec4 o;
void main(){
  vec3 n = normalize(v_n)*u_face;
  vec3 v = normalize(u_eye-v_w);
  float ndv = clamp(dot(n,v),0.0,1.0);
  float fres = pow(1.0-ndv,3.0);
  vec3 rl = reflect(-v,n);
  vec3 env = mix(vec3(0.60,0.65,0.72), vec3(1.0), smoothstep(-0.25,0.9,rl.y));
  float box = smoothstep(0.86,0.97,dot(rl,normalize(vec3(0.35,0.86,0.30))));
  float strip = smoothstep(0.90,0.985,dot(rl,normalize(vec3(-0.75,0.35,0.55))))*0.6;
  float spec = pow(max(dot(rl,u_light),0.0),220.0);
  vec3 col = env*(0.25+0.75*fres) + vec3(box*0.55+strip*0.5) + vec3(spec*1.5);
  col = mix(col, vec3(0.50,0.62,0.74), 0.30*(1.0-fres));               // a faint cool tint in the body of the glass
  float a = 0.045 + 0.62*fres + (box*0.30+strip*0.26)*0.8 + spec;
  o = vec4(col, clamp(a,0.0,1.0)*(u_face>0.0 ? 1.0 : 0.5));
}`;

/* each run drawn as a short streak: where the force alone carried it during the last u_tau */
const FLOW = `
uniform float u_g;
vec3 flowStep(vec3 x, float t){      // one split step of the force alone: half of f1, all of f3, half of f1
  float a = u_g*x.x*t, c = cos(a), s = sin(a);
  x = vec3(x.x, x.y*c + x.z*s, x.z*c - x.y*s);
  a = u_g*x.z*t; c = cos(a); s = sin(a);
  x = vec3(x.x*c + x.y*s, x.y*c - x.x*s, x.z);
  a = u_g*x.x*t; c = cos(a); s = sin(a);
  return vec3(x.x, x.y*c + x.z*s, x.z*c - x.y*s);
}
`;
const VS_STREAK = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_st;       // x: 0 at the run's position, 1 at the end of the tail; y: side
layout(location=1) in vec3 a_pos;
${COMMON}${FLOW}
uniform float u_tau, u_width, u_cap;
out float v_s; out float v_side; out float v_r; out float v_w; flat out float v_seed;
float hash1(float n){ return fract(sin(n*12.9898)*43758.5453); }
void main(){
  vec3 p = a_pos;
  int n = int(a_st.x*5.0+0.5);
  float len0 = u_g*length(vec3(p.y*p.z, p.z*p.x, -2.0*p.x*p.y))*u_tau;   // length the streak would have
  float h = -u_tau*u_cap/(u_cap+len0)/5.0;                               // long streaks saturate at u_cap
  for(int i=0;i<5;i++){ if(i>=n) break; p = flowStep(p,h); }
  vec3 f = vec3(p.y*p.z, p.z*p.x, -2.0*p.x*p.y);
  float fl = length(f);
  vec3 tg = fl>1e-6 ? f/fl : vec3(1.0,0.0,0.0);
  vec4 c = toClip(p), c2 = toClip(p+tg*0.04);
  vec2 d = (c2.xy/c2.w-c.xy/c.w)*u_viewport;
  float dl = length(d); d = dl>1e-6 ? d/dl : vec2(1.0,0.0);
  float hw = u_width*0.5+1.0;
  c.xy += vec2(-d.y,d.x)*a_st.y*hw/(u_viewport*0.5)*c.w;
  gl_Position = c;
  v_s = a_st.x; v_side = a_st.y*hw; v_r = length(a_pos); v_w = c.w; v_seed = hash1(float(gl_InstanceID));
}`;
const FS_STREAK_A = `#version 300 es
precision highp float;
${RAMP}
uniform float u_width, u_intensity, u_dist, u_rmax; uniform vec2 u_spark;
in float v_s; in float v_side; in float v_r; in float v_w; flat in float v_seed;
out vec4 o;
void main(){
  float a = smoothstep(u_width*0.5+0.5, u_width*0.5-0.5, abs(v_side));
  float tail = (1.0-v_s)*(1.0-v_s);
  float depth = mix(1.35, 0.45, smoothstep(u_dist-3.2, u_dist+3.2, v_w));
  float sp = v_seed*v_seed; sp *= sp; sp *= sp;                       // a few streaks shine much brighter than the rest
  o = vec4(ramp(v_r/u_rmax)*(a*tail*u_intensity*depth*(u_spark.x+u_spark.y*sp*sp)),1.0);
}`;
const FS_STREAK_B = `#version 300 es
precision highp float;
uniform vec3 u_ink, u_spot; uniform float u_width, u_alpha, u_shellR, u_dist, u_out;
in float v_s; in float v_side; in float v_r; in float v_w; flat in float v_seed;
out vec4 o;
void main(){
  float a = smoothstep(u_width*0.5+0.5, u_width*0.5-0.5, abs(v_side));
  float taper = smoothstep(1.0,0.35,v_s);
  float outside = smoothstep(u_shellR-0.02, u_shellR+0.02, v_r);
  float fade = mix(1.0, 0.45, smoothstep(u_dist-3.0, u_dist+3.5, v_w));
  float al = u_alpha*a*taper*fade*(0.6+0.8*v_seed)*mix(1.0,u_out,outside);
  if(al<=0.003) discard;
  o = vec4(mix(u_ink,u_spot,outside), al);
}`;

/* lines, drawn as screen-space ribbons so the width is under control */
const VS_LINE = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_corner;
layout(location=1) in vec3 a_p0;
layout(location=2) in vec3 a_p1;
layout(location=3) in vec2 a_t;
${COMMON}
uniform float u_width, u_capLen; uniform vec3 u_camDir;
out float v_side; out float v_t; out float v_facing;
void main(){
  vec4 c0 = toClip(a_p0), c1 = toClip(a_p1);
  vec2 s0 = c0.xy/c0.w*u_viewport*0.5, s1 = c1.xy/c1.w*u_viewport*0.5;
  vec2 dir = s1-s0; float len = length(dir);
  dir = len>1e-4 ? dir/len : vec2(1.0,0.0);
  vec2 nrm = vec2(-dir.y,dir.x);
  float hw = u_width*0.5+1.0;
  vec4 c = mix(c0,c1,a_corner.x);
  vec2 off = nrm*a_corner.y*hw + dir*(a_corner.x*2.0-1.0)*u_capLen;
  c.xy += off/(u_viewport*0.5)*c.w;
  gl_Position = c;
  v_side = a_corner.y*hw;
  v_t = mix(a_t.x,a_t.y,a_corner.x);
  v_facing = dot(u_rot*normalize(mix(a_p0,a_p1,a_corner.x)), u_camDir);
}`;
const FS_LINE = `#version 300 es
precision highp float;
uniform vec4 u_color; uniform float u_width; uniform vec4 u_dash; uniform vec2 u_fade; uniform float u_additive;
in float v_side; in float v_t; in float v_facing;
out vec4 o;
void main(){
  float a = smoothstep(u_width*0.5+0.5, u_width*0.5-0.5, abs(v_side));
  float back = 1.0-smoothstep(-0.04,0.04,v_facing);
  if(u_dash.x>0.0){
    float on = step(fract(v_t/u_dash.x), u_dash.y);
    a *= mix(1.0, on, max(back, 1.0-u_dash.z));
  }
  a *= mix(1.0, u_dash.w, back);
  if(u_fade.y>0.0) a *= clamp(1.0-(u_fade.x-v_t)/u_fade.y, 0.0, 1.0);
  a *= u_color.a;
  if(a<=0.002) discard;
  o = u_additive>0.5 ? vec4(u_color.rgb*a,1.0) : vec4(u_color.rgb,a);
}`;

/* the sites of the level lattice, and the marker at the cloud's center */
const VS_NODE = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_corner;
layout(location=1) in vec3 a_center;
layout(location=2) in vec2 a_vd;
${COMMON}
uniform float u_radius, u_extent;
out vec2 v_uv; out float v_val; out float v_deg; out float v_pxR; out float v_zv;
void main(){
  vec4 vp = u_view*vec4(toWorld(a_center),1.0);
  vec4 c = u_proj*vp; c.xy = c.xy*u_group.z + u_group.xy*c.w;
  float pxR = u_radius*u_pxPerUnit/c.w;
  c.xy += a_corner*(pxR*u_extent+1.0)/(u_viewport*0.5)*c.w;
  gl_Position = c;
  v_uv = a_corner*(u_extent+1.0/pxR);
  v_val = a_vd.x; v_deg = a_vd.y; v_pxR = pxR; v_zv = vp.z;
}`;
const FS_NODE_A = `#version 300 es
precision highp float;
${RAMP}${LIT}
uniform float u_degMax, u_gain, u_base;
in vec2 v_uv; in float v_val; in float v_deg; in float v_pxR;
out vec4 o;
void main(){
  float r2 = dot(v_uv,v_uv);
  float lit = litOf(v_val, 0.55);
  float core = exp(-r2*5.0), halo = exp(-r2*0.55)*0.16, seat = exp(-r2*9.0)*u_base;
  o = vec4(ramp(v_deg/u_degMax)*((core+halo)*lit*u_gain+seat),1.0);
}`;
const FS_NODE_B = `#version 300 es
precision highp float;
${LIT}
uniform vec3 u_ink, u_spot; uniform float u_lw, u_cap;
in vec2 v_uv; in float v_val; in float v_pxR;
out vec4 o;
void main(){
  float r = length(v_uv)*v_pxR;
  float ra = smoothstep(u_lw*0.5+0.6, u_lw*0.5-0.4, abs(r-v_pxR))*0.85;
  float lg = logLit(v_val);
  float amp = mix(min(sqrt(max(v_val,0.0)), u_cap), lg > 0.0 ? 0.18 + 1.25*lg : 0.0, u_log);   // radius of the filled disc, in units of the ring
  float fr = v_pxR*amp;
  float fa = smoothstep(fr+0.6, fr-0.6, r)*0.88*step(0.06,amp);
  float a = ra + fa*(1.0-ra);
  if(a<=0.003) discard;
  o = vec4((u_ink*ra + u_spot*fa*(1.0-ra))/a, a);
}`;
const FS_NODE_C = `#version 300 es
precision highp float;
uniform mat4 u_view; uniform mat4 u_proj;
${RAMP}${SHADE}${LIT}
uniform float u_degMax, u_radW;
in vec2 v_uv; in float v_val; in float v_deg; in float v_pxR; in float v_zv;
out vec4 o;
void main(){
  float r2 = dot(v_uv,v_uv);
  if(r2>1.0) discard;
  float nz = sqrt(1.0-r2);
  mat3 v2w = transpose(mat3(u_view));
  float lit = litOf(v_val, 0.7);
  vec3 hue = ramp(v_deg/u_degMax);
  vec3 base = mix(vec3(0.84,0.86,0.89), hue, 0.14+0.62*lit);
  vec3 col = shadeBall(base, vec3(v_uv,nz), v2w, 70.0, 1.0, lit, hue*(0.22+0.95*nz*nz));
  float zv = v_zv + u_radW*nz;
  gl_FragDepth = ((u_proj[2][2]*zv + u_proj[3][2])/(-zv))*0.5+0.5;
  o = vec4(col, clamp((1.0-sqrt(r2))*v_pxR+0.5, 0.0, 1.0));
}`;
const VS_DISC = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_corner;
layout(location=1) in vec3 a_center;
layout(location=2) in vec2 a_vd;
${COMMON}
uniform vec3 u_light; uniform float u_floorY, u_discR;
out vec2 v_uv; out float v_val; out float v_deg; out float v_h;
void main(){
  vec3 wp = toWorld(a_center);
  float hgt = wp.y-u_floorY;
  vec3 f = wp - u_light*(hgt/u_light.y);
  vec3 q = f + vec3(a_corner.x,0.0,a_corner.y)*u_discR*(1.0+0.30*hgt);
  gl_Position = worldToClip(q);
  v_uv = a_corner; v_val = a_vd.x; v_deg = a_vd.y; v_h = hgt;
}`;
const FS_DISC = `#version 300 es
precision highp float;
${RAMP}${LIT}
uniform float u_degMax, u_mode; uniform vec3 u_tint;
in vec2 v_uv; in float v_val; in float v_deg; in float v_h;
out vec4 o;
void main(){
  float f = exp(-dot(v_uv,v_uv)*3.2)-0.04;
  if(f<=0.0) discard;
  float soft = 1.0/(1.0+0.8*v_h);
  if(u_mode<0.5) o = vec4(u_tint, f*0.30*soft);
  else o = vec4(ramp(v_deg/u_degMax), f*0.45*litOf(v_val, 0.7)*soft);
}`;

const PR = {
  ptA: program(VS_PT_A, FS_PT_A, 'ptA'), down: program(VS_FULL, FS_DOWN, 'down'), up: program(VS_FULL, FS_UP, 'up'), comp: program(VS_FULL, FS_COMP, 'comp'),
  bgB: program(VS_FULL, FS_BG_B, 'bgB'),
  ptC: program(VS_PT_C, FS_PT_C, 'ptC'), depth: program(VS_DEPTH, FS_DEPTH, 'depth'), dens: program(VS_DENS, FS_DENS, 'dens'),
  blur: program(VS_FULL, FS_BLUR, 'blur'), bgC: program(VS_FULL, FS_BG_C, 'bgC'), floor: program(VS_FLOOR, FS_FLOOR, 'floor'), glass: program(VS_GLASS, FS_GLASS, 'glass'),
  line: program(VS_LINE, FS_LINE, 'line'), stA: program(VS_STREAK, FS_STREAK_A, 'stA'), stB: program(VS_STREAK, FS_STREAK_B, 'stB'),
  nodeA: program(VS_NODE, FS_NODE_A, 'nodeA'), nodeB: program(VS_NODE, FS_NODE_B, 'nodeB'), nodeC: program(VS_NODE, FS_NODE_C, 'nodeC'), disc: program(VS_DISC, FS_DISC, 'disc'),
};

/* ---------------------------------------------------------------- geometry */
const ptBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ptBuf); gl.bufferData(gl.ARRAY_BUFFER, pos.byteLength, gl.DYNAMIC_DRAW);
const ptVao = gl.createVertexArray(); gl.bindVertexArray(ptVao); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

const streakBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, streakBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0,-1, 0,1, 0.2,-1, 0.2,1, 0.4,-1, 0.4,1, 0.6,-1, 0.6,1, 0.8,-1, 0.8,1, 1,-1, 1,1]), gl.STATIC_DRAW);
const streakVao = gl.createVertexArray(); gl.bindVertexArray(streakVao); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.bindBuffer(gl.ARRAY_BUFFER, ptBuf); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(1, 1);

const fullBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, fullBuf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
const fullVao = gl.createVertexArray(); gl.bindVertexArray(fullVao); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

const quadBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
const quadVao = gl.createVertexArray(); gl.bindVertexArray(quadVao); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

const lineCorner = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, lineCorner); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0,-1, 0,1, 1,-1, 1,1]), gl.STATIC_DRAW);

function makeBatch(maxSeg) {
  const data = new Float32Array(maxSeg*8);
  const vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo); gl.bufferData(gl.ARRAY_BUFFER, data.byteLength, gl.DYNAMIC_DRAW);
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, lineCorner); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 0); gl.vertexAttribDivisor(1, 1);
  gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 32, 12); gl.vertexAttribDivisor(2, 1);
  gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 32, 24); gl.vertexAttribDivisor(3, 1);
  gl.bindVertexArray(null);
  return { data, vbo, vao, count: 0, max: maxSeg };
}
/* append a polyline of n points (xyz triples); t is arc length unless tvals is given */
function addPolyline(b, pts, n, closed, tvals) {
  const d = b.data; let t = 0;
  const nseg = closed ? n : n-1;
  for (let i = 0; i < nseg && b.count < b.max; i++) {
    const j = (i+1) % n, o = b.count*8;
    const ax = pts[i*3], ay = pts[i*3+1], az = pts[i*3+2], bx = pts[j*3], by = pts[j*3+1], bz = pts[j*3+2];
    const len = Math.hypot(bx-ax, by-ay, bz-az);
    d[o] = ax; d[o+1] = ay; d[o+2] = az; d[o+3] = bx; d[o+4] = by; d[o+5] = bz;
    if (tvals) { d[o+6] = tvals[i]; d[o+7] = tvals[j]; } else { d[o+6] = t; d[o+7] = t+len; }
    t += len; b.count++;
  }
}
function upload(b) { gl.bindBuffer(gl.ARRAY_BUFFER, b.vbo); gl.bufferSubData(gl.ARRAY_BUFFER, 0, b.data.subarray(0, Math.max(1, b.count)*8)); }

function makeNodes(n) {
  const data = new Float32Array(n*5);
  const vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo); gl.bufferData(gl.ARRAY_BUFFER, data.byteLength, gl.DYNAMIC_DRAW);
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 20, 0); gl.vertexAttribDivisor(1, 1);
  gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 20, 12); gl.vertexAttribDivisor(2, 1);
  gl.bindVertexArray(null);
  return { data, vbo, vao, count: n };
}
function uploadNodes(s) { gl.bindBuffer(gl.ARRAY_BUFFER, s.vbo); gl.bufferSubData(gl.ARRAY_BUFFER, 0, s.data); }

/* unit sphere for the glass shell */
const sphere = (function () {
  const nu = 96, nv = 48, v = [], ix = [];
  for (let j = 0; j <= nv; j++) { const th = Math.PI*j/nv;
    for (let i = 0; i <= nu; i++) { const ph = 2*Math.PI*i/nu; v.push(Math.sin(th)*Math.cos(ph), Math.cos(th), Math.sin(th)*Math.sin(ph)); } }
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) { const a = j*(nu+1)+i, b = a+nu+1; ix.push(a, a+1, b, b, a+1, b+1); }
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(ix), gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { vao, n: ix.length };
})();

/* the lattice of levels: site (n1,n2,n3) sits at n1 u1 + n2 u2 + n3 u3, a corner of
   the octant standing on its tip, so height is the total number of excitations */
const LS = 0.46;                                          // spacing between neighboring sites
const LH = LS/Math.sqrt(3);                               // height gained per excitation, once the frame is turned
const sitePos = (n) => [n[0]*LS, n[1]*LS, n[2]*LS];       // site (n1, n2, n3): n_i steps along axis i
/* The lattice's outline as seen from the camera: its corner (site 000) at the bottom, and the
   highest point of the circle around its top layer, which is the same for every turn of the scene. */
const TOP_R = D*LS*Math.sqrt(2/3);                        // radius of the circle through the top layer's corners
const levelsTop = () => unrot([-TOP_R*Math.sin(cam.yaw), D*LH, -TOP_R*Math.cos(cam.yaw)]);
const levelsTopCenter = unrot([0, D*LH, 0]), levelsCorner = [0, 0, 0];
const tagLo = document.getElementById('lab-lo'), tagHi = document.getElementById('lab-hi');   // the two tags on the lattice
const levels = makeNodes(NS);
sites.forEach((n, s) => { const p = sitePos(n); levels.data.set([p[0], p[1], p[2], 0, n[0]+n[1]+n[2]], s*5); });
const head = makeNodes(1); head.data.set([0, 0, 0, 1, 1]);

const B = {
  rings: makeBatch(3*128), limb: makeBatch(160), axes: makeBatch(64), orbit: makeBatch(3200), trail: makeBatch(1600),
  edges: makeBatch(64),
};
(function staticLines() {
  const n = 128, c = new Float32Array(n*3);
  for (let k = 0; k < 3; k++) {                       // three great circles of the shell |X| = R0
    for (let i = 0; i < n; i++) { const a = 2*Math.PI*i/n, p = [0,0,0]; p[k] = R0*Math.cos(a); p[(k+1)%3] = R0*Math.sin(a); c.set(p, i*3); }
    addPolyline(B.rings, c, n, true);
  }
  upload(B.rings);
  const A = 3.3, tk = 0.07;
  for (let k = 0; k < 3; k++) {                       // axes with unit ticks
    const p0 = [0,0,0], p1 = [0,0,0]; p0[k] = -A; p1[k] = A; addPolyline(B.axes, [...p0, ...p1], 2, false);
    for (let m = -3; m <= 3; m++) { if (!m) continue; const q0 = [0,0,0], q1 = [0,0,0]; q0[k] = m; q1[k] = m; q0[(k+1)%3] = -tk; q1[(k+1)%3] = tk; addPolyline(B.axes, [...q0, ...q1], 2, false); }
  }
  upload(B.axes);
  const corner = (k, i) => { const n = [0, 0, 0]; n[i] = k; return sitePos(n); };
  for (let i = 0; i < 3; i++) addPolyline(B.edges, [...sitePos([0, 0, 0]), ...corner(D, i)], 2, false);   // one oscillator excited alone
  for (let k = 1; k <= D; k++) addPolyline(B.edges, [...corner(k, 0), ...corner(k, 1), ...corner(k, 2)], 3, true);  // all sites with k excitations
  upload(B.edges);
})();

/* ------------------------------------------------------------ scene state */
const DIRS = [[0.55,0.30,0.78], [-0.38,0.78,0.50], [0.80,-0.42,-0.43], [0.22,0.62,-0.75], [-0.72,-0.28,0.63], [0.48,0.80,0.36]].map(norm);
let dirIdx = 0, simTime = 0, throwAnim = null, settle = 0, throws = 0;
const TR = 1600, trailP = new Float32Array(TR*3), trailT = new Float32Array(TR); let trailN = 0;

function flowStep(x, dt) {                             // one split step of the force alone, with g = 1
  let c = Math.cos(x[0]*dt), sn = Math.sin(x[0]*dt);
  const y2 = x[1]*c + x[2]*sn, y3 = x[2]*c - x[1]*sn;
  c = Math.cos(y3*dt); sn = Math.sin(y3*dt);
  const z1 = x[0]*c + y2*sn, z2 = y2*c - x[0]*sn;
  c = Math.cos(z1*dt); sn = Math.sin(z1*dt);
  return [z1, z2*c + y3*sn, y3*c - z2*sn];
}
function computeOrbit(x0) {                             // the path of x0 under the force alone
  const pts = []; let x = x0.slice(), left = false; const dt = 0.003;
  for (let i = 0; i < 9000; i++) {
    if (i % 3 === 0) pts.push(x[0], x[1], x[2]);
    const xn = flowStep(x, dt);
    const stepLen = Math.hypot(xn[0]-x[0], xn[1]-x[1], xn[2]-x[2]);
    x = xn;
    const d = Math.hypot(x[0]-x0[0], x[1]-x0[1], x[2]-x0[2]);
    if (!left && d > 0.3) left = true;
    if (left && d < Math.max(2*stepLen, 0.01)) { pts.push(x0[0], x0[1], x0[2]); break; }
  }
  B.orbit.count = 0; addPolyline(B.orbit, pts, pts.length/3, false); upload(B.orbit);
}
/* A throw restarts the system from the paper's initial distribution: fresh runs drawn from the
   resting Gaussian shifted to a new point x0. During the throw each dot flies from where it was to
   its new starting point along
       X(e) = c(e) + cos(th) (X_old - m) + sin(th) sqrt(p) xi,   th = (pi/2) e,
   with m the old center, c(e) moving from m to x0, xi a fresh standard normal draw and e running
   from 0 to 1. A round cloud keeps its width throughout, and any old shape fades out on the way.
   At e = 1 the dots are exactly x0 + sqrt(p) xi. The clock stops during the flight. */
const throwFrom = new Float32Array(MMAX*3), throwNoise = new Float32Array(MMAX*3);
function startThrow(dir) {
  const d = dir || DIRS[dirIdx++ % DIRS.length];
  const tgt = [d[0]*R0, d[1]*R0, d[2]*R0];
  centroid();
  throwFrom.set(pos);
  for (let i = 0; i < MMAX*3; i++) throwNoise[i] = GT[rngIdx()];
  throwAnim = { from: mean.slice(), to: tgt, w: restWidth(), u: 0 };
  computeOrbit(tgt); trailN = 0; settle = 0; throws++;
}
function throwFrame(u) {
  const T = throwAnim, e = ease(u), m = T.from, t = T.to;
  const c = u >= 1 ? 0 : Math.cos(0.5*Math.PI*e), sn = (u >= 1 ? 1 : Math.sin(0.5*Math.PI*e))*T.w;
  const cx = m[0] + (t[0]-m[0])*e, cy = m[1] + (t[1]-m[1])*e, cz = m[2] + (t[2]-m[2])*e;
  for (let i = 0; i < MMAX*3; i += 3) {
    pos[i]   = cx + c*(throwFrom[i]   - m[0]) + sn*throwNoise[i];
    pos[i+1] = cy + c*(throwFrom[i+1] - m[1]) + sn*throwNoise[i+1];
    pos[i+2] = cz + c*(throwFrom[i+2] - m[2]) + sn*throwNoise[i+2];
  }
}
function pushTrail() {
  if (trailN > 0) { const o = (trailN-1)*3; if (Math.hypot(mean[0]-trailP[o], mean[1]-trailP[o+1], mean[2]-trailP[o+2]) < 0.012) return; }
  if (trailN >= TR) { trailP.copyWithin(0, (TR/2)*3); trailT.copyWithin(0, TR/2); trailN = TR/2; }
  trailP.set(mean, trailN*3); trailT[trailN] = simTime; trailN++;
}
const ease = (u) => u*u*(3 - 2*u);
function advance(dtReal) {
  if (throwAnim) {                                        // the cloud is in flight: no dynamics yet
    throwAnim.u = Math.min(1, throwAnim.u + dtReal/0.55);
    throwFrame(throwAnim.u);
    if (throwAnim.u >= 1) { throwAnim = null; trailN = 0; }
    measure(0.3); centroid();
    return;
  }
  const want = dtReal*SPEED*P.rate;                       // scene time to cover in this frame
  const nsub = Math.min(4, Math.max(1, Math.ceil(want/0.018)));
  const h = Math.min(want/nsub, 0.035);                   // a slow machine falls behind the requested speed instead of choking
  for (let k = 0; k < nsub; k++) step(h, P.M, P.lambda, P.q, P.g);
  simTime += nsub*h;
  measure(0.3); centroid();
  if (!throwAnim) pushTrail();
  if (P.auto && !throwAnim) { settle = excitedLight() < 0.03 ? settle + dtReal : 0; if (settle > 2.4) startThrow(); }
}

/* -------------------------------------------------------- camera & layout */
const cam = { yaw: 0.62, pitch: 0.42, dist: 12.5, fov: 34*Math.PI/180, view: null, proj: null, eye: [0,0,1], dir: [0,0,1], f: 1 };
const G = { cloud: { off: [-0.23, 0.02], scale: 1, model: [0, 0, 0, 1] }, levels: { off: [0.60, 0], scale: 1, model: [0, -D*LH/2, 0, 1] } };   // the lattice's middle height sits at 0
let W = 2, H = 2, cssW = 2, cssH = 2, dpr = 1, wide = true;
const titleEl = document.querySelector('.head h1');
function fitTitle() {                                     // keep the title on one line: shrink it only if it would not fit
  if (!titleEl) return;
  titleEl.style.fontSize = '';
  const avail = titleEl.parentElement.clientWidth, need = titleEl.scrollWidth;
  if (need > avail && avail > 0) titleEl.style.fontSize = (parseFloat(getComputedStyle(titleEl).fontSize)*avail/need*0.98).toFixed(1) + 'px';
}
if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitTitle);
function layout() {
  if (AMBIENT) {                                          // the scene fills the frame: cloud on the left, ladder on the right
    wide = true; root.dataset.layout = 'wide';
    const s = Math.min(1.15, (cssW/cssH)/1.9), c = P.look === 'c';
    G.cloud.off = [-0.43, 0.0]; G.cloud.scale = s*(c ? 0.92 : 1.0);
    G.levels.off = [0.47, 0.04]; G.levels.scale = s*(c ? 1.35 : 1.45);
    return;
  }
  fitTitle();
  const aspect = cssW/cssH; wide = aspect >= 1.2;
  root.dataset.layout = wide ? 'wide' : 'tall';
  const c = P.look === 'c';
  if (wide) {
    const s = Math.min(1, aspect/1.7);
    G.cloud.off = [-0.27, c ? 0.13 : 0.04]; G.cloud.scale = s*(c ? 0.80 : 0.93);
    G.levels.off = [0.54, c ? 0.05 : 0.02]; G.levels.scale = s*(c ? 1.25 : 1.35);
    fitLevels();
  } else {                                               // stacked: fit both objects between the header and the controls
    const ppu = 0.5*cssH/Math.tan(cam.fov/2)/cam.dist;    // CSS pixels per world unit at zoom 1
    const gutter = Math.min(34, Math.max(16, 0.024*cssW));
    const top = document.querySelector('.side').getBoundingClientRect().bottom + 6;
    const bot = cssH - document.getElementById('controls').offsetHeight - 2*gutter;
    const room = Math.max(200, bot - top), yCloud = top + 0.33*room, yLevels = top + 0.79*room;
    G.cloud.off = [0, 1 - 2*yCloud/cssH]; G.cloud.scale = Math.min(0.30*room/(2.9*ppu), 0.48*cssW/(3.1*ppu))*(c ? 0.9 : 1);
    G.levels.off = [0, 1 - 2*yLevels/cssH]; G.levels.scale = Math.min(0.185*room/(1.8*ppu), 0.46*cssW/(2.3*ppu));
  }
}
/* Wide layout: keep the levels, with their two tags, between the buttons at the top right
   and the caption below them, shrinking and moving the lattice only when it does not fit. */
function fitLevels() {
  if (!cam.view) return;
  const cap = document.getElementById('cap-levels').getBoundingClientRect();
  if (!cap.height) return;
  const top = document.querySelector('.side').getBoundingClientRect().bottom + 14 + tagHi.offsetHeight + 8;   // room for the top tag
  const bot = cap.top - 18 - tagLo.offsetHeight - 8;                                                       // room for the bottom tag
  const base = (p) => 1 - 2*project(p, { off: [0, 0], scale: 1, model: G.levels.model })[1]/cssH;
  const bHi = base(levelsTop()), bLo = base(levelsCorner);
  if (!(bHi > bLo) || bot <= top) return;
  const g1 = { off: [0, 0], scale: 1, model: G.levels.model };                      // widest part: the circle around the top layer
  const half1 = Math.abs(project(unrot([TOP_R*Math.cos(cam.yaw), D*LH, -TOP_R*Math.sin(cam.yaw)]), g1)[0] - project(levelsTopCenter, g1)[0]);
  const cx = (G.levels.off[0]*0.5 + 0.5)*cssW, halfMax = Math.min(cssW - 28 - cx, cx - 0.52*cssW);
  const s = Math.min(G.levels.scale, 2*(bot - top)/((bHi - bLo)*cssH), half1 > 0 ? halfMax/half1 : Infinity);
  let o = G.levels.off[1];
  const py = (b) => 0.5*cssH*(1 - b*s - o);
  if (py(bHi) < top) o = 1 - bHi*s - 2*top/cssH;
  if (py(bLo) > bot) o = 1 - bLo*s - 2*bot/cssH;
  G.levels.scale = s; G.levels.off[1] = o;
}
function updateCam() {
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  cam.eye = [cam.dist*cp*Math.sin(cam.yaw), cam.dist*sp, cam.dist*cp*Math.cos(cam.yaw)];
  cam.dir = norm(cam.eye); cam.view = lookAt(cam.eye, [0,0,0], [0,1,0]);
  cam.f = 1/Math.tan(cam.fov/2); cam.proj = perspective(cam.fov, W/H, 0.5, 60);
}
function project(p, grp) {
  const m = grp.model, w = rot(p), x = w[0]*m[3]+m[0], y = w[1]*m[3]+m[1], z = w[2]*m[3]+m[2], v = cam.view, pr = cam.proj;
  const vx = v[0]*x+v[4]*y+v[8]*z+v[12], vy = v[1]*x+v[5]*y+v[9]*z+v[13], vz = v[2]*x+v[6]*y+v[10]*z+v[14];
  const nx = pr[0]*vx/(-vz)*grp.scale+grp.off[0], ny = pr[5]*vy/(-vz)*grp.scale+grp.off[1];
  return [(nx*0.5+0.5)*cssW, (0.5-ny*0.5)*cssH];
}

/* render targets */
const T = { scene: null, mips: [], shA: null, shB: null };
const SH = 384, SM = 1024;
T.shA = target(SH, SH, true); T.shB = target(SH, SH, true);
const shadowMap = (function () {
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, SM, SM, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, tex, 0);
  gl.drawBuffers([gl.NONE]); gl.readBuffer(gl.NONE);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { tex, fbo };
})();
function resize() {
  const w = canvas.clientWidth || window.innerWidth, h = canvas.clientHeight || window.innerHeight;
  const d = Math.min(window.devicePixelRatio || 1, 2);
  if (w === cssW && h === cssH && d === dpr) return;
  cssW = w; cssH = h; dpr = d; W = Math.max(2, Math.round(w*d)); H = Math.max(2, Math.round(h*d));
  canvas.width = W; canvas.height = H;
  freeTarget(T.scene); T.mips.forEach(freeTarget); T.mips = [];
  T.scene = target(W, H, true);
  let mw = W, mh = H; for (let i = 0; i < 6; i++) { mw = Math.max(2, mw >> 1); mh = Math.max(2, mh >> 1); T.mips.push(target(mw, mh, true)); }
  layout();
}

/* ------------------------------------------------------------------ looks */
const lin = (a) => Float32Array.from(a, (v) => Math.pow(v, 2.2));          // palettes are chosen in display space and lit in linear light
const RAMP_A = lin([1.00,0.78,0.36, 1.00,0.36,0.34, 0.74,0.30,1.00, 0.20,0.50,1.00, 0.30,0.95,0.92]);
const RAMP_C = new Float32Array([1.00,0.74,0.28, 0.97,0.42,0.38, 0.68,0.45,0.88, 0.30,0.56,0.92, 0.36,0.84,0.82]);
const INK = [0.078, 0.086, 0.106], SPOT = [0.10, 0.22, 0.86], PAPER = [0.948, 0.936, 0.906];
const LIGHT = norm([0.38, 1.0, 0.30]);
const LX = norm(cross([0, 0, 1], LIGHT)), LY = cross(LIGHT, LX);
const FLOOR_Y = -3.7, FLOOR_E = 8.0, SH_E = 6.5, SH_Z = 12.0;
const BEADS = 10000, BEAD = 0.098;
const LOOKA = { streak: 0.07, head: 0.04, dot: 0.030, bloom: 0.12, exposure: 1.5, width: 1.1, cap: 0.2, rmax: 5.0, sparkBase: 0.5, spark: 14.0, node: 1.9 };
const LOOKB = { stroke: 0.55, width: 0.8, cap: 0.2, out: 0.85, count: 40000 };
const LIGHT_RANGE = [1e-3, 10];   // a level site's glow runs from dark at light 0.001 to full at light 10, on a log scale

let cur = null;
function use(pr, grp) {
  cur = pr; gl.useProgram(pr.p);
  if (!grp) return;
  const u = pr.u;
  gl.uniformMatrix4fv(u.u_view, false, cam.view); gl.uniformMatrix4fv(u.u_proj, false, cam.proj);
  gl.uniform3f(u.u_group, grp.off[0], grp.off[1], grp.scale);
  gl.uniform4f(u.u_model, grp.model[0], grp.model[1], grp.model[2], grp.model[3]);
  gl.uniformMatrix3fv(u.u_rot, false, ROT.gl);
  gl.uniform1f(u.u_pxPerUnit, 0.5*H*cam.f*grp.scale*grp.model[3]);
  gl.uniform2f(u.u_viewport, W, H);
}
function lines(b, grp, o) {
  if (!b.count) return;
  use(PR.line, grp); const u = cur.u;
  gl.uniform4fv(u.u_color, o.color); gl.uniform1f(u.u_width, o.width*dpr);
  gl.uniform4fv(u.u_dash, o.dash || [0, 1, 0, 1]); gl.uniform2fv(u.u_fade, o.fade || [0, 0]);
  gl.uniform1f(u.u_additive, o.additive ? 1 : 0); gl.uniform3fv(u.u_camDir, cam.dir); gl.uniform1f(u.u_capLen, o.additive ? 0 : Math.min(0.5, o.width*dpr*0.5));
  gl.bindVertexArray(b.vao); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b.count);
}
function nodes(pr, set, grp, radius, extent, setup) {
  use(pr, grp); const u = cur.u;
  gl.uniform1f(u.u_radius, radius); gl.uniform1f(u.u_extent, extent); gl.uniform1f(u.u_degMax, D);
  gl.uniform1f(u.u_log, set === levels ? 1 : 0); gl.uniform2f(u.u_lrange, Math.log(LIGHT_RANGE[0]), Math.log(LIGHT_RANGE[1]));
  if (setup) setup(u);
  gl.bindVertexArray(set.vao); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, set.count);
}
function fullscreen(pr) { use(pr); gl.bindVertexArray(fullVao); }
function tri() { gl.drawArrays(gl.TRIANGLES, 0, 3); }
function bindTex(unit, tex, loc) { gl.activeTexture(gl.TEXTURE0+unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(loc, unit); }

const TAU = 0.05;                                        // a streak shows the last TAU of motion under the force alone
function streaks(pr, setup, count, cap) {
  use(pr, G.cloud); const u = cur.u;
  gl.uniform1f(u.u_g, P.g); gl.uniform1f(u.u_tau, TAU); gl.uniform1f(u.u_dist, cam.dist); gl.uniform1f(u.u_cap, cap || 0.6);
  setup(u);
  gl.bindVertexArray(streakVao); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 12, Math.min(P.M, count || P.M));
}

function drawA() {
  gl.bindFramebuffer(gl.FRAMEBUFFER, T.scene.fbo); gl.viewport(0, 0, W, H);
  gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
  const add = true, dens = Math.sqrt(90000/P.M);
  lines(B.rings, G.cloud, { color: [0.42, 0.56, 1.0, 0.20], width: 1.0, additive: add, dash: [0, 1, 0, 0.35] });
  lines(B.limb, G.cloud, { color: [0.42, 0.56, 1.0, 0.16], width: 1.0, additive: add });
  lines(B.axes, G.cloud, { color: [0.62, 0.68, 0.86, 0.07], width: 1.0, additive: add });
  lines(B.orbit, G.cloud, { color: [1.0, 0.80, 0.52, 0.22], width: 1.1, additive: add, dash: [0, 1, 0, 0.4] });
  streaks(PR.stA, (u) => { gl.uniform3fv(u.u_ramp, RAMP_A); gl.uniform1f(u.u_width, LOOKA.width*dpr); gl.uniform1f(u.u_intensity, LOOKA.streak*dens); gl.uniform1f(u.u_rmax, LOOKA.rmax); gl.uniform2f(u.u_spark, LOOKA.sparkBase, LOOKA.spark); }, 0, LOOKA.cap);
  use(PR.ptA, G.cloud);
  gl.uniform3fv(cur.u.u_ramp, RAMP_A); gl.uniform1f(cur.u.u_size, LOOKA.dot); gl.uniform1f(cur.u.u_intensity, LOOKA.head*dens); gl.uniform1f(cur.u.u_rmax, LOOKA.rmax); gl.uniform1f(cur.u.u_dist, cam.dist);
  gl.bindVertexArray(ptVao); gl.drawArrays(gl.POINTS, 0, P.M);
  lines(B.trail, G.cloud, { color: [1.0, 0.80, 0.42, 0.95], width: 1.7, additive: add, fade: [simTime, 7.0] });
  nodes(PR.nodeA, head, G.cloud, 0.085, 3.0, (u) => { gl.uniform3fv(u.u_ramp, RAMP_A); gl.uniform1f(u.u_gain, 2.2); gl.uniform1f(u.u_base, 0.0); });
  lines(B.edges, G.levels, { color: [0.55, 0.66, 1.0, 0.13], width: 1.0, additive: add });
  nodes(PR.nodeA, levels, G.levels, 0.092, 3.0, (u) => { gl.uniform3fv(u.u_ramp, RAMP_A); gl.uniform1f(u.u_gain, LOOKA.node); gl.uniform1f(u.u_base, 0.10); });

  gl.disable(gl.BLEND);                                   // bloom: blur at six scales, then add them up
  fullscreen(PR.down);
  let src = T.scene;
  for (let i = 0; i < T.mips.length; i++) {
    const dst = T.mips[i]; gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo); gl.viewport(0, 0, dst.w, dst.h);
    bindTex(0, src.tex, cur.u.u_tex); gl.uniform2f(cur.u.u_texel, 1/src.w, 1/src.h); tri(); src = dst;
  }
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
  fullscreen(PR.up);
  for (let i = T.mips.length-1; i > 0; i--) {
    const sm = T.mips[i], dst = T.mips[i-1]; gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo); gl.viewport(0, 0, dst.w, dst.h);
    bindTex(0, sm.tex, cur.u.u_tex); gl.uniform2f(cur.u.u_texel, 1/sm.w, 1/sm.h); gl.uniform1f(cur.u.u_k, 1.0); tri();
  }
  gl.disable(gl.BLEND);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H);
  fullscreen(PR.comp);
  bindTex(0, T.scene.tex, cur.u.u_scene); bindTex(1, T.mips[0].tex, cur.u.u_bloom);
  gl.uniform1f(cur.u.u_bloomK, LOOKA.bloom); gl.uniform1f(cur.u.u_exposure, LOOKA.exposure); gl.uniform2f(cur.u.u_res, W, H);
  gl.uniform2f(cur.u.u_focus, 0.5+0.5*G.cloud.off[0], 0.5+0.5*G.cloud.off[1]);
  tri();
}

function drawB() {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H);
  gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.disable(gl.BLEND);
  fullscreen(PR.bgB); gl.uniform2f(cur.u.u_res, W, H); gl.uniform3fv(cur.u.u_paper, PAPER); tri();
  gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  const ink = (a) => [INK[0], INK[1], INK[2], a], spot = (a) => [SPOT[0], SPOT[1], SPOT[2], a];
  lines(B.axes, G.cloud, { color: ink(0.34), width: 0.8 });
  streaks(PR.stB, (u) => { gl.uniform3fv(u.u_ink, INK); gl.uniform3fv(u.u_spot, SPOT); gl.uniform1f(u.u_width, LOOKB.width*dpr);
    gl.uniform1f(u.u_alpha, LOOKB.stroke); gl.uniform1f(u.u_shellR, R0); gl.uniform1f(u.u_out, LOOKB.out); }, LOOKB.count, LOOKB.cap);
  lines(B.rings, G.cloud, { color: ink(0.62), width: 0.9, dash: [0.20, 0.5, 1, 0.55] });
  lines(B.limb, G.cloud, { color: ink(0.78), width: 1.1 });
  lines(B.orbit, G.cloud, { color: spot(0.55), width: 1.0, dash: [0.16, 0.55, 0, 0.45] });
  lines(B.trail, G.cloud, { color: spot(1.0), width: 1.9, fade: [simTime, 9.0] });
  const setB = (u) => { gl.uniform3fv(u.u_ink, INK); gl.uniform3fv(u.u_spot, SPOT); gl.uniform1f(u.u_lw, 1.1*dpr); gl.uniform1f(u.u_cap, 2.3); };
  nodes(PR.nodeB, head, G.cloud, 0.07, 2.4, setB);
  lines(B.edges, G.levels, { color: ink(0.42), width: 0.8 });
  nodes(PR.nodeB, levels, G.levels, 0.062, 2.4, setB);
}

function drawC() {
  const nb = Math.min(P.M, BEADS);
  gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(true);
  gl.bindFramebuffer(gl.FRAMEBUFFER, shadowMap.fbo); gl.viewport(0, 0, SM, SM); gl.clearDepth(1); gl.clear(gl.DEPTH_BUFFER_BIT);
  use(PR.depth); gl.uniformMatrix3fv(cur.u.u_rot, false, ROT.gl); const lightU = (u) => { gl.uniform3fv(u.u_Lx, LX); gl.uniform3fv(u.u_Ly, LY); gl.uniform3fv(u.u_L, LIGHT); gl.uniform1f(u.u_shE, SH_E); gl.uniform1f(u.u_shZ, SH_Z); };
  lightU(cur.u); gl.uniform1f(cur.u.u_ps, BEAD/(2*SH_E)*SM);
  gl.bindVertexArray(ptVao); gl.drawArrays(gl.POINTS, 0, nb);

  gl.disable(gl.DEPTH_TEST); gl.depthMask(false);                  // soft shadow of the cloud on the floor
  gl.bindFramebuffer(gl.FRAMEBUFFER, T.shA.fbo); gl.viewport(0, 0, SH, SH); gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
  use(PR.dens); gl.uniformMatrix3fv(cur.u.u_rot, false, ROT.gl); gl.uniform3fv(cur.u.u_light, LIGHT); gl.uniform1f(cur.u.u_floorY, FLOOR_Y); gl.uniform1f(cur.u.u_extent, FLOOR_E);
  gl.uniform1f(cur.u.u_ps, 9.0); gl.uniform1f(cur.u.u_k, (HALF ? 0.020 : 0.05)*9000/nb);
  gl.bindVertexArray(ptVao); gl.drawArrays(gl.POINTS, 0, nb);
  gl.disable(gl.BLEND);
  fullscreen(PR.blur);
  for (let i = 0; i < 2; i++) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.shB.fbo); bindTex(0, T.shA.tex, cur.u.u_tex); gl.uniform2f(cur.u.u_dir, 1.6/SH, 0); tri();
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.shA.fbo); bindTex(0, T.shB.tex, cur.u.u_tex); gl.uniform2f(cur.u.u_dir, 0, 1.6/SH); tri();
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H); gl.clearDepth(1); gl.clear(gl.DEPTH_BUFFER_BIT);
  fullscreen(PR.bgC); gl.uniform2f(cur.u.u_res, W, H); tri();
  gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  const tint = [0.16, 0.20, 0.30], floorL = G.levels.model[1] - 0.30;
  use(PR.floor, G.cloud); bindTex(0, T.shA.tex, cur.u.u_tex);
  gl.uniform1f(cur.u.u_floorY, FLOOR_Y); gl.uniform1f(cur.u.u_extent, FLOOR_E); gl.uniform1f(cur.u.u_strength, 0.50); gl.uniform3fv(cur.u.u_tint, tint);
  gl.uniform3f(cur.u.u_shell, -LIGHT[0]*(-FLOOR_Y/LIGHT[1]), -LIGHT[2]*(-FLOOR_Y/LIGHT[1]), R0);
  gl.bindVertexArray(quadVao); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  for (let mode = 0; mode < 2; mode++) {
    use(PR.disc, G.levels); const u = cur.u;
    gl.uniform3fv(u.u_ramp, RAMP_C); gl.uniform1f(u.u_degMax, D); gl.uniform1f(u.u_mode, mode); gl.uniform3fv(u.u_tint, tint);
    gl.uniform1f(u.u_log, 1); gl.uniform2f(u.u_lrange, Math.log(LIGHT_RANGE[0]), Math.log(LIGHT_RANGE[1]));
    gl.uniform3fv(u.u_light, LIGHT); gl.uniform1f(u.u_floorY, floorL); gl.uniform1f(u.u_discR, mode ? 0.34 : 0.22);
    gl.bindVertexArray(levels.vao); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, levels.count);
  }

  gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); gl.depthMask(true); gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
  use(PR.ptC, G.cloud); let u = cur.u;
  gl.uniform3fv(u.u_ramp, RAMP_C); gl.uniform1f(u.u_size, BEAD); gl.uniform1f(u.u_radW, BEAD/2); gl.uniform3fv(u.u_light, LIGHT);
  lightU(u); gl.uniform1f(u.u_shTexel, 1.1/SM); gl.uniform1f(u.u_bias, 0.016/(2*SH_Z)); gl.uniform1f(u.u_haze, 0.55); gl.uniform1f(u.u_dist, cam.dist);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, shadowMap.tex); gl.uniform1i(u.u_shadow, 0);
  gl.bindVertexArray(ptVao); gl.drawArrays(gl.POINTS, 0, nb);
  const setC = (r) => (uu) => { gl.uniform3fv(uu.u_ramp, RAMP_C); gl.uniform3fv(uu.u_light, LIGHT); gl.uniform1f(uu.u_radW, r); };
  nodes(PR.nodeC, head, G.cloud, 0.10, 1.0, setC(0.10));
  nodes(PR.nodeC, levels, G.levels, 0.105, 1.0, setC(0.105));
  gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);

  gl.enable(gl.BLEND); gl.depthMask(false);
  lines(B.edges, G.levels, { color: [0.20, 0.25, 0.34, 0.42], width: 1.0 });
  lines(B.rings, G.cloud, { color: [1.0, 1.0, 1.0, 0.62], width: 1.0, dash: [0, 1, 0, 0.45] });      // lines etched on the glass
  lines(B.limb, G.cloud, { color: [0.20, 0.27, 0.38, 0.30], width: 1.2 });
  lines(B.orbit, G.cloud, { color: [0.14, 0.18, 0.26, 0.42], width: 1.1 });
  lines(B.trail, G.cloud, { color: [1.0, 0.36, 0.10, 1.0], width: 2.2, fade: [simTime, 9.0] });
  gl.disable(gl.DEPTH_TEST);
  lines(B.trail, G.cloud, { color: [1.0, 0.36, 0.10, 0.22], width: 2.2, fade: [simTime, 9.0] });
  gl.enable(gl.DEPTH_TEST);

  gl.enable(gl.CULL_FACE); use(PR.glass, G.cloud); u = cur.u;
  gl.uniform1f(u.u_R, R0); gl.uniform3fv(u.u_eye, cam.eye); gl.uniform3fv(u.u_light, LIGHT); gl.bindVertexArray(sphere.vao);
  gl.cullFace(gl.FRONT); gl.uniform1f(u.u_face, -1); gl.drawElements(gl.TRIANGLES, sphere.n, gl.UNSIGNED_SHORT, 0);
  gl.cullFace(gl.BACK); gl.uniform1f(u.u_face, 1); gl.drawElements(gl.TRIANGLES, sphere.n, gl.UNSIGNED_SHORT, 0);
  gl.disable(gl.CULL_FACE); gl.depthMask(true);
}

/* ------------------------------------------------------------------- frame */
const limbPts = new Float32Array(160*3);
function dynamicGeometry() {
  const d = cam.dist, c = cam.dir, rr = R0*Math.sqrt(1 - R0*R0/(d*d)), sh = R0*R0/d;   // outline of the shell as seen from the eye
  const rt = norm(cross([0,1,0], c)), up = cross(c, rt);
  for (let i = 0; i < 160; i++) { const a = 2*Math.PI*i/160, ca = Math.cos(a)*rr, sa = Math.sin(a)*rr;
    limbPts.set(unrot([c[0]*sh + rt[0]*ca + up[0]*sa, c[1]*sh + rt[1]*ca + up[1]*sa, c[2]*sh + rt[2]*ca + up[2]*sa]), i*3); }
  B.limb.count = 0; addPolyline(B.limb, limbPts, 160, true); upload(B.limb);
  B.trail.count = 0; if (trailN > 1) { addPolyline(B.trail, trailP, trailN, false, trailT); upload(B.trail); }
  for (let s = 0; s < NS; s++) levels.data[s*5+3] = phi[s]*phi[s];
  uploadNodes(levels);
  head.data[0] = mean[0]; head.data[1] = mean[1]; head.data[2] = mean[2]; uploadNodes(head);
  gl.bindBuffer(gl.ARRAY_BUFFER, ptBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos.subarray(0, P.M*3));
}

const labels = {
  x1: [document.getElementById('lab-x1'), [3.6, 0, 0], 'cloud'], x2: [document.getElementById('lab-x2'), [0, 3.6, 0], 'cloud'], x3: [document.getElementById('lab-x3'), [0, 0, 3.6], 'cloud'],
  
};
const capCloud = document.getElementById('cap-cloud'), capLevels = document.getElementById('cap-levels');
function placeLabels() {
  const put = (el, x, y, dy) => { el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px) translate(-50%,${dy})`; };
  for (const k in labels) { const [el, p, g] = labels[k]; const s = project(p, G[g]); put(el, s[0], s[1], '-50%'); }
  const lo = project(levelsCorner, G.levels), top = project(levelsTop(), G.levels), mid = project(levelsTopCenter, G.levels);
  put(tagLo, lo[0], lo[1] + 18, '0%');                  // just below the corner
  put(tagHi, mid[0], top[1] - 14, '-100%');             // just above the top layer
  if (wide) {
    capCloud.style.left = ((G.cloud.off[0]*0.5+0.5)*100).toFixed(2) + '%';
    capLevels.style.left = ((G.levels.off[0]*0.5+0.5)*100).toFixed(2) + '%';
  }
}

let last = 0, idle = 0, dragging = false, slow = 0, frames = 0, simMs = 0;
function frame(tms) {
  const now = tms*0.001; const dtReal = Math.min(Math.max(now - last, 0), 0.05); last = now;
  if (AMBIENT && !ambientVisible) { requestAnimationFrame(frame); return; }   // scrolled out of view: do nothing
  resize();
  const t0 = performance.now();
  if (!P.paused) advance(dtReal);
  simMs += 0.05*(performance.now() - t0 - simMs);
  idle += dtReal;
  if (!dragging && idle > 2.5 && !reduceMotion && !P.paused) cam.yaw += dtReal*0.07;
  updateCam(); dynamicGeometry();
  if (P.look === 'a') drawA(); else if (P.look === 'b') drawB(); else drawC();
  placeLabels();
  frames++;
  slow = dtReal > 0.030 ? slow + 1 : Math.max(0, slow - 2);
  if (slow > 120 && P.M > 30000 && !P.paused) { P.M = Math.round(P.M*0.75); slow = 0; showRunCount(); }
  requestAnimationFrame(frame);
}

/* ---------------------------------------------------------------------- UI */
const fmt = new Intl.NumberFormat('en-US');
function showRunCount() {                                // the caption under the cloud says how many runs are drawn
  const el = document.getElementById('run-count');
  if (el) el.textContent = fmt.format(P.look === 'c' ? Math.min(P.M, BEADS) : P.M);
}
if (AMBIENT) window.addEventListener('message', (e) => {
  const d = e.data || {};
  if (d.type === 'qsv-theme') setLook(d.theme === 'dark' ? 'a' : 'b');
  if (d.type === 'qsv-visible') ambientVisible = !!d.visible;
});
function setLook(k) {
  P.look = k; root.dataset.look = k;
  document.querySelectorAll('.looks button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.look === k)));
  layout(); showRunCount();
}
document.querySelectorAll('.looks button').forEach(b => b.addEventListener('click', () => setLook(b.dataset.look)));
window.addEventListener('keydown', (e) => {
  if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
  const k = { '1': 'a', '2': 'b', '3': 'c', a: 'a', b: 'b', c: 'c' }[e.key.toLowerCase()];
  if (k) setLook(k); else if (e.key === ' ') { e.preventDefault(); togglePause(); } else if (e.key.toLowerCase() === 't') startThrow();
});
const gIn = document.getElementById('g'), lamIn = document.getElementById('lam'), qIn = document.getElementById('q'), speedIn = document.getElementById('speed');
function syncSliders() {
  gIn.value = P.g; lamIn.value = P.lambda; qIn.value = P.q; speedIn.value = P.rate;
  document.getElementById('g-out').textContent = P.g.toFixed(2); document.getElementById('lam-out').textContent = P.lambda.toFixed(2);
  document.getElementById('q-out').textContent = P.q.toFixed(2);
  document.getElementById('speed-out').textContent = P.rate.toFixed(1) + '\u00d7';
}
gIn.addEventListener('input', () => { P.g = +gIn.value; syncSliders(); });
lamIn.addEventListener('input', () => { P.lambda = +lamIn.value; syncSliders(); });
qIn.addEventListener('input', () => { P.q = +qIn.value; syncSliders(); });
speedIn.addEventListener('input', () => { P.rate = +speedIn.value; syncSliders(); });
const controlsEl = document.getElementById('controls');              // captions sit just above the controls, however many rows they take
const setControlsH = () => { root.style.setProperty('--controls-h', controlsEl.offsetHeight + 'px'); if (cssW > 2) layout(); };
if (window.ResizeObserver) new ResizeObserver(setControlsH).observe(controlsEl);
window.addEventListener('resize', setControlsH); setControlsH();
document.getElementById('throw').addEventListener('click', () => { startThrow(); if (P.paused) togglePause(); });
const pauseBtn = document.getElementById('pause');
function togglePause() {
  P.paused = !P.paused; pauseBtn.setAttribute('aria-pressed', String(P.paused));
  pauseBtn.textContent = P.paused ? (pauseBtn.dataset.labelPlay || 'Play') : (pauseBtn.dataset.labelPause || 'Pause');
}
pauseBtn.addEventListener('click', togglePause);
const autoIn = document.getElementById('auto');
autoIn.addEventListener('change', () => { P.auto = autoIn.checked; settle = 0; });
controlsEl.addEventListener('submit', (e) => e.preventDefault());
const notes = document.getElementById('notes'), notesBtn = document.getElementById('notes-btn');
function toggleNotes(open) { const o = open === undefined ? notes.hidden : open; notes.hidden = !o; notesBtn.setAttribute('aria-expanded', String(o)); root.classList.toggle('notes-open', o); }
notesBtn.addEventListener('click', () => toggleNotes());
document.getElementById('notes-close').addEventListener('click', () => toggleNotes(false));

canvas.addEventListener('pointerdown', (e) => { dragging = true; idle = 0; canvas.setPointerCapture(e.pointerId); root.classList.add('dragged'); });
canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return; idle = 0;
  cam.yaw -= e.movementX*0.006; cam.pitch = Math.min(1.25, Math.max(-0.35, cam.pitch + e.movementY*0.005));
});
const endDrag = () => { dragging = false; idle = 0; };
canvas.addEventListener('pointerup', endDrag); canvas.addEventListener('pointercancel', endDrag);

/* ------------------------------------------------------------------- start */
function start() {
  const d0 = DIRS[dirIdx++];
  const x0 = [d0[0]*R0, d0[1]*R0, d0[2]*R0];
  fillGaussian(x0); computeOrbit(x0);
  measure(1); centroid();
  for (let i = 0; i < 12; i++) {                       // run ahead a little: the first frame shows the cloud just after a throw
    step(0.03, P.M, P.lambda, P.q, P.g); simTime += 0.03; measure(0.5); centroid(); pushTrail();
  }
  autoIn.checked = P.auto; syncSliders(); setLook(P.look);
  if (reduceMotion) { P.auto = false; autoIn.checked = false; togglePause(); }
  resize(); updateCam(); layout();
  requestAnimationFrame((t) => { last = t*0.001; requestAnimationFrame(frame); });
}
start();

/* ------------------------------------------------- checks used in testing */
window.__scene = {
  P, D, sites, phi, raw, cam, G, LOOKA, LOOKB, RAMP_A, RAMP_C,
  frames: () => frames, simMs: () => simMs,
  setLook, startThrow,
  tick(n, dtReal) { for (let i = 0; i < n; i++) advance(dtReal); let bad = 0; for (let i = 0; i < P.M*3; i++) if (!Number.isFinite(pos[i])) bad++; return { throws, simTime, light: excitedLight(), mean: mean.slice(), bad }; },
  amplitudes(x0, t, lambda, g, dt, q) {       // fresh displaced Gaussian, run for time t, return every measured amplitude
    if (q === undefined) q = 2*lambda;
    const sd = Math.sqrt(q/(2*lambda));
    fillGaussian(x0, sd); const n = 2*Math.max(1, Math.round(t/dt/2)); stepParity = 0;
    for (let i = 0; i < n; i++) step(t/n, MMAX, lambda, q, g);
    const nb = 9, per = MMAX/nb, batches = [];                         // batch means, so the caller can estimate the sampling error
    for (let b = 0; b < nb; b++) { measure(1, per, b*per, sd); batches.push(Array.from(raw)); }
    return batches;
  },
  throwNow(i) {                               // a throw with the flight skipped
    startThrow(DIRS[i % DIRS.length]); throwFrame(1); throwAnim = null; trailN = 0;
    measure(1); centroid(); pushTrail();
  },
  inFlight: () => !!throwAnim,
  flight(u) { if (throwAnim) { throwAnim.u = u; throwFrame(u); measure(1); centroid(); } },
  run(t) {
    const n = Math.max(1, Math.ceil(t/0.015));
    for (let i = 0; i < n; i++) { step(t/n, P.M, P.lambda, P.q, P.g); simTime += t/n; measure(0.5); centroid(); pushTrail(); }
  },
  timeStep(n) { const t0 = performance.now(); for (let i = 0; i < n; i++) { step(0.015, P.M, P.lambda, P.q, P.g); } const t1 = performance.now(); for (let i = 0; i < n; i++) measure(0.3); return [(t1-t0)/n, (performance.now()-t1)/n]; },
  selfTest() {
    const out = {}, keep = pos.slice(), keepRs = rs, fact = [1, 1, 2, 6, 24, 120, 720];
    /* 1. the force alone keeps |X| */
    fillGaussian([1.2, -0.7, 0.9], 1);
    const r0 = new Float64Array(2000); for (let p = 0; p < 2000; p++) r0[p] = Math.hypot(pos[p*3], pos[p*3+1], pos[p*3+2]);
    for (let i = 0; i < 400; i++) step(0.013, 2000, 0, 0, 1.4);
    let dev = 0; for (let p = 0; p < 2000; p++) dev = Math.max(dev, Math.abs(Math.hypot(pos[p*3], pos[p*3+1], pos[p*3+2])/r0[p] - 1));
    out.normDrift = dev;
    /* 2. a displaced Gaussian is a coherent state: phi_n = prod x_i^n_i / sqrt(n_i!).
          Report the worst deviation in units of the sampling error, whose variance is
          prod L_n_i(-x_i^2) - phi_n^2 over the number of particles (L = Laguerre polynomial). */
    const x0 = [1.1, -0.8, 1.3]; fillGaussian(x0, 1); measure(1, MMAX, 0, 1);
    const binom = (n, k) => fact[n]/(fact[k]*fact[n-k]);
    const lag = (n, y) => { let v = 0; for (let k = 0; k <= n; k++) v += binom(n, k)*Math.pow(y, k)/fact[k]; return v; };
    let worst = 0, worstAbs = 0; sites.forEach((n, i) => {
      const ex = Math.pow(x0[0], n[0])*Math.pow(x0[1], n[1])*Math.pow(x0[2], n[2])/Math.sqrt(fact[n[0]]*fact[n[1]]*fact[n[2]]);
      const varr = lag(n[0], x0[0]*x0[0])*lag(n[1], x0[1]*x0[1])*lag(n[2], x0[2]*x0[2]) - ex*ex;
      if (i > 0) worst = Math.max(worst, Math.abs(raw[i] - ex)/Math.sqrt(varr/MMAX));
      worstAbs = Math.max(worstAbs, Math.abs(raw[i] - ex)); });
    out.coherentWorstZ = worst; out.coherentWorstAbs = worstAbs;
    /* 3. the resting Gaussian has no light above the bottom site, and the dynamics keeps it so */
    fillGaussian([0, 0, 0], 1); measure(1, MMAX, 0, 1);
    let e0 = 0; for (let s = 1; s < NS; s++) e0 = Math.max(e0, Math.abs(raw[s])); out.restMaxAmp = e0;
    for (let i = 0; i < 300; i++) step(0.013, MMAX, 0.22, 0.44, 1.4);
    measure(1, MMAX, 0, 1); let e1 = 0; for (let s = 1; s < NS; s++) e1 = Math.max(e1, Math.abs(raw[s])); out.restAfterRun = e1;
    out.restExpectedMax = 3.5/Math.sqrt(MMAX);
    /* 4. the same at another width: lambda = 0.3, q = 0.9, so p = 1.5 */
    const sdW = Math.sqrt(0.9/(2*0.3));
    fillGaussian([0, 0, 0], sdW);
    for (let i = 0; i < 300; i++) step(0.013, MMAX, 0.3, 0.9, 1.4);
    measure(1, MMAX, 0, sdW); let e2 = 0; for (let s = 1; s < NS; s++) e2 = Math.max(e2, Math.abs(raw[s])); out.restAfterRunWide = e2;
    /* 5. a throw from a swirled cloud (mid-relaxation) lands on the coherent state of the new x0 */
    const keepP = [P.lambda, P.q]; P.lambda = 0.22; P.q = 0.44;
    fillGaussian([1.4, -0.9, 1.0], 1); for (let i = 0; i < 40; i++) step(0.013, MMAX, 0.22, 0.44, 1.4);
    startThrow(DIRS[1]); throwFrame(0.5);
    const xm = throwAnim.to; throwFrame(1); throwAnim = null; trailN = 0;
    measure(1, MMAX, 0, 1);
    let worst5 = 0; sites.forEach((n, i) => { if (i === 0) return;
      const ex = Math.pow(xm[0], n[0])*Math.pow(xm[1], n[1])*Math.pow(xm[2], n[2])/Math.sqrt(fact[n[0]]*fact[n[1]]*fact[n[2]]);
      const varr = lag(n[0], xm[0]*xm[0])*lag(n[1], xm[1]*xm[1])*lag(n[2], xm[2]*xm[2]) - ex*ex;
      worst5 = Math.max(worst5, Math.abs(raw[i] - ex)/Math.sqrt(varr/MMAX)); });
    out.throwLandsWorstZ = worst5;
    P.lambda = keepP[0]; P.q = keepP[1];
    pos.set(keep); rs = keepRs; measure(1); centroid();
    return out;
  },
};
})();
