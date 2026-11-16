"""Check the page's measured oscillator amplitudes against the linear evolution of the oscillator state.

Run from the project folder, after  node tools/amplitudes.js :
    python tools/spectral_check.py tools/amplitudes.json [cutoff]       (needs numpy and scipy; cutoff defaults to 72)

The cloud's amplitudes phi_n(t) = E[e_n(X_t)], e_n = product of normalized Hermite polynomials, obey
    d phi / dt = L^T phi,   L = -A + C,
    A = lambda * sum_i a_i^dag a_i                       (damping: rate lambda per quantum)
    C = g [ x2 x3 d1 + x3 x1 d2 - 2 x1 x2 d3 ]           (the force), with x_i = a_i + a_i^dag, d_i = a_i,
starting from the coherent state phi_n(0) = prod_i x0_i^n_i / sqrt(n_i!).
This is the unit-width case q = 2 lambda; other q are reduced to it by rescaling (see below).
"""
import json, sys, itertools, math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

K = int(sys.argv[2]) if len(sys.argv) > 2 else 72   # cutoff on the total number of excitations; raise it if the result still moves
states = [(i, j, k) for i in range(K+1) for j in range(K+1-i) for k in range(K+1-i-j)]
index = {s: n for n, s in enumerate(states)}
dim = len(states)

def lower(i):
    rows, cols, vals = [], [], []
    for s, n in index.items():
        if s[i] > 0:
            t = list(s); t[i] -= 1
            rows.append(index[tuple(t)]); cols.append(n); vals.append(math.sqrt(s[i]))
    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))

a = [lower(i) for i in range(3)]
ad = [m.T.tocsr() for m in a]            # raising operators, cut off at K excitations
x = [a[i] + ad[i] for i in range(3)]
num = sum(ad[i] @ a[i] for i in range(3))

def generator(lam, g):
    C = g * (x[1] @ x[2] @ a[0] + x[2] @ x[0] @ a[1] - 2.0 * x[0] @ x[1] @ a[2])
    return -lam * num + C, C

data = json.load(open(sys.argv[1]))
sites = [tuple(s) for s in data['sites']]

# 1. the force term is anti-Hermitian (checked away from the cutoff, where truncation does not interfere)
_, C = generator(0.22, 1.4)
low = np.array([sum(s) <= K - 3 for s in states])
S = (C + C.T).tocsr()[low][:, low]
print('max |C + C^T| on states with at most %d excitations: %.2e   (largest entry of C: %.2f)' % (K-3, abs(S).max() if S.nnz else 0.0, abs(C).max()))
# and it only links neighbouring heights
coo = C.tocoo(); dh = {int(sum(states[r]) - sum(states[c])) for r, c, v in zip(coo.row, coo.col, coo.data) if abs(v) > 1e-9}
print('changes of total excitations produced by C:', sorted(dh))

for case in data['cases']:
    lam, g, t, x0 = case['lambda'], case['g'], case['t'], case['x0']
    q = case.get('q', 2*lam); p_var = q/(2*lam)
    # In Y = X/sqrt(p) the system is the unit-width one with force strength g*sqrt(p) and start x0/sqrt(p).
    g, x0 = g*math.sqrt(p_var), [v/math.sqrt(p_var) for v in x0]
    L, _ = generator(lam, g)
    phi0 = np.array([np.prod([x0[i]**s[i] / math.sqrt(math.factorial(s[i])) for i in range(3)]) for s in states])
    phi = expm_multiply((L.T * t).tocsc(), phi0)
    exact = np.array([phi[index[s]] for s in sites])
    B = np.array(case['batches'])                 # batch means from the page's simulation
    mean, se = B.mean(axis=0), B.std(axis=0, ddof=1) / math.sqrt(B.shape[0])
    z = (mean[1:] - exact[1:]) / se[1:]                                     # site 0 is the constant function: exactly 1 on both sides
    deg = np.array([sum(s) for s in sites])
    print('\nx0=%s t=%.1f lambda=%.2f q=%.2f g=%.1f dt=%.3f' % (case['x0'], t, lam, q, case['g'], case['dt']))
    print('  light on levels 1..6:  exact %.4f   measured %.4f   (coherent state at t=0: %.3f of total %.3f)' % (
        (exact[1:]**2).sum(), (mean[1:]**2).sum(), (np.array([phi0[index[s]] for s in sites])[1:]**2).sum(), math.exp(sum(v*v for v in x0)) - 1))
    print('  worst |z| = %.2f, rms z = %.2f over %d sites; worst abs error %.4f (largest amplitude %.3f)' % (abs(z).max(), math.sqrt((z**2).mean()), len(z), abs(mean - exact)[1:].max(), abs(exact[1:]).max()))
    for d in range(1, 7):
        m = deg == d
        print('    %d excitations: rms z %.2f, light exact %.4f measured %.4f' % (d, math.sqrt(((mean[m]-exact[m])**2 / se[m]**2).mean()), (exact[m]**2).sum(), (mean[m]**2).sum()))
