"""Analysis of the five-solver confirmation (seeds 300-329, legacy scenarios), both encodings.
HV is computed on the normalised pooled non-dominated front of all runs of both encodings (reference point 1.1)."""
import sys, pickle, itertools, collections, numpy as np
from scipy import stats
from model import *
from algos import nds

import os
TAG = os.environ.get('EXTRA_TAG', '')
names = ['NSGA-II', 'HNSGA-PSO', 'MOEA/D', 'NSGA-III', 'MOPSO']
G = {e: pickle.load(open(f'res_extra{TAG}_{e}.pkl', 'rb')) for e in ('real', 'discrete')}


def feas_nd(F, cv):
    f = F[cv <= 1e-12]
    if len(f) == 0: return f
    f = f[nds(f, np.zeros(len(f))) == 0]
    return np.unique(np.round(f, 10), axis=0)


def hv2d(P, r):
    if len(P) == 0: return 0.
    P = P[np.argsort(P[:, 0])]; h = 0.; ymin = r[1]
    for p in P:
        if p[1] < ymin: h += (r[0] - p[0]) * (ymin - p[1]); ymin = p[1]
    return h


R = np.array([1.1, 1.1, 1.1])
allp = np.vstack([feas_nd(r['F'], r['cv']) for e in G for n in names for r in G[e]['res'][n]])
ref = np.unique(allp[nds(allp, np.zeros(len(allp))) == 0], axis=0)
ideal, nadir = ref.min(0), ref.max(0)
norm = lambda f: (f - ideal) / (nadir - ideal)


def hv3d(F):
    if len(F) == 0: return 0.
    P = norm(F); P = P[np.all(P < R, 1)]
    if len(P) == 0: return 0.
    P = P[np.argsort(P[:, 2])]; h = 0.
    for i in range(len(P)):
        z1 = P[i, 2]; z2 = P[i + 1, 2] if i + 1 < len(P) else R[2]
        h += hv2d(P[:i + 1, :2], R[:2]) * (z2 - z1)
    return h


def a12(x, y): return (np.sum(x[:, None] > y[None]) + 0.5 * np.sum(x[:, None] == y[None])) / (len(x) * len(y))


def holm(p):
    o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1, run)
    return adj


hvref = hv3d(ref)
print(f"pooled reference front (seeds 300-329, both encodings, 5 solvers): {len(ref)} points, HV_ref = {hvref:.3f}")
HV = {}
for e in ('real', 'discrete'):
    HV[e] = {n: np.array([hv3d(feas_nd(r['F'], r['cv'])) for r in G[e]['res'][n]]) for n in names}
    print(f"\n=== encoding: {e} ===")
    print(f"{'solver':11s}{'mean HV':>8s}{'median':>8s}{'SD':>7s}{'rec% med':>9s}{'#<50%':>6s}{'#>=95%':>7s}{'s/run':>7s}")
    for n in names:
        h = HV[e][n]; rec = h / hvref
        print(f"{n:11s}{h.mean():8.3f}{np.median(h):8.3f}{h.std(ddof=1):7.3f}{100*np.median(rec):9.1f}{int((rec<.5).sum()):6d}{int((rec>=.95).sum()):7d}{G[e]['times'][n]:7.2f}")
    kw = stats.kruskal(*[HV[e][n] for n in names])
    fr = stats.friedmanchisquare(*[HV[e][n] for n in names])
    X = np.array([HV[e][n] for n in names]).T
    mr = np.array([stats.rankdata(-r) for r in X]).mean(0)
    print(f"Kruskal-Wallis H={kw.statistic:.2f}, p={kw.pvalue:.2e}; Friedman (descriptive, seeds as blocks) chi2={fr.statistic:.2f}, p={fr.pvalue:.2e}")
    print("mean Friedman ranks (1 = best): " + ", ".join(f"{n} {r:.2f}" for n, r in zip(names, mr)))
    pairs = list(itertools.combinations(range(5), 2))
    ps = np.array([stats.ranksums(HV[e][names[i]], HV[e][names[j]]).pvalue for i, j in pairs])
    pa = holm(ps)
    print("pairwise Wilcoxon rank-sum, Holm over 10 pairs (A12 = P(first > second)):")
    for (i, j), p, q in zip(pairs, ps, pa):
        print(f"   {names[i]:10s} vs {names[j]:10s}  p={p:.4f}  Holm p={q:.4f}  A12={a12(HV[e][names[i]], HV[e][names[j]]):.2f}")
    d = HV[e]['HNSGA-PSO'] - HV[e]['NSGA-II']
    w = stats.wilcoxon(HV[e]['HNSGA-PSO'], HV[e]['NSGA-II']).pvalue if np.any(d != 0) else 1.0
    print(f"paired signed-rank (shared initial populations) HNSGA-PSO vs NSGA-II: p={w:.4f}, mean diff={d.mean():+.4f}")
    # hub siting of the knee plan for the two Pareto-based baselines (diagnostic)
for n in names:
    x, y = HV['discrete'][n], HV['real'][n]
    print(f"encoding effect {n:10s}: median HV real {np.median(y):.3f} -> discrete {np.median(x):.3f}; A12={a12(x, y):.2f}")
pickle.dump(dict(HV=HV, hvref=hvref), open(f'analysis_extra{TAG}.pkl', 'wb'))
