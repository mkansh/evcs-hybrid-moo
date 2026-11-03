"""IGD+, additive epsilon and bootstrap equivalence (TOST via 90% CI) for the five-solver confirmation (seeds 300-329)."""
import pickle, numpy as np, sys
from model import *
from algos import nds
names = ['NSGA-II', 'HNSGA-PSO', 'MOEA/D', 'NSGA-III', 'MOPSO']
DELTA = 0.02          # equivalence margin on HV, IGD+ and epsilon (normalised objectives); fixed before the computation
rng = np.random.default_rng(12345)

def feas_nd(F, cv):
    f = F[cv <= 1e-12]
    if len(f) == 0: return f
    f = f[nds(f, np.zeros(len(f))) == 0]
    return np.unique(np.round(f, 10), axis=0)
def hv2d(P, r):
    if len(P) == 0: return 0.
    P = P[np.argsort(P[:, 0])]; h = 0.; ymin = r[1]
    for p in P:
        if p[1] < ymin: h += (r[0]-p[0])*(ymin-p[1]); ymin = p[1]
    return h
R = np.array([1.1, 1.1, 1.1])
def boot_ci(x, y, paired=False, B=10000, lo=5, hi=95):
    n = len(x)
    if paired:
        d = x - y; idx = rng.integers(0, n, (B, n)); m = d[idx].mean(1)
    else:
        ix = rng.integers(0, n, (B, n)); iy = rng.integers(0, len(y), (B, len(y)))
        m = x[ix].mean(1) - y[iy].mean(1)
    return np.percentile(m, [lo, hi])

res = {}
for tag, sc in (('', 'legacy'), ('_daily', 'daily')):
    G = {e: pickle.load(open(f'res_extra{tag}_{e}.pkl', 'rb'))['res'] for e in ('real', 'discrete')}
    allp = np.vstack([feas_nd(r['F'], r['cv']) for e in G for n in names for r in G[e][n]])
    ref = np.unique(allp[nds(allp, np.zeros(len(allp))) == 0], axis=0)
    ideal, nadir = ref.min(0), ref.max(0); norm = lambda f: (f-ideal)/(nadir-ideal)
    Rn = norm(ref)
    def hv3d(F):
        if len(F) == 0: return 0.
        P = norm(F); P = P[np.all(P < R, 1)]
        if len(P) == 0: return 0.
        P = P[np.argsort(P[:, 2])]; h = 0.
        for i in range(len(P)):
            z1 = P[i, 2]; z2 = P[i+1, 2] if i+1 < len(P) else R[2]
            h += hv2d(P[:i+1, :2], R[:2])*(z2-z1)
        return h
    def igdplus(F):
        if len(F) == 0: return np.nan
        A = norm(F); d = np.sqrt((np.maximum(A[None, :, :] - Rn[:, None, :], 0)**2).sum(-1)).min(1)
        return d.mean()
    def eps(F):
        if len(F) == 0: return np.nan
        A = norm(F); return (A[None, :, :] - Rn[:, None, :]).max(-1).min(1).max()
    for e in ('real', 'discrete'):
        M = {}
        for n in names:
            hv, ig, ep = [], [], []
            for r in G[e][n]:
                f = feas_nd(r['F'], r['cv']); hv.append(hv3d(f)); ig.append(igdplus(f)); ep.append(eps(f))
            M[n] = dict(hv=np.array(hv), igd=np.array(ig), eps=np.array(ep))
        for n in names:                      # empty fronts -> worst observed value of the setting
            for k in ('igd', 'eps'):
                allv = np.concatenate([M[m][k] for m in names]); worst = np.nanmax(allv)
                M[n][k] = np.where(np.isnan(M[n][k]), worst, M[n][k])
        out = {}
        for n in names:
            row = dict(hv_med=np.median(M[n]['hv']), igd_med=np.median(M[n]['igd']), eps_med=np.median(M[n]['eps']))
            if n != 'NSGA-II':
                for k in ('hv', 'igd', 'eps'):
                    ci = boot_ci(M[n][k], M['NSGA-II'][k], paired=(n == 'HNSGA-PSO'))
                    row[k+'_diff'] = M[n][k].mean() - M['NSGA-II'][k].mean(); row[k+'_ci'] = ci
                    row[k+'_equiv'] = bool(ci[0] > -DELTA and ci[1] < DELTA)
            out[n] = row
        res[f'{sc}_{e}'] = out
        print(f'\n=== {sc} / {e} (margin +-{DELTA}) ===')
        print(f"{'solver':10s} {'medHV':>6s} {'medIGD+':>8s} {'medEps':>7s} | HV diff [90% CI] equiv | IGD+ diff [90% CI] equiv | Eps diff [90% CI] equiv")
        for n in names:
            r = out[n]
            if n == 'NSGA-II': print(f"{n:10s} {r['hv_med']:6.3f} {r['igd_med']:8.4f} {r['eps_med']:7.4f}"); continue
            print(f"{n:10s} {r['hv_med']:6.3f} {r['igd_med']:8.4f} {r['eps_med']:7.4f} | {r['hv_diff']:+.3f} [{r['hv_ci'][0]:+.3f},{r['hv_ci'][1]:+.3f}] {'Y' if r['hv_equiv'] else 'n'} | "
                  f"{r['igd_diff']:+.4f} [{r['igd_ci'][0]:+.4f},{r['igd_ci'][1]:+.4f}] {'Y' if r['igd_equiv'] else 'n'} | {r['eps_diff']:+.4f} [{r['eps_ci'][0]:+.4f},{r['eps_ci'][1]:+.4f}] {'Y' if r['eps_equiv'] else 'n'}")
pickle.dump(res, open('equiv.pkl', 'wb'))
