import pickle, itertools, numpy as np
from scipy import stats
from algos import nds
names = ['NSGA-II', 'HNSGA-PSO', 'MOEA/D', 'NSGA-III', 'MOPSO']
G = {e: pickle.load(open(f'r69_{e}.pkl', 'rb')) for e in ('real', 'discrete')}
def feas_nd(F, cv):
    f = F[cv <= 1e-12]
    if len(f) == 0: return f
    f = f[nds(f, np.zeros(len(f))) == 0]; return np.unique(np.round(f, 8), axis=0)
fronts = {(e, n): [feas_nd(r['F'], r['cv']) for r in G[e]['res'][n]] for e in G for n in names}
ref = np.zeros((0, 4))
pts = np.vstack([f for k in fronts for f in fronts[k]]); print('union of run fronts:', len(pts))
for i in range(0, len(pts), 1500):
    c = np.vstack([ref, pts[i:i+1500]]); ref = c[nds(c, np.zeros(len(c))) == 0]
ref = np.unique(np.round(ref, 8), axis=0); print('pooled reference front:', len(ref), 'ranges', ref.min(0).round(3), ref.max(0).round(3))
ideal, nadir = ref.min(0), ref.max(0); norm = lambda f: (f-ideal)/(nadir-ideal); Rn = norm(ref); RP = np.full(4, 1.1)
def hv2d(P, r):
    if len(P) == 0: return 0.
    P = P[np.argsort(P[:, 0])]; h = 0.; ymin = r[1]
    for p in P:
        if p[1] < ymin: h += (r[0]-p[0])*(ymin-p[1]); ymin = p[1]
    return h
def hv3d(P, r):
    if len(P) == 0: return 0.
    P = P[np.argsort(P[:, 2])]; h = 0.
    for i in range(len(P)):
        z1 = P[i, 2]; z2 = P[i+1, 2] if i+1 < len(P) else r[2]; h += hv2d(P[:i+1, :2], r[:2])*(z2-z1)
    return h
def hv4d(F):
    if len(F) == 0: return 0.
    P = norm(F); P = P[np.all(P < RP, 1)]
    if len(P) == 0: return 0.
    P = P[np.argsort(P[:, 3])]; h = 0.
    for i in range(len(P)):
        z1 = P[i, 3]; z2 = P[i+1, 3] if i+1 < len(P) else RP[3]; h += hv3d(P[:i+1, :3], RP[:3])*(z2-z1)
    return h
def igdp(F):
    if len(F) == 0: return np.nan
    A = norm(F); return np.sqrt((np.maximum(A[None] - Rn[:, None], 0)**2).sum(-1)).min(1).mean()
hvref = hv4d(ref); print('HV of pooled reference: %.3f' % hvref)
M = {}
for e in G:
    for n in names:
        hv = np.array([hv4d(f) for f in fronts[(e, n)]]); ig = np.array([igdp(f) for f in fronts[(e, n)]]); M[(e, n)] = dict(hv=hv, igd=ig)
def a12(x, y): return (np.sum(x[:, None] > y[None]) + 0.5*np.sum(x[:, None] == y[None]))/(len(x)*len(y))
def holm(p):
    o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m-i)*p[k]); adj[k] = min(1, run)
    return adj
out = {'hvref': hvref, 'refn': len(ref)}
for e in ('real', 'discrete'):
    print(f'\n=== 69-bus + BESS, 4 objectives, {e} encoding (seeds 400-429) ===')
    d = {n: M[(e, n)]['hv'] for n in names}
    for n in names:
        x = d[n]; q = np.percentile(x, [25, 50, 75]); print('%-10s median HV %.3f [%.3f-%.3f] mean %.3f SD %.3f | median IGD+ %.4f | s/run %.1f' % (n, q[1], q[0], q[2], x.mean(), x.std(ddof=1), np.nanmedian(M[(e, n)]['igd']), G[e]['times'][n]))
    kw = stats.kruskal(*[d[n] for n in names]).pvalue
    pairs = list(itertools.combinations(range(5), 2)); pp = np.array([stats.ranksums(d[names[i]], d[names[j]]).pvalue for i, j in pairs]); ph = holm(pp)
    print('Kruskal-Wallis p = %.2e; Holm-adjusted rank-sum (10 pairs):' % kw)
    for (i, j), p, q in zip(pairs, pp, ph): print('   %-10s vs %-10s Holm p=%.4f A12=%.2f' % (names[i], names[j], q, a12(d[names[i]], d[names[j]])))
    out[e] = dict(kw=kw, med={n: float(np.median(d[n])) for n in names}, q1={n: float(np.percentile(d[n], 25)) for n in names}, q3={n: float(np.percentile(d[n], 75)) for n in names},
                  holm={(names[i], names[j]): float(q) for (i, j), q in zip(pairs, ph)}, a12={(names[i], names[j]): float(a12(d[names[i]], d[names[j]])) for i, j in pairs},
                  igd={n: float(np.nanmedian(M[(e, n)]['igd'])) for n in names}, sec={n: float(G[e]['times'][n]) for n in names})
print('\nEncoding effect per solver (integer vs continuous):')
out['enc'] = {}
for n in names:
    x, y = M[('discrete', n)]['hv'], M[('real', n)]['hv']; p = stats.mannwhitneyu(x, y, alternative='two-sided', method='asymptotic').pvalue
    print('  %-10s median %.3f -> %.3f  A12=%.2f  p=%.4f' % (n, np.median(y), np.median(x), a12(x, y), p)); out['enc'][n] = dict(a12=float(a12(x, y)), p=float(p))
pickle.dump(out, open('analysis69.pkl', 'wb'))
pickle.dump({k: v['hv'] for k, v in M.items()}, open('hv69.pkl', 'wb'))
# structure of the knee-like solution: BESS size chosen by the integer-encoding NSGA-II runs
for n in ('NSGA-II',):
    be = []; 
    for r in G['discrete']['res'][n]:
        m = r['cv'] <= 1e-12; X = r['X'][m]; F = r['F'][m]; k = np.argmin(np.sqrt((((F-F.min(0))/(F.max(0)-F.min(0)+1e-12))**2).sum(1)))
        be.append((X[k, 7]*2000, X[k, 8]*1000))
    be = np.array(be); print('knee BESS (kWh, kW) over runs, integer NSGA-II: median', np.median(be, 0).round(0), 'IQR E', np.percentile(be[:, 0], [25, 75]).round(0))
