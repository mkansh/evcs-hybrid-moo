"""Exhaustive siting landscape (33-bus, legacy scenario, DG at upper bounds): objective vectors of all (hub pair, PV bus, WT bus) plans
and ruggedness of the loss objective under index-neighbour moves (+-1 bus index) versus random moves."""
import numpy as np, pickle, itertools, time
import model
from model import Problem, make_scenarios
sc = make_scenarios(); p = Problem(sc)
pairs = [(a, b) for a in range(1, 33) for b in range(a, 33)]                       # 528 unordered hub pairs
t0 = time.time(); L = np.full((528, 32, 32), np.nan); V = np.full((528, 32, 32), np.nan)
gene = lambda k: (k-1)/31.0
for i, (a, b) in enumerate(pairs):
    X = np.array([[gene(a), gene(b), gene(pb), gene(wb), 1.0, 1.0] for pb in range(1, 33) for wb in range(1, 33)])
    F, cv = p.evaluate(X); L[i] = F[:, 1].reshape(32, 32); V[i] = np.where(cv <= 1e-12, 0, 1).reshape(32, 32)
    if i % 100 == 0: print(i, '%.0f s' % (time.time()-t0), flush=True)
pickle.dump(dict(L=L, V=V, pairs=pairs), open('landscape.pkl', 'wb'))
feas = V == 0
print('plans', L.size, 'feasible', feas.sum(), 'loss range %.1f-%.1f MWh' % (np.nanmin(np.where(feas, L, np.nan)), np.nanmax(np.where(feas, L, np.nan))))
# ruggedness: correlation of loss between a plan and its +-1 index neighbour (PV bus or WT bus), versus a random partner
Lf = np.where(feas, L, np.nan)
def corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b); return np.corrcoef(a[m], b[m])[0, 1]
r_pv = corr(Lf[:, :-1, :].ravel(), Lf[:, 1:, :].ravel()); r_wt = corr(Lf[:, :, :-1].ravel(), Lf[:, :, 1:].ravel())
rng = np.random.default_rng(3); idx = rng.permutation(Lf.size); r_rand = corr(Lf.ravel(), Lf.ravel()[idx])
print('loss correlation, PV bus +-1 index neighbour: %.3f; WT bus +-1: %.3f; random partner: %.3f' % (r_pv, r_wt, r_rand))
# local optima of loss under +-1 index moves of PV/WT bus (hub pair fixed)
nloc = 0; npl = 0
for i in range(528):
    A = Lf[i]
    for pv in range(32):
        for wt in range(32):
            v = A[pv, wt]
            if np.isnan(v): continue
            nb = [A[q, wt] for q in (pv-1, pv+1) if 0 <= q < 32] + [A[pv, q] for q in (wt-1, wt+1) if 0 <= q < 32]
            nb = [x for x in nb if not np.isnan(x)]; npl += 1
            if nb and v < min(nb): nloc += 1
print('local optima of annual loss under +-1 index moves: %d of %d feasible plans (%.2f%%)' % (nloc, npl, 100*nloc/npl))
pickle.dump(dict(r_pv=r_pv, r_wt=r_wt, r_rand=r_rand, nloc=nloc, npl=npl), open('landscape_stats.pkl', 'wb'))
