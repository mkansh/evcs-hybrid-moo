import pickle, numpy as np, itertools
from scipy import stats
names = ['NSGA-II', 'HNSGA-PSO', 'MOEA/D', 'NSGA-III', 'MOPSO']
T = {n: pickle.load(open('test_%s.pkl' % n.replace('/', '_'), 'rb')) for n in names}
def a12(x, y): return (np.sum(x[:, None] > y[None]) + 0.5*np.sum(x[:, None] == y[None]))/(len(x)*len(y))
def holm(p):
    o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m-i)*p[k]); adj[k] = min(1, run)
    return adj
print('seeds 800-829, legacy scenario; HV on the fixed normalisation of the pooled 300-329 front (reference HV 0.741)')
print('%-10s %-14s %6s %6s %6s %5s' % ('solver', 'setting', 'median', 'mean', 'SD', 'poor'))
for n in names:
    for k in ('default_cont', 'tuned_cont', 'default_int'):
        h = T[n][k]; print('%-10s %-14s %6.3f %6.3f %6.3f %5d' % (n, k, np.median(h), h.mean(), h.std(ddof=1), (h < .35).sum()))
rows = []
print('\nWithin-algorithm effects (Mann-Whitney, Holm over 10 tests; A12 = P(first > second)):')
ps = []; lab = []
for n in names:
    for a, b in (('tuned_cont', 'default_cont'), ('default_int', 'tuned_cont')):
        p = stats.mannwhitneyu(T[n][a], T[n][b], alternative='two-sided', method='asymptotic').pvalue; ps.append(p); lab.append((n, a, b))
pa = holm(np.array(ps))
for (n, a, b), p, q in zip(lab, ps, pa):
    print('  %-10s %-12s vs %-12s p=%.4f Holm=%.4f A12=%.2f  median diff %+.3f' % (n, a, b, p, q, a12(T[n][a], T[n][b]), np.median(T[n][a])-np.median(T[n][b])))
for setting in ('tuned_cont', 'default_int'):
    d = {n: T[n][setting] for n in names}
    kw = stats.kruskal(*[d[n] for n in names]).pvalue
    pairs = list(itertools.combinations(range(5), 2)); pp = np.array([stats.ranksums(d[names[i]], d[names[j]]).pvalue for i, j in pairs]); ph = holm(pp)
    print('\nBetween solvers, %s: Kruskal-Wallis p=%.2e' % (setting, kw))
    for (i, j), p, q in zip(pairs, pp, ph):
        print('   %-10s vs %-10s p=%.4f Holm=%.4f A12=%.2f' % (names[i], names[j], p, q, a12(d[names[i]], d[names[j]])))
pickle.dump(T, open('tuned_all.pkl', 'wb'))
