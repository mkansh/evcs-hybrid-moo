"""Ablation of the integer encoding components for NSGA-II (legacy scenario, seeds 700-729) + mutation-neutrality metrics."""
import os, sys, pickle, numpy as np, time
os.environ['EVCS_ENC'] = 'discrete'
import model, algos
from model import Problem, make_scenarios
from hvtools import feas_nd, hv3d
sc = make_scenarios()
VARIANTS = {'continuous (published)': None, 'integer init only': {'init'}, 'init + crossover': {'init', 'cross'}, 'init + mutation': {'init', 'mut'},
            'crossover + mutation (continuous init)': {'cross', 'mut'}, 'full integer encoding': {'init', 'cross', 'mut'}}
out = {}
for name, parts in VARIANTS.items():
    t0 = time.time()
    if parts is None: algos.ENC = 'real'
    else: algos.ENC = 'discrete'; algos.PARTS = parts
    hv = []
    for seed in range(700, 730):
        p = Problem(sc); X, F, cv, _ = algos.nsga2(p, seed, False); hv.append(hv3d(feas_nd(F, cv)))
    out[name] = np.array(hv); print('%-42s median %.3f mean %.3f SD %.3f  poor(<0.35): %d  (%.0f s)' % (name, np.median(hv), np.mean(hv), np.std(hv, ddof=1), (np.array(hv) < .35).sum(), time.time()-t0), flush=True)
pickle.dump(out, open('ablate.pkl', 'wb'))
# neutrality of the mutation operators on a bus gene: probability that a mutation event changes the decoded bus, and mean bus step
rng = np.random.default_rng(1)
n = 200000
x = rng.random(n)
u = rng.random(n); eta = 20.0
d = np.where(u < .5, (2*u)**(1/(eta+1)) - 1, 1 - (2*(1-u))**(1/(eta+1)))
y = np.clip(x + d, 0, 1)
bi = lambda g: np.clip(np.round(1 + g*31), 1, 32).astype(int)
ch = bi(y) != bi(x)
print('continuous polynomial mutation (eta=20): P(bus changes | gene mutated) = %.3f; mean |step| when changed = %.2f buses' % (ch.mean(), np.abs(bi(y)-bi(x))[ch].mean()))
k = rng.integers(1, 33, n); step = rng.integers(1, 4, n)*np.where(rng.random(n) < .5, -1, 1); reset = rng.random(n) < .5
knew = np.where(reset, rng.integers(1, 33, n), np.clip(k+step, 1, 32))
ch2 = knew != k
print('integer mutation: P(bus changes | gene mutated) = %.3f; mean |step| when changed = %.2f buses' % (ch2.mean(), np.abs(knew-k)[ch2].mean()))
pickle.dump(dict(p_real=float(ch.mean()), p_int=float(ch2.mean()), step_real=float(np.abs(bi(y)-bi(x))[ch].mean()), step_int=float(np.abs(knew-k)[ch2].mean())), open('neutrality.pkl', 'wb'))
