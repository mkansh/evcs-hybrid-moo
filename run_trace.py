"""Five solvers x 30 seeds (300-329), legacy scenario: HV versus evaluations and final fronts. Usage: EVCS_ENC=real|discrete python3 run_trace.py out.pkl"""
import sys, os, pickle, time, numpy as np
import algos, algos_extra
from model import Problem, make_scenarios
from algos import nsga2, moead, nds
from algos_extra import nsga3, mopso
from hvtools import feas_nd, hv3d
sc = make_scenarios(); out = {}; t0 = time.time()
for name in ['NSGA-II', 'HNSGA-PSO', 'MOEA/D', 'NSGA-III', 'MOPSO']:
    out[name] = []
    for seed in range(300, 330):
        p = Problem(sc)
        if name == 'NSGA-II': o = nsga2(p, seed, False, hvfn=True)
        elif name == 'HNSGA-PSO': o = nsga2(p, seed, True, hvfn=True)
        elif name == 'MOEA/D': o = moead(p, seed, hvfn=True)
        elif name == 'NSGA-III': o = nsga3(p, seed, hvfn=True)
        else: o = mopso(p, seed, hvfn=True)
        X, F, cv, tr = o
        per = 55 if name == 'MOEA/D' else 60
        hv = np.array([hv3d(feas_nd(f, np.zeros(len(f)))) if len(f) else 0.0 for f in tr])
        out[name].append(dict(hv=hv, evals=per*(np.arange(len(tr))+1), F=feas_nd(F, cv), hv_final=hv3d(feas_nd(F, cv)), seed=seed))
    print(name, 'done %.0f s' % (time.time()-t0), flush=True)
pickle.dump(out, open(sys.argv[1], 'wb'))
