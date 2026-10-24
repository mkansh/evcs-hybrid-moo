import numpy as np, pickle, time, sys
from model import *
from algos import *
sc = make_scenarios()
NR = 30
res = {}
t0 = time.time()
for name in ['NSGA-II', 'HNSGA-PSO', 'MOEA/D']:
    res[name] = []
    for r in range(NR):
        prob = Problem(sc)
        seed = 100 + r
        if name == 'NSGA-II': out = nsga2(prob, seed, False, hvfn=True)
        elif name == 'HNSGA-PSO': out = nsga2(prob, seed, True, hvfn=True)
        else: out = moead(prob, seed, hvfn=True)
        X, F, cv, tr = out
        res[name].append(dict(X=X, F=F, cv=cv, tr=tr, nfe=prob.nfe))
    print(name, 'done', time.time() - t0, 'nfe', res[name][0]['nfe'], flush=True)
pickle.dump(res, open('results.pkl', 'wb'))
