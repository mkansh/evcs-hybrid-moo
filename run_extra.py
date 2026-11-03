"""Five solvers x 30 fresh seeds (300-329) for one encoding.
Usage: EVCS_ENC=discrete python3 run_extra.py res_extra_discrete.pkl   (scenario set: legacy, as published)"""
import sys, time, pickle
import model, algos
from model import Problem, make_scenarios
from algos import nsga2, moead
from algos_extra import nsga3, mopso

out = sys.argv[1]
SEEDS = range(300, 330)
sc = make_scenarios()
print('scenario:', model.SCEN_MODE, '| encoding:', algos.ENC, flush=True)
res, times = {}, {}
for name in ['NSGA-II', 'HNSGA-PSO', 'MOEA/D', 'NSGA-III', 'MOPSO']:
    res[name] = []; ts = time.time()
    for seed in SEEDS:
        prob = Problem(sc)
        if name == 'NSGA-II': o = nsga2(prob, seed, False)
        elif name == 'HNSGA-PSO': o = nsga2(prob, seed, True)
        elif name == 'MOEA/D': o = moead(prob, seed)
        elif name == 'NSGA-III': o = nsga3(prob, seed)
        else: o = mopso(prob, seed)
        X, F, cv, tr = o
        res[name].append(dict(X=X, F=F, cv=cv, nfe=prob.nfe, seed=seed))
    times[name] = (time.time() - ts) / len(SEEDS)
    print(name, 'done; s/run = %.2f; nfe = %d' % (times[name], res[name][0]['nfe']), flush=True)
pickle.dump(dict(res=res, times=times), open(out, 'wb'))
