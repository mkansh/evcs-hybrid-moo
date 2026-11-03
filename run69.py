"""Five solvers x 30 seeds on the 69-bus + BESS + 4-objective benchmark. Usage: EVCS_ENC=discrete python3 run69.py out.pkl [first_seed]"""
import sys, os, time, pickle, numpy as np
import algos, algos_extra, model69
from algos import nsga2, moead
from algos_extra import nsga3, mopso
algos.NBG = 5; algos.NCAND = 68
out = sys.argv[1]; first = int(sys.argv[2]) if len(sys.argv) > 2 else 400; NR = int(os.environ.get('NR', 30))
sc = model69.make_scenarios(); res = {}; times = {}
print('encoding:', algos.ENC, 'NBG', algos.NBG, 'NCAND', algos.NCAND, flush=True)
for name in ['NSGA-II', 'HNSGA-PSO', 'MOEA/D', 'NSGA-III', 'MOPSO']:
    res[name] = []; ts = time.time()
    for seed in range(first, first+NR):
        prob = model69.Problem69(sc)
        if name == 'NSGA-II': o = nsga2(prob, seed, False)
        elif name == 'HNSGA-PSO': o = nsga2(prob, seed, True)
        elif name == 'MOEA/D': o = moead(prob, seed)
        elif name == 'NSGA-III': o = nsga3(prob, seed)
        else: o = mopso(prob, seed)
        X, F, cv, _ = o; res[name].append(dict(X=X, F=F, cv=cv, nfe=prob.nfe, seed=seed))
    times[name] = (time.time()-ts)/NR
    print(name, 'done; s/run = %.2f; nfe = %d; feasible members of last run: %d' % (times[name], res[name][0]['nfe'], int((res[name][-1]['cv'] <= 1e-12).sum())), flush=True)
pickle.dump(dict(res=res, times=times), open(out, 'wb'))
