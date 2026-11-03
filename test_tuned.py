"""Seeds 800-829: default continuous, tuned continuous and default integer encoding for each algorithm (legacy scenario).
Usage: python3 test_tuned.py ALG1 ALG2 ..."""
import sys, os, copy, pickle, time, numpy as np
os.environ['EVCS_ENC'] = 'real'
import algos, algos_extra, tune
from model import Problem, make_scenarios
from hvtools import feas_nd, hv3d
SEEDS = range(800, 830); sc = make_scenarios(); DEF = copy.deepcopy(algos.PARAMS)
def go(alg, cfg, enc):
    algos.ENC = enc; hv = []; Fs = []
    for s in SEEDS:
        algos.PARAMS.update(DEF); algos.PARAMS.update({k: v for k, v in cfg.items() if k in DEF})
        p = Problem(sc)
        if alg == 'NSGA-II': o = algos.nsga2(p, s, False)
        elif alg == 'HNSGA-PSO': o = algos.nsga2(p, s, True, ppso=cfg.get('ppso', 0.5))
        elif alg == 'MOEA/D': o = algos.moead(p, s, T=cfg.get('T', 10), delta=cfg.get('delta', 0.9), nr=cfg.get('nr', 2), CR=cfg.get('CR', 1.0), Fd=cfg.get('Fd', 0.5))
        elif alg == 'NSGA-III': o = algos_extra.nsga3(p, s)
        else: o = algos_extra.mopso(p, s)
        X, F, cv, _ = o; f = feas_nd(F, cv); hv.append(hv3d(f)); Fs.append(f)
    return np.array(hv), Fs
for alg in sys.argv[1:]:
    best = pickle.load(open(f'tune_{alg.replace("/", "_")}.pkl', 'rb'))['best']
    out = {'best': best}
    for label, cfg, enc in (('default_cont', {}, 'real'), ('tuned_cont', best, 'real'), ('default_int', {}, 'discrete')):
        t0 = time.time(); hv, Fs = go(alg, cfg, enc); out[label] = hv; out[label+'_F'] = Fs
        print(alg, label, 'median %.3f mean %.3f SD %.3f poor(<0.35) %d (%.0f s)' % (np.median(hv), hv.mean(), hv.std(ddof=1), (hv < .35).sum(), time.time()-t0), flush=True)
    algos.PARAMS.update(DEF); algos.ENC = 'real'
    pickle.dump(out, open(f'test_{alg.replace("/", "_")}.pkl', 'wb'))
