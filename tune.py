"""Random-search tuning of the continuous encoding, 8 sampled configurations + the default per algorithm, 10 tuning seeds (500-509).
Usage: python3 tune.py ALG1 ALG2 ...   -> tune_<alg>.pkl"""
import sys, os, copy, pickle, time, numpy as np
os.environ['EVCS_ENC'] = 'real'
import model, algos, algos_extra
from model import Problem, make_scenarios
from hvtools import feas_nd, hv3d
SEEDS = range(500, 510); NCFG = 8
sc = make_scenarios()
DEF = copy.deepcopy(algos.PARAMS)
SPACE = {
 'sbx_eta': (5, 30), 'sbx_pc': (0.6, 1.0), 'pm_eta': (10, 40), 'pm_scale': (0.5, 3.0),
 'w0': (0.3, 0.9), 'c1': (0.5, 2.0), 'c2': (0.5, 2.0), 'vmax': (0.1, 0.5), 'ppso': (0.1, 0.9)}
KEYS = {'NSGA-II': ['sbx_eta', 'sbx_pc', 'pm_eta', 'pm_scale'], 'NSGA-III': ['sbx_eta', 'sbx_pc', 'pm_eta', 'pm_scale'],
        'HNSGA-PSO': ['sbx_eta', 'sbx_pc', 'pm_eta', 'pm_scale', 'ppso', 'w0', 'c1', 'c2', 'vmax'],
        'MOPSO': ['w0', 'c1', 'c2', 'vmax', 'pm_eta', 'pm_scale'],
        'MOEA/D': ['T', 'delta', 'nr', 'Fd', 'CR', 'pm_eta', 'pm_scale']}
def sample(alg, rng):
    cfg = {}
    for k in KEYS[alg]:
        if k == 'T': cfg[k] = int(rng.choice([5, 10, 20]))
        elif k == 'nr': cfg[k] = int(rng.choice([1, 2, 3]))
        elif k == 'delta': cfg[k] = float(rng.uniform(0.5, 1.0))
        elif k == 'Fd': cfg[k] = float(rng.uniform(0.3, 0.9))
        elif k == 'CR': cfg[k] = float(rng.uniform(0.5, 1.0))
        else: lo, hi = SPACE[k]; cfg[k] = float(rng.uniform(lo, hi))
    return cfg
def run(alg, cfg, seed):
    algos.PARAMS.update(DEF); algos.PARAMS.update({k: v for k, v in cfg.items() if k in DEF})
    p = Problem(sc)
    if alg == 'NSGA-II': o = algos.nsga2(p, seed, False)
    elif alg == 'HNSGA-PSO': o = algos.nsga2(p, seed, True, ppso=cfg.get('ppso', 0.5))
    elif alg == 'MOEA/D':
        o = algos.moead(p, seed, T=cfg.get('T', 10), delta=cfg.get('delta', 0.9), nr=cfg.get('nr', 2), CR=cfg.get('CR', 1.0), Fd=cfg.get('Fd', 0.5))
    elif alg == 'NSGA-III': o = algos_extra.nsga3(p, seed)
    else: o = algos_extra.mopso(p, seed)
    X, F, cv, _ = o
    return hv3d(feas_nd(F, cv))
if __name__ == '__main__':
    for alg in sys.argv[1:]:
        t0 = time.time(); rng = np.random.default_rng(abs(hash(alg)) % 2**32 if False else {'NSGA-II': 1, 'NSGA-III': 2, 'HNSGA-PSO': 3, 'MOPSO': 4, 'MOEA/D': 5}[alg])
        cfgs = [dict()] + [sample(alg, rng) for _ in range(NCFG)]       # first = default
        out = []
        for cfg in cfgs:
            hv = np.array([run(alg, cfg, s) for s in SEEDS]); out.append((cfg, hv))
            print(alg, 'cfg', {k: round(v, 3) for k, v in cfg.items()}, 'median HV %.3f mean %.3f' % (np.median(hv), hv.mean()), flush=True)
        best = max(out, key=lambda t: (np.median(t[1]), t[1].mean()))
        pickle.dump(dict(all=out, best=best[0]), open(f'tune_{alg.replace("/", "_")}.pkl', 'wb'))
        print(alg, 'BEST', best[0], 'median %.3f' % np.median(best[1]), 'time %.0f s' % (time.time()-t0), flush=True)
