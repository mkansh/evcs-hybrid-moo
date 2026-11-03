import numpy as np
from algos import nds
import os
_N = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'norm_legacy.npz')); IDEAL, NADIR, REF = _N['ideal'], _N['nadir'], _N['ref']
R = np.array([1.1, 1.1, 1.1])
def feas_nd(F, cv):
    f = F[cv <= 1e-12]
    if len(f) == 0: return f
    f = f[nds(f, np.zeros(len(f))) == 0]
    return np.unique(np.round(f, 10), axis=0)
def hv2d(P, r):
    if len(P) == 0: return 0.
    P = P[np.argsort(P[:, 0])]; h = 0.; ymin = r[1]
    for p in P:
        if p[1] < ymin: h += (r[0]-p[0])*(ymin-p[1]); ymin = p[1]
    return h
def hv3d(F, ideal=None, nadir=None):
    ideal = IDEAL if ideal is None else ideal; nadir = NADIR if nadir is None else nadir
    if len(F) == 0: return 0.
    P = (F-ideal)/(nadir-ideal); P = P[np.all(P < R, 1)]
    if len(P) == 0: return 0.
    P = P[np.argsort(P[:, 2])]; h = 0.
    for i in range(len(P)):
        z1 = P[i, 2]; z2 = P[i+1, 2] if i+1 < len(P) else R[2]
        h += hv2d(P[:i+1, :2], R[:2])*(z2-z1)
    return h
