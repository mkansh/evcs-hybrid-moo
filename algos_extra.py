"""Additional reference solvers for the EVCS benchmark: NSGA-III and MOPSO.

Both use the same budget as NSGA-II (POP + (GENS-1)*POP = 3600 evaluations), the same constrained-domination
rule (algos.dominates / algos.nds), the same initialisation and, for NSGA-III, the same SBX/polynomial-mutation
(real encoding) or integer bus-gene operators (discrete encoding, EVCS_ENC=discrete) as NSGA-II.
Parameters were fixed from the literature defaults before running (no tuning on any seed).
"""
import numpy as np
import algos
from algos import POP, GENS, init_pop, cross, mutate, dominates, nds, crowding, simplex_weights, _gene


# ----------------------------------------------------------------------------- NSGA-III
def _normalize(F):
    """Deb & Jain (2014) adaptive normalisation: ideal point, ASF extreme points, hyperplane intercepts."""
    m = F.shape[1]
    ideal = F.min(0)
    Ft = F - ideal
    E = np.zeros((m, m))
    for j in range(m):
        w = np.full(m, 1e-6); w[j] = 1.0
        E[j] = Ft[np.argmin(np.max(Ft / w, 1))]
    try:
        b = np.linalg.solve(E, np.ones(m))
        a = 1.0 / b
        if np.any(~np.isfinite(a)) or np.any(a <= 1e-9):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        a = Ft.max(0)
    a = np.maximum(a, 1e-12)
    return Ft / a


def _associate(Fn, W):
    """Perpendicular distance of every point to every reference line; return nearest line and distance."""
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    proj = Fn @ Wn.T                                    # (n, nref)
    d2 = (Fn**2).sum(1)[:, None] - proj**2
    d = np.sqrt(np.maximum(d2, 0))
    j = np.argmin(d, 1)
    return j, d[np.arange(len(Fn)), j]


def nsga3_select(X, F, cv, k, W, rng):
    rank = nds(F, cv)
    sel = []
    last = None
    for r in np.unique(rank):
        idx = np.where(rank == r)[0]
        if len(sel) + len(idx) <= k:
            sel.extend(idx.tolist())
            if len(sel) == k:
                break
        else:
            last = idx
            break
    if len(sel) < k:
        S = np.array(sel + last.tolist())
        Fn = _normalize(F[S])
        j, dist = _associate(Fn, W)
        nsel = len(sel)
        rho = np.zeros(len(W), int)
        for t in range(nsel):
            rho[j[t]] += 1
        cand = {}                                       # ref line -> list of positions (in S) from the last front
        for t in range(nsel, len(S)):
            cand.setdefault(j[t], []).append(t)
        chosen = []
        while len(sel) + len(chosen) < k and cand:
            lines = np.array(list(cand.keys()))
            rmin = rho[lines].min()
            jl = rng.choice(lines[rho[lines] == rmin])
            members = cand[jl]
            if rho[jl] == 0:
                t = members[int(np.argmin(dist[members]))]
            else:
                t = members[int(rng.integers(len(members)))]
            chosen.append(t)
            members.remove(t)
            rho[jl] += 1
            if not members:
                del cand[jl]
        sel = sel + [int(S[t]) for t in chosen]
    sel = np.array(sel)
    return X[sel], F[sel], cv[sel]


def nsga3(prob, seed, hvfn=None):
    rng = np.random.default_rng(seed)
    m_ = getattr(prob, 'nobj', 3); W = simplex_weights({3: 9, 4: 5}[m_], m_)   # Das-Dennis points: 55 (M = 3, H = 9), 56 (M = 4, H = 5)
    X = init_pop(rng, POP, prob.n)
    F, cv = prob.evaluate(X)
    tr = [F[cv <= 1e-12].copy()] if hvfn else []
    for g in range(GENS - 1):
        off = np.zeros_like(X)
        par = rng.integers(0, POP, POP + 2)              # random mating (Deb & Jain)
        i = 0
        while i < POP:
            c1, c2 = cross(X[par[i]], X[par[i + 1]], rng)
            off[i] = mutate(c1, rng)
            if i + 1 < POP:
                off[i + 1] = mutate(c2, rng)
            i += 2
        Fo, cvo = prob.evaluate(off)
        X, F, cv = nsga3_select(np.vstack([X, off]), np.vstack([F, Fo]), np.concatenate([cv, cvo]), POP, W, rng)
        if hvfn: tr.append(F[cv <= 1e-12].copy())
    return X, F, cv, tr


# ----------------------------------------------------------------------------- MOPSO
def _archive_update(AX, AF, Ac, NX, NF, Nc, amax):
    X = np.vstack([AX, NX]); F = np.vstack([AF, NF]); c = np.concatenate([Ac, Nc])
    keep = nds(F, c) == 0
    X, F, c = X[keep], F[keep], c[keep]
    _, u = np.unique(np.round(F, 10), axis=0, return_index=True)       # drop duplicate objective vectors
    u = np.sort(u); X, F, c = X[u], F[u], c[u]
    while len(F) > amax:                                               # crowding-distance truncation
        cd = crowding(F, np.zeros(len(F), int))
        fin = np.where(np.isfinite(cd))[0]
        drop = fin[np.argmin(cd[fin])] if len(fin) else int(np.argmin(cd))
        m = np.ones(len(F), bool); m[drop] = False
        X, F, c = X[m], F[m], c[m]
    return X, F, c


def mopso(prob, seed, hvfn=None, w0=None, c1=None, c2=None, vmax=None):
    """Archive-based MOPSO with crowding-distance leader selection (binary tournament) and archive truncation,
    the same inertia schedule and clamp as the swarm move of HNSGA-PSO, and one polynomial-mutation
    (turbulence) pass on every new particle. Discrete encoding: bus genes copy the personal-best gene (p = 0.3)
    or the leader gene (p = 0.3), as in the hybrid."""
    w0 = algos.PARAMS["w0"] if w0 is None else w0; c1 = algos.PARAMS["c1"] if c1 is None else c1
    c2 = algos.PARAMS["c2"] if c2 is None else c2; vmax = algos.PARAMS["vmax"] if vmax is None else vmax
    rng = np.random.default_rng(seed)
    X = init_pop(rng, POP, prob.n)
    F, cv = prob.evaluate(X)
    V = np.zeros_like(X)
    pb, pbF, pbcv = X.copy(), F.copy(), cv.copy()
    AX, AF, Ac = _archive_update(X[:0], F[:0], cv[:0], X, F, cv, POP)
    tr = [AF[Ac <= 1e-12].copy()] if hvfn else []
    for g in range(GENS - 1):
        w = w0 * (1 - 0.5 * g / (GENS - 2))
        cd = crowding(AF, np.zeros(len(AF), int))
        a = rng.integers(0, len(AF), POP); b = rng.integers(0, len(AF), POP)
        lead = np.where(cd[a] >= cd[b], a, b)
        off = np.zeros_like(X)
        for i in range(POP):
            L = AX[lead[i]]
            v = w * V[i] + c1 * rng.random(prob.n) * (pb[i] - X[i]) + c2 * rng.random(prob.n) * (L - X[i])
            v = np.clip(v, -vmax, vmax); V[i] = v
            xn = np.clip(X[i] + v, 0, 1)
            if algos.ENC == 'discrete':
                r = rng.random(algos.NBG)
                nb = algos.NBG
                xn[:nb] = np.where(r < 0.3, pb[i, :nb], np.where(r < 0.6, L[:nb], X[i, :nb]))
                V[i, :nb] = 0.0
            off[i] = mutate(xn, rng)
        Fo, cvo = prob.evaluate(off)
        for i in range(POP):
            if dominates(Fo[i], cvo[i], pbF[i], pbcv[i]) or (not dominates(pbF[i], pbcv[i], Fo[i], cvo[i]) and rng.random() < .5):
                pb[i] = off[i]; pbF[i] = Fo[i]; pbcv[i] = cvo[i]
        X, F, cv = off, Fo, cvo                           # swarm positions are replaced (no elitism in the swarm)
        AX, AF, Ac = _archive_update(AX, AF, Ac, X, F, cv, POP)
        if hvfn: tr.append(AF[Ac <= 1e-12].copy())
    return AX, AF, Ac, tr
