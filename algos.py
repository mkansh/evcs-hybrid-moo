import numpy as np

POP, GENS = 60, 60            # 3600 evaluations per run


def dominates(Fa, cva, Fb, cvb):
    """Deb's constrained domination a over b (arrays broadcast)."""
    fa = cva <= 1e-12; fb = cvb <= 1e-12
    better = np.where(fa & fb, np.all(Fa <= Fb, -1) & np.any(Fa < Fb, -1),
                      np.where(fa & ~fb, True, np.where(~fa & fb, False, cva < cvb)))
    return better


def nds(F, cv):
    n = len(F)
    D = dominates(F[:, None, :], cv[:, None], F[None, :, :], cv[None, :])   # D[i,j]: i dom j
    cnt = D.sum(0)
    rank = np.full(n, -1); r = 0; rem = np.ones(n, bool)
    while rem.any():
        front = rem & (cnt == 0)
        if not front.any():
            front = rem.copy()
        rank[front] = r
        for i in np.where(front)[0]:
            cnt = cnt - D[i]
        rem &= ~front
        cnt = np.where(rem, cnt, 1)
        r += 1
    return rank


def crowding(F, rank):
    n = len(F); cd = np.zeros(n)
    for r in np.unique(rank):
        idx = np.where(rank == r)[0]
        if len(idx) <= 2:
            cd[idx] = np.inf; continue
        for m in range(F.shape[1]):
            o = idx[np.argsort(F[idx, m])]
            rng_ = F[o[-1], m] - F[o[0], m]
            cd[o[0]] = cd[o[-1]] = np.inf
            if rng_ > 0:
                cd[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / rng_
    return cd


def sbx(p1, p2, rng, eta=15.0, pc=0.9):
    n = len(p1); c1 = p1.copy(); c2 = p2.copy()
    if rng.random() < pc:
        u = rng.random(n);
        beta = np.where(u <= .5, (2*u)**(1/(eta+1)), (1/(2*(1-u)))**(1/(eta+1)))
        sw = rng.random(n) < .5
        a = .5*((1+beta)*p1 + (1-beta)*p2); b = .5*((1-beta)*p1 + (1+beta)*p2)
        c1 = np.where(sw, a, p1); c2 = np.where(sw, b, p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def pmut(x, rng, eta=20.0):
    n = len(x); pm = 1.0/n
    m = rng.random(n) < pm
    u = rng.random(n)
    d = np.where(u < .5, (2*u)**(1/(eta+1)) - 1, 1 - (2*(1-u))**(1/(eta+1)))
    return np.clip(np.where(m, x + d, x), 0, 1)


def tour(rank, cd, rng, n):
    a = rng.integers(0, len(rank), n); b = rng.integers(0, len(rank), n)
    win = np.where((rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b])), a, b)
    return win


def env_select(X, F, cv, k):
    rank = nds(F, cv); cd = crowding(F, rank)
    o = np.lexsort((-cd, rank))[:k]
    return X[o], F[o], cv[o]


def hv_trace(F, cv, fn):
    return F[cv <= 1e-12].copy()


def nsga2(prob, seed, hybrid=False, hvfn=None, ppso=0.5):
    rng = np.random.default_rng(seed)
    X = rng.random((POP, prob.n)); F, cv = prob.evaluate(X)
    V = np.zeros_like(X); pb = X.copy(); pbF = F.copy(); pbcv = cv.copy()
    tr = []
    w0 = 0.5
    for g in range(GENS - 1):
        rank = nds(F, cv); cd = crowding(F, rank)
        if hvfn: tr.append(hv_trace(F, cv, hvfn))
        # rng stream for the PSO move is separate so both algorithms share GA draws
        rng2 = np.random.default_rng(seed*7919 + g)
        off = np.zeros_like(X)
        i = 0
        par = tour(rank, cd, rng, POP + 2)
        while i < POP:
            c1, c2 = sbx(X[par[i]], X[par[i+1]], rng)
            off[i] = pmut(c1, rng)
            if i + 1 < POP: off[i+1] = pmut(c2, rng)
            i += 2
        if hybrid:
            w = w0*(1 - 0.5*g/(GENS-2))
            front = np.where(rank == 0)[0]
            use = rng2.random(POP) < ppso
            for i in np.where(use)[0]:
                L = X[rng2.choice(front)]
                v = w*V[i] + 1.5*rng2.random(prob.n)*(pb[i]-X[i]) + 1.5*rng2.random(prob.n)*(L-X[i])
                v = np.clip(v, -.25, .25); V[i] = v
                off[i] = np.clip(X[i] + v, 0, 1)
        Fo, cvo = prob.evaluate(off)
        if hybrid:   # update personal bests
            for i in range(POP):
                if dominates(Fo[i], cvo[i], pbF[i], pbcv[i]) or (not dominates(pbF[i], pbcv[i], Fo[i], cvo[i]) and rng2.random() < .5):
                    pb[i] = off[i]; pbF[i] = Fo[i]; pbcv[i] = cvo[i]
        XA = np.vstack([X, off]); FA = np.vstack([F, Fo]); cA = np.concatenate([cv, cvo])
        if hybrid:
            Vn = np.vstack([V, np.zeros_like(V)])
            XA2, FA2, cA2 = XA, FA, cA
        X, F, cv = env_select(XA, FA, cA, POP)
        if hybrid:   # keep velocities/pbest aligned by re-matching survivors (nearest copies)
            pbn = np.zeros_like(X); pbFn = np.zeros_like(F); pbcn = np.zeros_like(cv); Vnn = np.zeros_like(X)
            allpb = np.vstack([pb, pb]); allpF = np.vstack([pbF, pbF]); allpc = np.concatenate([pbcv, pbcv]); allV = Vn
            for j in range(POP):
                k = np.argmin(np.abs(XA - X[j]).sum(1))
                pbn[j] = allpb[k]; pbFn[j] = allpF[k]; pbcn[j] = allpc[k]; Vnn[j] = allV[k]
            pb, pbF, pbcv, V = pbn, pbFn, pbcn, Vnn
    if hvfn: tr.append(hv_trace(F, cv, hvfn))
    return X, F, cv, tr


def simplex_weights(H=9, m=3):
    import itertools
    W = [np.array(c)/H for c in itertools.product(range(H+1), repeat=m) if sum(c) == H]
    return np.array(W)


def moead(prob, seed, hvfn=None, T=10, delta=0.9, nr=2, CR=1.0, Fd=0.5):
    rng = np.random.default_rng(seed)
    W = np.maximum(simplex_weights(9), 1e-6); N = len(W)
    gens = int(round(POP*GENS/N)) - 1
    B = np.argsort(((W[:, None] - W[None])**2).sum(-1), 1)[:, :T]
    X = rng.random((N, prob.n)); F, cv = prob.evaluate(X)
    zmin = F.min(0); zmax = F.max(0)
    tr = []
    def g(f, w):
        return np.max(w*np.abs(f - zmin)/np.maximum(zmax - zmin, 1e-12), -1)
    for it in range(gens):
        if hvfn: tr.append(hv_trace(F, cv, hvfn))
        for i in rng.permutation(N):
            P = B[i] if rng.random() < delta else np.arange(N)
            a, b = rng.choice(P, 2, replace=False)
            y = X[i] + Fd*(X[a] - X[b])
            cr = rng.random(prob.n) < CR
            y = np.where(cr, y, X[i]); y = np.clip(y, 0, 1)
            y = pmut(y, rng)
            Fy, cy = prob.evaluate(y[None]); Fy = Fy[0]; cy = cy[0]
            zmin = np.minimum(zmin, Fy); zmax = np.maximum(zmax, Fy)
            c = 0
            for j in rng.permutation(P):
                better = (cy < cv[j]) if (cy > 1e-12 or cv[j] > 1e-12) else (g(Fy, W[j]) <= g(F[j], W[j]))
                if better:
                    X[j] = y; F[j] = Fy; cv[j] = cy; c += 1
                    if c >= nr: break
    if hvfn: tr.append(hv_trace(F, cv, hvfn))
    return X, F, cv, tr
