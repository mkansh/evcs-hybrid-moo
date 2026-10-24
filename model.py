"""IEEE 33-bus EVCS + PV + WT benchmark: network, scenarios, objectives."""
import numpy as np

# ---------------- IEEE 33-bus data (Baran & Wu) ----------------
BR = [(1,2,.0922,.0470),(2,3,.4930,.2511),(3,4,.3660,.1864),(4,5,.3811,.1941),(5,6,.8190,.7070),
(6,7,.1872,.6188),(7,8,.7114,.2351),(8,9,1.0300,.7400),(9,10,1.0440,.7400),(10,11,.1966,.0650),
(11,12,.3744,.1238),(12,13,1.4680,1.1550),(13,14,.5416,.7129),(14,15,.5910,.5260),(15,16,.7463,.5450),
(16,17,1.2890,1.7210),(17,18,.7320,.5740),(2,19,.1640,.1565),(19,20,1.5042,1.3554),(20,21,.4095,.4784),
(21,22,.7089,.9373),(3,23,.4512,.3083),(23,24,.8980,.7091),(24,25,.8960,.7011),(6,26,.2030,.1034),
(26,27,.2842,.1447),(27,28,1.0590,.9337),(28,29,.8042,.7006),(29,30,.5075,.2585),(30,31,.9744,.9630),
(31,32,.3105,.3619),(32,33,.3410,.5302)]
LD = {2:(100,60),3:(90,40),4:(120,80),5:(60,30),6:(60,20),7:(200,100),8:(200,100),9:(60,20),10:(60,20),
11:(45,30),12:(60,35),13:(60,35),14:(120,80),15:(60,10),16:(60,20),17:(60,20),18:(90,40),19:(90,40),
20:(90,40),21:(90,40),22:(90,40),23:(90,50),24:(420,200),25:(420,200),26:(60,25),27:(60,25),28:(60,20),
29:(120,70),30:(200,600),31:(150,70),32:(210,100),33:(60,40)}
NB = 33
SBASE = 100000.0          # kVA
VBASE = 12.66             # kV
ZBASE = VBASE**2 / (SBASE/1000.0)
parent = np.full(NB, -1)
Z = np.zeros(NB, complex)
for f, t, r, x in BR:
    parent[t-1] = f-1
    Z[t-1] = (r + 1j*x) / ZBASE
P0 = np.zeros(NB); Q0 = np.zeros(NB)
for b, (p, q) in LD.items():
    P0[b-1] = p; Q0[b-1] = q
ORDER = list(range(1, NB))        # branch/child ids are topologically ordered (parent < child)


def bfs(Pl, Ql, tol=1e-7, maxit=60):
    """Backward/forward sweep. Pl,Ql: (NB,C) net bus demand in kW/kvar. Returns |V| (NB,C), loss kW (C,), Pslack kW."""
    C = Pl.shape[1]
    S = (Pl + 1j*Ql) / SBASE
    V = np.ones((NB, C), complex)
    for _ in range(maxit):
        I = np.conj(S / V)
        Ib = I.copy()
        for i in reversed(ORDER):
            Ib[parent[i]] += Ib[i]
        Vn = V.copy()
        Vn[0] = 1.0
        for i in ORDER:
            Vn[i] = Vn[parent[i]] - Z[i]*Ib[i]
        err = np.max(np.abs(np.abs(Vn) - np.abs(V)))
        V = Vn
        if err < tol:
            break
    I = np.conj(S / V)
    Ib = I.copy()
    for i in reversed(ORDER):
        Ib[parent[i]] += Ib[i]
    loss = np.zeros(C)
    for i in ORDER:
        loss += (Z[i].real * np.abs(Ib[i])**2) * SBASE
    pslack = Pl.sum(0) + loss
    return np.abs(V), loss, pslack


# ---------------- scenarios ----------------
T = 24
LOADPROF = np.array([.60,.55,.52,.50,.52,.58,.70,.82,.90,.93,.95,.95,.93,.92,.92,.93,.95,.98,1.0,1.0,.97,.90,.80,.68])
EVPROF = np.array([.05,.03,.02,.02,.03,.06,.12,.20,.28,.32,.35,.38,.40,.40,.42,.48,.60,.78,.92,1.0,.95,.70,.35,.15])


def make_scenarios(seed=2024, ndays=2000, ns=4, tmax=47.0, swing=12.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) + 0.5
    gclear = 1000*np.clip(np.sin(np.pi*(t-6)/12), 0, None)**1.2
    # clearness index: Beta(mean .8, sd .15) per day with hourly noise
    mu, sd = 0.80, 0.15
    k = mu*(1-mu)/sd**2 - 1
    kt = rng.beta(mu*k, (1-mu)*k, size=(ndays, 1))
    hn = np.clip(1 + 0.08*rng.standard_normal((ndays, T)), 0.5, 1.2)
    G = np.clip(gclear*kt*hn, 0, 1100)
    # wind: Weibull(k=2, c~N(6.5,1.5)) with AR(1) hourly dependence (Gaussian copula)
    c = np.clip(rng.normal(6.5, 1.5, size=(ndays, 1)), 3, 10)
    z = np.zeros((ndays, T)); z[:, 0] = rng.standard_normal(ndays)
    rho = 0.9
    for h in range(1, T):
        z[:, h] = rho*z[:, h-1] + np.sqrt(1-rho**2)*rng.standard_normal(ndays)
    from scipy.stats import norm
    u = np.clip(norm.cdf(z), 1e-9, 1-1e-9)
    kw = 2.0
    v = c*(-np.log(1-u))**(1/kw)
    ev = np.clip(1 + 0.2*rng.standard_normal((ndays, 1)), 0.5, 1.5)
    from sklearn.cluster import KMeans
    feat = np.hstack([G/1000.0, v/10.0])
    km = KMeans(ns, n_init=10, random_state=seed).fit(feat)
    lab = km.labels_
    Gs = np.array([G[lab == s].mean(0) for s in range(ns)])
    vs = np.array([v[lab == s].mean(0) for s in range(ns)])
    evs = np.array([ev[lab == s].mean() for s in range(ns)])
    pi = np.array([(lab == s).mean() for s in range(ns)])
    o = np.argsort(-Gs.sum(1))
    Ta = tmax - swing*(1 - np.cos(2*np.pi*(t-15)/24))/2
    return dict(G=Gs[o], v=vs[o], ev=evs[o], pi=pi[o], Ta=Ta, ndays=ndays)


# ---------------- DG models ----------------
BETA_T, NOCT, SOIL = 0.0045, 45.0, 0.07
VCI, VR, VCO = 3.0, 12.0, 25.0


def pv_unit(G, Ta):
    Tc = Ta + (NOCT-20)/800*G
    return G/1000.0*(1 - BETA_T*(Tc-25))*(1 - SOIL)


def wt_unit(v):
    p = np.where(v < VCI, 0, np.where(v < VR, (v-VCI)/(VR-VCI), np.where(v <= VCO, 1.0, 0.0)))
    return p


class Problem:
    """x in [0,1]^6: hub1, hub2, pv bus, wt bus, pv size, wt size."""
    n = 6
    PVMAX, WTMAX = 1500.0, 1000.0
    NPORT, PORTKW, PORTCOST = 6, 50.0, 30000.0
    CPV, CWT, OM, R, N, IMP = 750.0, 1300.0, 0.02, 0.08, 20, 0.08
    CRF = R*(1+R)**N/((1+R)**N - 1)
    VMIN, VMAX = 0.90, 1.05

    def __init__(self, sc=None, evcs=True):
        self.sc = sc or make_scenarios()
        self.evcs = evcs
        s = self.sc
        ns = len(s['pi'])
        self.ns = ns
        self.pvu = np.array([pv_unit(s['G'][i], s['Ta']) for i in range(ns)])   # (ns,T)
        self.wtu = np.array([wt_unit(s['v'][i]) for i in range(ns)])
        self.w = (s['pi'][:, None]*np.ones((1, T))).ravel()                       # case weights
        self.evm = np.repeat(s['ev'], T)
        self.lp = np.tile(LOADPROF, ns)
        self.evp = np.tile(EVPROF, ns)
        self.pvu_c = self.pvu.ravel(); self.wtu_c = self.wtu.ravel()
        self.C = ns*T
        self.nfe = 0

    @staticmethod
    def bus(x):
        return np.clip(np.round(1 + x*31).astype(int), 1, 32)   # index 1..32 -> buses 2..33

    def decode(self, x):
        return dict(h1=self.bus(x[0]), h2=self.bus(x[1]), pb=self.bus(x[2]), wb=self.bus(x[3]),
                    pv=x[4]*self.PVMAX, wt=x[5]*self.WTMAX)

    def plan_eval(self, d, hubs=True):
        C = self.C
        Pl = np.repeat(P0[:, None]*self.lp[None, :], 1, axis=1).copy()
        Ql = Q0[:, None]*self.lp[None, :]
        Pl = Pl.copy()
        if hubs:
            hubkw = self.NPORT*self.PORTKW*self.evp*self.evm
            for h in (d['h1'], d['h2']):
                Pl[h] += hubkw; Ql[h] += hubkw*0.3287   # pf 0.95
        Pl[d['pb']] -= d['pv']*self.pvu_c
        Pl[d['wb']] -= d['wt']*self.wtu_c
        return Pl, Ql

    def evaluate(self, X):
        """X (N,6) -> F (N,3), CV (N,). F = [annual cost USD, annual loss MWh, VDI]."""
        N = X.shape[0]
        Pls, Qls = [], []
        for x in X:
            d = self.decode(x)
            Pl, Ql = self.plan_eval(d, self.evcs)
            Pls.append(Pl); Qls.append(Ql)
        Pl = np.concatenate(Pls, 1); Ql = np.concatenate(Qls, 1)
        V, loss, ps = bfs(Pl, Ql)
        self.nfe += N
        C = self.C
        F = np.zeros((N, 3)); CV = np.zeros(N)
        for k in range(N):
            sl = slice(k*C, (k+1)*C)
            Vk = V[:, sl]; lk = loss[sl]; pk = ps[sl]
            d = self.decode(X[k])
            loss_mwh = 365*np.sum(self.w*lk)/1000.0
            imp = 365*np.sum(self.w*np.maximum(pk, 0))*self.IMP
            capex = (2*self.NPORT*self.PORTCOST if self.evcs else 0) + d['pv']*self.CPV + d['wt']*self.CWT
            cost = capex*(self.CRF + self.OM) + imp
            vdi = np.sum(self.w*np.sum((1 - Vk[1:])**2, 0))/np.sum(self.w)
            F[k] = [cost, loss_mwh, vdi]
            viol = np.maximum(0, self.VMIN - Vk[1:]).max(0) + np.maximum(0, Vk[1:] - self.VMAX).max(0)
            CV[k] = np.sum(self.w*viol)/np.sum(self.w)*100 + (viol.max() if viol.max() > 0 else 0)
        return F, CV

    def details(self, x, hubs=True):
        d = self.decode(x)
        Pl, Ql = self.plan_eval(d, hubs)
        V, loss, ps = bfs(Pl, Ql)
        w = self.w
        res = dict(d=d,
                   loss=365*np.sum(w*loss)/1000, vmin=V[1:].min(), nviol=int((V[1:, :].min(0) < 0.9).sum()),
                   Vworst=V[:, np.argmin(V[1:].min(0))],
                   ren=np.sum(w*(d['pv']*self.pvu_c + d['wt']*self.wtu_c))/np.sum(w*(Pl.sum(0)+d['pv']*self.pvu_c + d['wt']*self.wtu_c)),
                   capex=(2*self.NPORT*self.PORTCOST if hubs else 0)+d['pv']*self.CPV+d['wt']*self.CWT)
        return res
