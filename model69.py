"""IEEE 69-bus extension: 2 EV hubs + PV + WT + BESS (rule-based TOU dispatch), four objectives
[annual cost, annual loss, VDI, annual grid emissions]. Network data: Baran & Wu (1989) as distributed with MATPOWER case69.
Cost, tariff, BESS and emission-factor values are illustrative planning assumptions (see paper)."""
import numpy as np
from model import make_scenarios, LOADPROF, EVPROF, pv_unit, wt_unit, T

import os
_D = os.path.dirname(os.path.abspath(__file__))
_bus = np.load(os.path.join(_D, 'bus69.npy')); _br = np.load(os.path.join(_D, 'br69.npy'))   # Baran & Wu 69-bus data from MATPOWER case69
NB = 69; SBASE = 100000.0; VBASE = 12.66; ZBASE = VBASE**2/(SBASE/1000.0)
parent = np.full(NB, -1); Z = np.zeros(NB, complex)
for f, t, r, x in _br[:, :4]:
    parent[int(t)-1] = int(f)-1; Z[int(t)-1] = (r+1j*x)/ZBASE
P0 = np.zeros(NB); Q0 = np.zeros(NB)
for row in _bus: P0[int(row[0])-1] = row[2]; Q0[int(row[0])-1] = row[3]
ORDER = list(range(1, NB))
assert all(parent[i] < i for i in ORDER)

def bfs(Pl, Ql, tol=1e-7, maxit=60):
    C = Pl.shape[1]; S = (Pl+1j*Ql)/SBASE; V = np.ones((NB, C), complex)
    for _ in range(maxit):
        I = np.conj(S/V); Ib = I.copy()
        for i in reversed(ORDER): Ib[parent[i]] += Ib[i]
        Vn = V.copy(); Vn[0] = 1.0
        for i in ORDER: Vn[i] = Vn[parent[i]] - Z[i]*Ib[i]
        err = np.max(np.abs(np.abs(Vn)-np.abs(V))); V = Vn
        if err < tol: break
    I = np.conj(S/V); Ib = I.copy()
    for i in reversed(ORDER): Ib[parent[i]] += Ib[i]
    loss = np.zeros(C)
    for i in ORDER: loss += (Z[i].real*np.abs(Ib[i])**2)*SBASE
    return np.abs(V), loss, Pl.sum(0)+loss

TOU = np.full(T, 0.08); TOU[0:6] = 0.05; TOU[17:23] = 0.12          # USD/kWh: off-peak 0-5, peak 17-22
CHG = np.zeros(T, bool); CHG[0:6] = True; CHG[11:16] = True          # BESS charging windows
DIS = np.zeros(T, bool); DIS[17:23] = True                           # BESS discharging window
EFF, SOCMIN, SOCMAX = 0.95, 0.10, 0.90

def bess_profile(E, Pm):
    """Net injection (kW, + discharge) of a rule-based BESS; two simulated days, the second one is returned."""
    if E <= 1e-6 or Pm <= 1e-6: return np.zeros(T), 0.0
    Pm = min(Pm, E); e = 0.5*E; prof = np.zeros(T); thr = 0.0
    for day in range(2):
        for h in range(T):
            if CHG[h]:
                pc = max(0.0, min(Pm, (SOCMAX*E-e)/EFF)); e += EFF*pc; v = -pc; t_ = pc
            elif DIS[h]:
                pd = max(0.0, min(Pm, (e-SOCMIN*E)*EFF)); e -= pd/EFF; v = pd; t_ = pd
            else: v = 0.0; t_ = 0.0
            if day == 1: prof[h] = v; thr += t_
    return prof, thr

class Problem69:
    n = 9; nobj = 4
    PVMAX, WTMAX, BEMAX, BPMAX = 1500.0, 1000.0, 2000.0, 1000.0
    NPORT, PORTKW, PORTCOST = 6, 50.0, 30000.0
    CPV, CWT, OM, R, N = 750.0, 1300.0, 0.02, 0.08, 20
    CBE, CBP, NB_LIFE = 250.0, 150.0, 10                 # USD/kWh, USD/kW, years (illustrative)
    EF = 0.60                                            # kgCO2/kWh grid emission factor (illustrative)
    VMIN, VMAX = 0.90, 1.05
    CRF = R*(1+R)**N/((1+R)**N-1); CRFB = R*(1+R)**NB_LIFE/((1+R)**NB_LIFE-1)
    def __init__(self, sc=None):
        self.sc = sc or make_scenarios(); s = self.sc; ns = len(s['pi']); self.ns = ns
        self.pvu_c = np.array([pv_unit(s['G'][i], s['Ta']) for i in range(ns)]).ravel()
        self.wtu_c = np.array([wt_unit(s['v'][i]) for i in range(ns)]).ravel()
        self.w = (s['pi'][:, None]*np.ones((1, T))).ravel(); self.evm = np.repeat(s['ev'], T)
        self.lp = np.tile(LOADPROF, ns); self.evp = np.tile(EVPROF, ns); self.tou = np.tile(TOU, ns); self.C = ns*T; self.nfe = 0
    @staticmethod
    def bus(x): return np.clip(np.round(1+x*(NB-2)).astype(int), 1, NB-1)       # 68 candidate nodes (buses 2..69)
    def decode(self, x):
        return dict(h1=self.bus(x[0]), h2=self.bus(x[1]), pb=self.bus(x[2]), wb=self.bus(x[3]), bb=self.bus(x[4]),
                    pv=x[5]*self.PVMAX, wt=x[6]*self.WTMAX, be=x[7]*self.BEMAX, bp=x[8]*self.BPMAX)
    def plan(self, d, evcs=True):
        Pl = P0[:, None]*self.lp[None, :]; Ql = Q0[:, None]*self.lp[None, :]; Pl = Pl.copy()
        if evcs:
            hk = self.NPORT*self.PORTKW*self.evp*self.evm
            for h in (d['h1'], d['h2']): Pl[h] += hk; Ql[h] += hk*0.3287
        Pl[d['pb']] -= d['pv']*self.pvu_c; Pl[d['wb']] -= d['wt']*self.wtu_c
        prof, thr = bess_profile(d['be'], d['bp'])
        Pl[d['bb']] -= np.tile(prof, self.ns)
        return Pl, Ql, thr
    def evaluate(self, X):
        N = X.shape[0]; Pls, Qls, thrs = [], [], []
        for x in X:
            Pl, Ql, thr = self.plan(self.decode(x)); Pls.append(Pl); Qls.append(Ql); thrs.append(thr)
        V, loss, ps = bfs(np.concatenate(Pls, 1), np.concatenate(Qls, 1)); self.nfe += N
        C = self.C; F = np.zeros((N, 4)); CV = np.zeros(N)
        for k in range(N):
            sl = slice(k*C, (k+1)*C); Vk = V[:, sl]; lk = loss[sl]; pk = np.maximum(ps[sl], 0); d = self.decode(X[k])
            loss_mwh = 365*np.sum(self.w*lk)/1000.0
            imp = 365*np.sum(self.w*pk*self.tou); em = 365*np.sum(self.w*pk)*self.EF/1000.0
            capex = 2*self.NPORT*self.PORTCOST + d['pv']*self.CPV + d['wt']*self.CWT
            bcap = d['be']*self.CBE + min(d['bp'], d['be'])*self.CBP
            cost = capex*(self.CRF+self.OM) + bcap*(self.CRFB+self.OM) + imp
            vdi = np.sum(self.w*np.sum((1-Vk[1:])**2, 0))/np.sum(self.w)
            F[k] = [cost, loss_mwh, vdi, em]
            viol = np.maximum(0, self.VMIN-Vk[1:]).max(0) + np.maximum(0, Vk[1:]-self.VMAX).max(0)
            CV[k] = np.sum(self.w*viol)/np.sum(self.w)*100 + (viol.max() if viol.max() > 0 else 0)
        return F, CV
