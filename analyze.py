import numpy as np, pickle, json
from scipy import stats
from model import *
from algos import nds
res = pickle.load(open('results.pkl','rb'))
names = ['NSGA-II','HNSGA-PSO','MOEA/D']
NR = 30
def feas_nd(F, cv):
    f = F[cv <= 1e-12]
    if len(f)==0: return f
    r = nds(f, np.zeros(len(f)))
    f = f[r==0]
    return np.unique(np.round(f,10),axis=0)
allp = np.vstack([feas_nd(r['F'], r['cv']) for n in names for r in res[n]])
ref = allp[nds(allp, np.zeros(len(allp)))==0]
ref = np.unique(ref,axis=0)
ideal = ref.min(0); nadir = ref.max(0)
norm = lambda f: (f-ideal)/(nadir-ideal)
R = np.array([1.1,1.1,1.1])
def hv2d(P, r):
    if len(P)==0: return 0.
    P = P[np.argsort(P[:,0])]; h=0.; ymin=r[1]
    for p in P:
        if p[1] < ymin:
            h += (r[0]-p[0])*(ymin-p[1]); ymin = p[1]
    return h
def hv3d(F):
    if len(F)==0: return 0.
    P = norm(F); P = P[np.all(P<R,1)]
    if len(P)==0: return 0.
    P = P[np.argsort(P[:,2])]; h=0.
    for i in range(len(P)):
        z1 = P[i,2]; z2 = P[i+1,2] if i+1<len(P) else R[2]
        h += hv2d(P[:i+1,:2], R[:2])*(z2-z1)
    return h
Pref = norm(ref)
def gd(F):
    if len(F)==0: return np.nan
    P = norm(F)
    d = np.sqrt(((P[:,None]-Pref[None])**2).sum(-1)).min(1)
    return np.sqrt((d**2).sum())/len(P)
def spacing(F):
    if len(F)<2: return 0.
    P = norm(F)
    D = np.abs(P[:,None]-P[None]).sum(-1); np.fill_diagonal(D,np.inf)
    d = D.min(1); return np.sqrt(((d-d.mean())**2).sum()/(len(P)-1))
M = {n:dict(hv=[],gd=[],sp=[],npt=[],trace=[]) for n in names}
for n in names:
    for r in res[n]:
        f = feas_nd(r['F'], r['cv'])
        M[n]['hv'].append(hv3d(f)); M[n]['gd'].append(gd(f)); M[n]['sp'].append(spacing(f)); M[n]['npt'].append(len(f))
        tr=[]
        for ff in r['tr']:
            if len(ff)==0: tr.append(0.); continue
            q = ff[nds(ff,np.zeros(len(ff)))==0]; tr.append(hv3d(q))
        M[n]['trace'].append(tr)
    for k in ('hv','gd','sp','npt'): M[n][k]=np.array(M[n][k])
pickle.dump(dict(M=M,ideal=ideal,nadir=nadir,ref=ref),open('metrics.pkl','wb'))
def holm(p):
    o=np.argsort(p); m=len(p); adj=np.zeros(m); run=0
    for i,k in enumerate(o):
        run=max(run,(m-i)*p[k]); adj[k]=min(1,run)
    return adj
def a12(x,y):  # P(x>y)+0.5P(x=y)
    return (np.sum(x[:,None]>y[None])+0.5*np.sum(x[:,None]==y[None]))/(len(x)*len(y))
out={}
for met,hib in (('hv',True),('gd',False),('sp',False)):
    d={n:M[n][met] for n in names}
    fr = stats.friedmanchisquare(*[d[n] for n in names])
    kw = stats.kruskal(*[d[n] for n in names])  # runs are independent: Kruskal-Wallis is the appropriate omnibus test
    pairs=[(0,1),(0,2),(1,2)]
    ps=[stats.ranksums(d[names[i]],d[names[j]]).pvalue for i,j in pairs]
    pa=holm(np.array(ps))
    # mean ranks (rank 1 best)
    X=np.array([d[n] for n in names]).T
    rk=np.array([stats.rankdata(-r if hib else r) for r in X]).mean(0)
    out[met]=dict(kruskal_H=float(kw.statistic),kruskal_p=float(kw.pvalue),friedman_chi2=fr.statistic,friedman_p=fr.pvalue,ranks=rk.tolist(),
        pairs=[(names[i],names[j],float(p),float(q),float(a12(d[names[i]],d[names[j]]))) for (i,j),p,q in zip(pairs,ps,pa)],
        mean={n:float(d[n].mean()) for n in names},sd={n:float(d[n].std(ddof=1)) for n in names},
        median={n:float(np.median(d[n])) for n in names})
print(json.dumps(out,indent=1))
hvall=np.concatenate([M[n]['hv'] for n in names])
print('HV sorted per alg'); 
for n in names: print(n, np.round(np.sort(M[n]['hv']),3))
print('npt',{n:float(M[n]['npt'].mean()) for n in names})
json.dump(out,open('stats.json','w'),indent=1)
