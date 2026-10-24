import numpy as np, pickle, json
import model
from model import *
from algos import *
met = pickle.load(open('metrics.pkl','rb'))
def knee(F):
    f=(F-F.min(0))/np.maximum(F.max(0)-F.min(0),1e-12)
    return np.argmin(np.sqrt((f**2).sum(1)))
def feas_front(F,cv):
    m=cv<=1e-12
    if not m.any(): return None,None
    return None,m
cases=[('Base case',{}),('Energy price 0.04 USD/kWh',{'IMP':0.04}),('Energy price 0.12 USD/kWh',{'IMP':0.12}),
 ('Soiling 0%',{'soil':0.0}),('Soiling 15%',{'soil':0.15}),('Peak ambient 35 °C',{'tmax':35.}),('Peak ambient 50 °C',{'tmax':50.}),
 ('V_min = 0.95 pu',{'VMIN':0.95})]
rows=[]
for name,kw in cases:
    model.SOIL=kw.get('soil',0.07)
    sc=make_scenarios(tmax=kw.get('tmax',47.))
    best=None
    for seed in (101,102,103):
        p=Problem(sc)
        if 'IMP' in kw: p.IMP=kw['IMP']
        if 'VMIN' in kw: p.VMIN=kw['VMIN']
        X,F,cv,_=nsga2(p,seed,True,True)
        m=cv<=1e-12
        if not m.any(): continue
        Xf,Ff=X[m],F[m]
        r=nds(Ff,np.zeros(len(Ff)))==0
        Xf,Ff=Xf[r],Ff[r]
        if best is None or Ff[:,1].min()<best[1][:,1].min(): best=(Xf,Ff,p)
    if best is None:
        rows.append((name,None)); print(name,'infeasible'); continue
    Xf,Ff,p=best; k=knee(Ff); d=p.decode(Xf[k]); det=p.details(Xf[k])
    rows.append((name,dict(h=sorted([int(d['h1'])+1,int(d['h2'])+1],reverse=False),h1=int(d['h1'])+1,h2=int(d['h2'])+1,pb=int(d['pb'])+1,pv=d['pv'],wb=int(d['wb'])+1,wt=d['wt'],cost=Ff[k,0]/1e6,loss=Ff[k,1],vdi=Ff[k,2],vmin=det['vmin'],ren=det['ren'])))
    print(rows[-1])
model.SOIL=0.07
pickle.dump(rows,open('sens.pkl','wb'))
