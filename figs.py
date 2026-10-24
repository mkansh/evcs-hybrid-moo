import numpy as np, pickle, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model import *
from algos import nds
res=pickle.load(open('results.pkl','rb')); mt=pickle.load(open('metrics.pkl','rb')); M=mt['M']; ideal,nadir=mt['ideal'],mt['nadir']
names=['NSGA-II','HNSGA-PSO','MOEA/D']
col={'NSGA-II':'#7f7f7f','HNSGA-PSO':'#0072B2','MOEA/D':'#D55E00'}
mk={'NSGA-II':'s','HNSGA-PSO':'o','MOEA/D':'^'}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'savefig.dpi':300})
sc=make_scenarios(); pr=Problem(sc)
# ---- knee of best-HV HNSGA-PSO run
hv=M['HNSGA-PSO']['hv']; rb=int(np.argmax(hv)); r=res['HNSGA-PSO'][rb]
m=r['cv']<=1e-12; Xf,Ff=r['X'][m],r['F'][m]; k=nds(Ff,np.zeros(len(Ff)))==0; Xf,Ff=Xf[k],Ff[k]
fn=(Ff-Ff.min(0))/np.maximum(Ff.max(0)-Ff.min(0),1e-12); ki=int(np.argmin(np.sqrt((fn**2).sum(1))))
xk=Xf[ki]; dk=pr.decode(xk); det=pr.details(xk)
Fk=Ff[ki]
# cases
prn=Problem(sc,evcs=False); d0=prn.details(np.zeros(6),hubs=False)
xb=np.array([(18-2)/31,(33-2)/31,0,0,0,0.]); db=pr.details(xb)
def nbelow(Vw): return int((Vw[1:]<0.9).sum())
info=dict(run_best=rb+1,seed=100+rb,knee=dict(h1=int(dk['h1'])+1,h2=int(dk['h2'])+1,pb=int(dk['pb'])+1,pv=float(dk['pv']),wb=int(dk['wb'])+1,wt=float(dk['wt']),cost=float(Fk[0]/1e6),loss=float(Fk[1]),vdi=float(Fk[2]),vmin=float(det['vmin']),ren=float(det['ren']),capex=float(det['capex']/1e6)),
 base=dict(loss=float(d0['loss']),vmin=float(d0['vmin'])),weak=dict(loss=float(db['loss']),vmin=float(db['vmin']),nbelow=nbelow(db['Vworst'])),
 nfront=int(len(Ff)))
print(json.dumps(info,indent=1))
# loss range of knee-run front clusters
print('front loss range',Ff[:,1].min(),Ff[:,1].max()); print(np.sort(Ff[:,1]).round(0))
# ---------------- Fig 6
bus=np.arange(1,34)
dk_det=pr.details(xk)
fig,ax=plt.subplots(figsize=(6.4,3.6))
ax.plot(bus,d0['Vworst'],'-s',ms=3.5,color='#7f7f7f',label='No EVCS (base case)')
ax.plot(bus,db['Vworst'],'-^',ms=3.5,color='#D55E00',label='Hubs at buses 18 and 33, no DG')
ax.plot(bus,dk_det['Vworst'],'-o',ms=3.5,color='#0072B2',label='HNSGA-PSO knee plan')
ax.axhline(0.90,ls='--',color='k',lw=.8); ax.text(1,0.903,'0.90 pu limit',fontsize=8)
ax.set_xlabel('Bus number'); ax.set_ylabel('Voltage magnitude (pu)'); ax.legend(frameon=False,fontsize=8,loc='lower left'); ax.set_xlim(1,33)
fig.tight_layout(); fig.savefig('figs/fig6.png'); plt.close()
# ---------------- Fig 7 schematic
pos={}
for b in range(1,19): pos[b]=(b-1,0)
for i,b in enumerate(range(19,23)): pos[b]=(1+i,1)
for i,b in enumerate(range(23,26)): pos[b]=(2+i,-1)
for i,b in enumerate(range(26,34)): pos[b]=(5+i,-2)
fig,ax=plt.subplots(figsize=(10,4.2))
for f,t,_,_ in BR:
    ax.plot([pos[f][0],pos[t][0]],[pos[f][1],pos[t][1]],color='#c9c9c0',lw=2,zorder=1)
ax.scatter([p[0] for p in pos.values()],[p[1] for p in pos.values()],c='k',s=22,zorder=3)
for b,(x,y) in pos.items(): ax.text(x,y+(0.22 if y>=0 else -0.32),str(b),ha='center',fontsize=7,color='#444')
ax.scatter([0],[0],marker='s',s=160,facecolors='none',edgecolors='k',lw=1.5,zorder=4)
def ring(b,c,face):
    ax.scatter([pos[b][0]],[pos[b][1]],s=230,facecolors=face,edgecolors=c,lw=2,zorder=2)
ring(info['knee']['h1'],'#1f77d0','#dbe8f7'); 
if info['knee']['h2']!=info['knee']['h1']: ring(info['knee']['h2'],'#1f77d0','#dbe8f7')
ring(info['knee']['pb'],'#1aa57a','#d5f0e6'); ring(info['knee']['wb'],'#d6457c','#f9dde8')
def lab(b,t,dy):
    if pos[b][1]>0: dy=abs(dy)+0.15
    ax.text(pos[b][0],pos[b][1]+dy,t,fontsize=8.5,fontweight='bold',ha='center')
lab(info['knee']['h1'],'EVCS 300 kW',-0.62)
if info['knee']['h2']!=info['knee']['h1']: lab(info['knee']['h2'],'EVCS 300 kW',-0.62)
else: lab(info['knee']['h1'],'EVCS 600 kW (2 hubs)',-0.62)
lab(info['knee']['pb'],f"PV {info['knee']['pv']:.0f} kWp",-0.62)
lab(info['knee']['wb'],f"WT {info['knee']['wt']:.0f} kW",0.62)
ax.text(-0.95,0.62,'Substation',fontsize=8.5)
ax.set_xlim(-1,19); ax.set_ylim(-3.3,2.3); ax.axis('off')
leg=[Line2D([],[],marker='o',ls='',mfc='#dbe8f7',mec='#1f77d0',mew=2,ms=10,label='EV charging hub (6 × 50 kW DC)'),
 Line2D([],[],marker='o',ls='',mfc='#d5f0e6',mec='#1aa57a',mew=2,ms=10,label='PV plant'),
 Line2D([],[],marker='o',ls='',mfc='#f9dde8',mec='#d6457c',mew=2,ms=10,label='Wind turbine'),
 Line2D([],[],marker='s',ls='',mfc='none',mec='k',mew=1.5,ms=10,label='Substation (slack bus)')]
ax.legend(handles=leg,loc='lower right',frameon=False,fontsize=8.5,ncol=2,bbox_to_anchor=(1,0.0))
fig.tight_layout(); fig.savefig('figs/fig7.png'); plt.close()
# ---------------- Fig 8 Pareto
fig,ax=plt.subplots(figsize=(6.4,3.8))
for n in names:
    hvn=M[n]['hv']; ri=int(np.argsort(hvn)[len(hvn)//2])    # median-HV run
    for rr in res[n]:
        mm=rr['cv']<=1e-12
        if mm.any(): ax.scatter(rr['F'][mm,1],rr['F'][mm,2],s=6,color=col[n],alpha=.08,marker=mk[n],lw=0)
for n in names:
    hvn=M[n]['hv']; ri=int(np.argsort(hvn)[len(hvn)//2])
    rr=res[n][ri]; mm=rr['cv']<=1e-12; f=rr['F'][mm]; q=nds(f,np.zeros(len(f)))==0
    ax.scatter(f[q,1],f[q,2],s=22,color=col[n],marker=mk[n],label=f'{n} (median-HV run)',edgecolors='w',lw=.4)
ax.scatter([Fk[1]],[Fk[2]],s=130,facecolors='none',edgecolors='k',lw=1.4,label='Knee plan (Section 7.2)',zorder=5)
ax.set_xlabel('Annual energy loss (MWh/yr)'); ax.set_ylabel('Voltage deviation index (VDI)'); ax.legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig('figs/fig8.png'); plt.close()
# ---------------- Fig 9 convergence
fig,ax=plt.subplots(figsize=(6.4,3.8))
for n in names:
    L=max(len(t) for t in M[n]['trace']); A=np.array([np.pad(t,(0,L-len(t)),mode='edge') for t in M[n]['trace']])
    ev=np.linspace(0,3600,L) if n!='MOEA/D' else np.linspace(0,3575,L)
    xs=np.arange(L)*(60 if n!='MOEA/D' else 55)+ (60 if n!='MOEA/D' else 55)
    q1,md,q3=np.percentile(A,[25,50,75],0)
    ax.plot(xs,md,color=col[n],lw=1.8,label=n); ax.fill_between(xs,q1,q3,color=col[n],alpha=.15,lw=0)
ax.set_xlabel('Function evaluations'); ax.set_ylabel('Normalised hypervolume'); ax.legend(frameon=False); ax.set_xlim(0,3600)
fig.tight_layout(); fig.savefig('figs/fig9.png'); plt.close()
# ---------------- Fig 10 boxplots
fig,axs=plt.subplots(1,3,figsize=(10,3.6))
for ax,(key,lab) in zip(axs,[('hv','Hypervolume (↑)'),('gd','Generational distance (↓)'),('sp','Spacing (↓)')]):
    data=[np.maximum(M[n][key],1e-4) if key=='gd' else M[n][key] for n in names]
    bp=ax.boxplot(data,widths=.5,patch_artist=True,showfliers=False,medianprops=dict(color='k'))
    for p,n in zip(bp['boxes'],names): p.set_facecolor(col[n]); p.set_alpha(.25)
    rng=np.random.default_rng(1)
    for i,n in enumerate(names):
        ax.scatter(i+1+rng.uniform(-.15,.15,len(data[i])),data[i],s=10,color=col[n],alpha=.85,lw=0,zorder=3)
    ax.set_xticks([1,2,3]); ax.set_xticklabels(names,fontsize=8.5); ax.set_title(lab,fontsize=10)
    if key in('gd','sp'): ax.set_yscale('log')
fig.tight_layout(); fig.savefig('figs/fig10.png'); plt.close()
# ---------------- Fig 11 CD
st=json.load(open('stats.json'))
k=3; N=30; CD=2.343*np.sqrt(k*(k+1)/(6*N))
fig,axs=plt.subplots(1,3,figsize=(10,2.1))
for ax,(key,lab) in zip(axs,[('hv','Hypervolume'),('gd','Generational distance'),('sp','Spacing')]):
    rk=np.array(st[key]['ranks']); ax.set_xlim(3.3,0.7); ax.set_ylim(0,1.6); ax.axis('off')
    ax.plot([3,1],[1,1],'k',lw=1)
    for t in (1,2,3): ax.plot([t,t],[1,1.06],'k',lw=1); ax.text(t,1.12,str(t),ha='center',fontsize=8)
    o=np.argsort(rk)
    for j,i in enumerate(o):
        y=0.75-0.0*j
        ax.plot([rk[i],rk[i]],[1,0.55-0.2*j],color=col[names[i]],lw=1)
        ax.text(rk[i]+(0.05 if True else 0),0.5-0.2*j,f'{names[i]} ({rk[i]:.2f})',fontsize=7.5,color=col[names[i]],ha='left')
    ax.plot([3.1,3.1-CD],[1.45,1.45],'k',lw=2); ax.text(3.1-CD/2,1.52,f'CD = {CD:.2f}',ha='center',fontsize=7.5)
    # clique bars
    ys=0.2
    for a in range(3):
        for b in range(a+1,3):
            pass
    cl=[]; srt=list(o)
    for a in range(3):
        grp=[srt[a]]
        for b in range(a+1,3):
            if rk[srt[b]]-rk[srt[a]]<CD: grp.append(srt[b])
        if len(grp)>1 and not any(set(grp)<=set(c) for c in cl): cl.append(grp)
    for j,g in enumerate(cl):
        ax.plot([rk[g[0]]-0.03,rk[g[-1]]+0.03],[0.9-0.0,0.9-0.0] if False else [0.88-0.06*j]*2,color='k',lw=3.5)
    ax.set_title(lab,fontsize=9,y=0.98)
fig.tight_layout(); fig.savefig('figs/fig11.png'); plt.close()
json.dump(info,open('info.json','w'),indent=1)
print('CD',CD)
