"""Publication-quality versions of Figs. 6-11 (IEEE column widths, Times-compatible font, 600 dpi PNG + vector PDF)."""
import numpy as np, pickle, json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from model import *
from algos import nds

OUT = 'figs_pub'; os.makedirs(OUT, exist_ok=True)
W1, W2 = 3.5, 7.16     # IEEE single / double column width (inches)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'axes.linewidth': 0.6,
    'lines.linewidth': 1.1, 'lines.markersize': 3.5, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.grid': True, 'grid.linewidth': 0.35,
    'grid.alpha': 0.35, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'legend.frameon': False})

res = pickle.load(open('results.pkl', 'rb')); mt = pickle.load(open('metrics.pkl', 'rb')); M = mt['M']
st = json.load(open('stats.json'))
names = ['NSGA-II', 'HNSGA-PSO', 'MOEA/D']
col = {'NSGA-II': '#555555', 'HNSGA-PSO': '#0072B2', 'MOEA/D': '#D55E00'}
mk = {'NSGA-II': 's', 'HNSGA-PSO': 'o', 'MOEA/D': '^'}
ls = {'NSGA-II': '--', 'HNSGA-PSO': '-', 'MOEA/D': '-.'}
hatch = {'NSGA-II': '//', 'HNSGA-PSO': '', 'MOEA/D': '..'}


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png'); fig.savefig(f'{OUT}/{name}.pdf'); plt.close(fig)


def panel(ax, s, dx=-0.16, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')


sc = make_scenarios(); pr = Problem(sc)
hv = M['HNSGA-PSO']['hv']; rb = int(np.argmax(hv)); r = res['HNSGA-PSO'][rb]
m = r['cv'] <= 1e-12; Xf, Ff = r['X'][m], r['F'][m]; k = nds(Ff, np.zeros(len(Ff))) == 0; Xf, Ff = Xf[k], Ff[k]
fn = (Ff - Ff.min(0))/np.maximum(Ff.max(0) - Ff.min(0), 1e-12); ki = int(np.argmin(np.sqrt((fn**2).sum(1))))
xk = Xf[ki]; dk = pr.decode(xk); Fk = Ff[ki]
prn = Problem(sc, evcs=False); d0 = prn.details(np.zeros(6), hubs=False)
db = pr.details(np.array([(18-2)/31, (33-2)/31, 0, 0, 0, 0.])); dkd = pr.details(xk)
kn = dict(h1=int(dk['h1'])+1, h2=int(dk['h2'])+1, pb=int(dk['pb'])+1, wb=int(dk['wb'])+1, pv=float(dk['pv']), wt=float(dk['wt']))

# ---------------- Fig. 6 ----------------
fig, ax = plt.subplots(figsize=(W1, 2.55))
bus = np.arange(1, 34)
ax.plot(bus, d0['Vworst'], ls='-', marker='s', color='#555555', label='No EVCS (base case)')
ax.plot(bus, db['Vworst'], ls='--', marker='^', color='#D55E00', label='Hubs at buses 18 and 33, no DG')
ax.plot(bus, dkd['Vworst'], ls='-', marker='o', color='#0072B2', label='HNSGA-PSO knee plan')
ax.axhline(0.90, color='k', lw=0.7, ls=':'); ax.text(1.5, 0.9035, '0.90 pu limit', ha='left', fontsize=7)
ax.set_xlim(1, 33); ax.set_ylim(0.835, 1.005); ax.set_xlabel('Bus number'); ax.set_ylabel('Voltage magnitude (pu)')
ax.legend(loc='lower left', handlelength=2.2)
fig.tight_layout(pad=0.4); save(fig, 'Fig6_voltage_profile')

# ---------------- Fig. 7 ----------------
pos = {}
for b in range(1, 19): pos[b] = (b-1, 0)
for i, b in enumerate(range(19, 23)): pos[b] = (1+i+1, 1)
for i, b in enumerate(range(23, 26)): pos[b] = (2+i+1, -1)
for i, b in enumerate(range(26, 34)): pos[b] = (5+i+1, -2)
pos[19] = (1, 1)
for i, b in enumerate(range(19, 23)): pos[b] = (1+i, 1)
for i, b in enumerate(range(23, 26)): pos[b] = (2+i, -1)
for i, b in enumerate(range(26, 34)): pos[b] = (5+i, -2)
fig, ax = plt.subplots(figsize=(W2, 2.9))
for f, t, _, _ in BR:
    ax.plot([pos[f][0], pos[t][0]], [pos[f][1], pos[t][1]], color='#b5b5b0', lw=1.3, zorder=1)
ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()], c='k', s=7, zorder=3)
for b, (x, y) in pos.items():
    ax.text(x, y + (0.22 if y >= 0 else -0.34), str(b), ha='center', fontsize=6, color='#444444')
ax.scatter([0], [0], marker='s', s=70, facecolors='none', edgecolors='k', lw=1.1, zorder=4)
def ring(b, ec, fc, mkr='o'):
    ax.scatter([pos[b][0]], [pos[b][1]], s=110, marker=mkr, facecolors=fc, edgecolors=ec, lw=1.3, zorder=2)
ring(kn['h1'], '#0072B2', '#d6e6f4'); ring(kn['h2'], '#0072B2', '#d6e6f4')
ring(kn['pb'], '#009E73', '#d3efe6'); ring(kn['wb'], '#CC79A7', '#f6dfeb')
def lab(b, t, dy, dx=0):
    ax.text(pos[b][0]+dx, pos[b][1]+dy, t, fontsize=7, fontweight='bold', ha='center')
ax.text(pos[kn['h2']][0]+0.35, pos[kn['h2']][1]+0.38, 'EVCS hub\n(300 kW)', fontsize=7, fontweight='bold', ha='left', va='bottom')
ax.text(pos[kn['h1']][0]-0.3, pos[kn['h1']][1]-0.5, 'EVCS hub\n(300 kW)', fontsize=7, fontweight='bold', ha='left', va='top')
lab(kn['pb'], f"PV {kn['pv']:.0f} kWp", -0.7); lab(kn['wb'], f"WT {kn['wt']:.0f} kW", 0.6)
ax.text(-0.45, 0.55, 'Substation\n(slack bus)', fontsize=6.5, ha='left', va='bottom')
ax.set_xlim(-0.8, 18.5); ax.set_ylim(-3.1, 2.1); ax.axis('off')
leg = [Line2D([], [], marker='o', ls='', mfc='#d6e6f4', mec='#0072B2', mew=1.3, ms=7, label='EV charging hub (6 × 50 kW DC)'),
       Line2D([], [], marker='o', ls='', mfc='#d3efe6', mec='#009E73', mew=1.3, ms=7, label='PV plant'),
       Line2D([], [], marker='o', ls='', mfc='#f6dfeb', mec='#CC79A7', mew=1.3, ms=7, label='Wind turbine'),
       Line2D([], [], marker='s', ls='', mfc='none', mec='k', mew=1.1, ms=7, label='Substation')]
ax.legend(handles=leg, loc='lower right', ncol=2, bbox_to_anchor=(1.0, -0.02), fontsize=7)
fig.tight_layout(pad=0.3); save(fig, 'Fig7_network_knee_plan')

# ---------------- Fig. 8 ----------------
fig, ax = plt.subplots(figsize=(W1, 3.1))
for n in names:
    for rr in res[n]:
        mm = rr['cv'] <= 1e-12
        if mm.any(): ax.scatter(rr['F'][mm, 1], rr['F'][mm, 2], s=4, color=col[n], alpha=0.07, marker=mk[n], lw=0, rasterized=True)
for n in names:
    hvn = M[n]['hv']; ri = int(np.argsort(hvn)[len(hvn)//2]); rr = res[n][ri]
    mm = rr['cv'] <= 1e-12; f = rr['F'][mm]; q = nds(f, np.zeros(len(f))) == 0
    ax.scatter(f[q, 1], f[q, 2], s=11, color=col[n], marker=mk[n], edgecolors='w', lw=0.25, label=f'{n} (median-HV run)', zorder=3)
ax.scatter([Fk[1]], [Fk[2]], s=70, facecolors='none', edgecolors='k', lw=0.9, label='Knee plan', zorder=5)
ax.set_xlabel('Annual energy loss (MWh/yr)'); ax.set_ylabel('Voltage deviation index (VDI)'); ax.set_ylim(None, 0.0575)
ax.legend(loc='upper right', handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout(pad=0.4); save(fig, 'Fig8_pareto_fronts')

# ---------------- Fig. 9 ----------------
fig, ax = plt.subplots(figsize=(W1, 2.55))
for n in names:
    L = max(len(t) for t in M[n]['trace']); A = np.array([np.pad(t, (0, L-len(t)), mode='edge') for t in M[n]['trace']])
    step = 60 if n != 'MOEA/D' else 55
    xs = (np.arange(L)+1)*step
    q1, md, q3 = np.percentile(A, [25, 50, 75], 0)
    ax.plot(xs, md, color=col[n], ls=ls[n], label=n); ax.fill_between(xs, q1, q3, color=col[n], alpha=0.13, lw=0)
ax.set_xlabel('Function evaluations'); ax.set_ylabel('Normalized hypervolume'); ax.set_xlim(0, 3600); ax.set_ylim(0, None)
ax.legend(loc='lower right', handlelength=2.4)
fig.tight_layout(pad=0.4); save(fig, 'Fig9_hv_convergence')

# ---------------- Fig. 10 ----------------
fig, axs = plt.subplots(1, 3, figsize=(W2*0.82, 2.6))
for ax, (key, lab_, s_) in zip(axs, [('hv', 'Hypervolume (↑)', '(a)'), ('gd', 'Generational distance (↓)', '(b)'), ('sp', 'Spacing (↓)', '(c)')]):
    data = [np.maximum(M[n][key], 1e-4) if key == 'gd' else M[n][key] for n in names]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color='k', lw=1.0),
                    whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7), boxprops=dict(lw=0.7))
    rng = np.random.default_rng(1)
    for i, n in enumerate(names):
        bp['boxes'][i].set_facecolor(col[n]); bp['boxes'][i].set_alpha(0.25); bp['boxes'][i].set_hatch(hatch[n])
        ax.scatter(i+1+rng.uniform(-0.14, 0.14, len(data[i])), data[i], s=5, color=col[n], marker=mk[n], lw=0, zorder=3, alpha=0.9)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['NSGA-II', 'HNSGA-\nPSO', 'MOEA/D'], fontsize=8)
    ax.set_ylabel(lab_)
    if key in ('gd', 'sp'): ax.set_yscale('log')
    panel(ax, s_, dx=-0.3 if key != 'hv' else -0.24)
fig.tight_layout(pad=0.4, w_pad=1.2); save(fig, 'Fig10_indicator_boxplots')

# ---------------- Fig. 11 (critical-difference diagrams) ----------------
k_, N_ = 3, 30; CD = 2.343*np.sqrt(k_*(k_+1)/(6*N_))
fig, axs = plt.subplots(3, 1, figsize=(W2*0.72, 3.9))
for ax, (key, ttl, s_) in zip(axs, [('hv', 'Hypervolume', '(a)'), ('gd', 'Generational distance', '(b)'), ('sp', 'Spacing', '(c)')]):
    rk = np.array(st[key]['ranks'])
    ax.set_xlim(0.8, 3.2); ax.set_ylim(-1.55, 1.2); ax.axis('off')
    ax.plot([1, 3], [0, 0], 'k', lw=0.8)
    for t in (1, 1.5, 2, 2.5, 3):
        ax.plot([t, t], [0, 0.07 if t % 1 else 0.13], 'k', lw=0.7)
    for t in (1, 2, 3): ax.text(t, 0.2, str(t), ha='center', fontsize=7.5)
    ax.text(0.8, 0.95, f'{s_} {ttl}', fontsize=8.5, fontweight='bold', va='center')
    ax.plot([2.2, 2.2+CD], [0.95, 0.95], 'k', lw=1.4); ax.text(2.2+CD/2, 1.08, f'CD = {CD:.2f}', ha='center', fontsize=7)
    o = list(np.argsort(rk))
    left = o[:2]; right = o[2:]
    # labels: best on the left column, worst on the right
    ypos = {}
    for j, i in enumerate(left):
        y = -0.45 - 0.5*j; ypos[i] = y
        ax.plot([rk[i], rk[i], 0.95], [0, y, y], color=col[names[i]], lw=0.8)
        ax.text(0.93, y, f'{names[i]}  ({rk[i]:.2f})', ha='right', va='center', fontsize=7.5, color=col[names[i]])
    for j, i in enumerate(right):
        y = -0.45 - 0.5*j; ypos[i] = y
        ax.plot([rk[i], rk[i], 3.05], [0, y, y], color=col[names[i]], lw=0.8)
        ax.text(3.07, y, f'({rk[i]:.2f})  {names[i]}', ha='left', va='center', fontsize=7.5, color=col[names[i]])
    # cliques: bars joining solvers not separated by the Nemenyi test
    cl = []
    for a in range(3):
        grp = [o[a]]
        for b in range(a+1, 3):
            if rk[o[b]] - rk[o[a]] < CD: grp.append(o[b])
        if len(grp) > 1 and not any(set(grp) <= set(c) for c in cl): cl.append(grp)
    for j, g in enumerate(cl):
        ax.plot([rk[g[0]]-0.02, rk[g[-1]]+0.02], [0.4+0.0*j]*2, color='k', lw=2.8, solid_capstyle='butt')
ax.set_xlim(0.0, 4.0)
for a_ in axs: a_.set_xlim(0.15, 3.95)
fig.tight_layout(pad=0.3, h_pad=0.4); save(fig, 'Fig11_critical_difference')
print('done')
