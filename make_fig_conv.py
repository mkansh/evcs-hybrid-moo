"""Figure 12 of the paper: convergence of the five solvers and final fronts of the median runs."""
import pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'font.size': 7.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7})
NAMES = ['NSGA-II', 'HNSGA-PSO', 'MOEA/D', 'NSGA-III', 'MOPSO']
COL = {'NSGA-II': '#000000', 'HNSGA-PSO': '#D55E00', 'MOEA/D': '#009E73', 'NSGA-III': '#0072B2', 'MOPSO': '#CC79A7'}
LS = {'NSGA-II': '-', 'HNSGA-PSO': '--', 'MOEA/D': '-.', 'NSGA-III': ':', 'MOPSO': '-'}
MK = {'NSGA-II': 'o', 'HNSGA-PSO': 's', 'MOEA/D': '^', 'NSGA-III': 'D', 'MOPSO': 'v'}
TR = {e: pickle.load(open(f'trace_{e}.pkl', 'rb')) for e in ('real', 'discrete')}
REF = np.load('norm_legacy.npz')['ref']
fig, axs = plt.subplots(1, 3, figsize=(6.95, 2.8), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
for ax, enc, lab, ttl in ((axs[0], 'real', '(a)', 'Continuous siting genes'), (axs[1], 'discrete', '(b)', 'Integer siting genes')):
    for n in NAMES:
        H = np.array([r['hv'] for r in TR[enc][n]]); x = TR[enc][n][0]['evals']
        q1, med, q3 = np.percentile(H, [25, 50, 75], axis=0)
        ax.fill_between(x, q1, q3, color=COL[n], alpha=0.10, lw=0)
        ax.plot(x, med, color=COL[n], ls=LS[n], lw=1.0, marker=MK[n], markevery=(8, 10), ms=2.6, mfc='white' if n != 'NSGA-II' else 'black', mew=0.6)
    ax.set_xlim(0, 3600); ax.set_ylim(0, 0.8); ax.set_xlabel('Function evaluations'); ax.set_title(ttl, fontsize=7.5, pad=3)
    ax.text(-0.02, 1.04, lab, transform=ax.transAxes, fontweight='bold', fontsize=8.5, ha='left', va='bottom')
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, lw=0.3, color='0.85'); ax.set_axisbelow(True)
    ax.axhline(0.741, ls=(0, (4, 2)), color='0.5', lw=0.6)
axs[0].set_ylabel('Median normalized HV'); axs[1].set_yticklabels([])
ax = axs[2]
ax.scatter(REF[:, 1], REF[:, 2], s=3.5, color='0.62', lw=0, zorder=1)
for n in NAMES:
    runs = TR['real'][n]; k = np.argsort([r['hv_final'] for r in runs])[len(runs)//2]; F = runs[k]['F']
    ax.scatter(F[:, 1], F[:, 2], s=5, marker=MK[n], facecolor='none' if n != 'NSGA-II' else 'black', edgecolor=COL[n], lw=0.6, zorder=3)
ax.set_xlabel('Annual energy loss (MWh)'); ax.set_ylabel('VDI (pu$^2$)'); ax.set_title('Median runs, continuous genes', fontsize=7.5, pad=3)
ax.text(-0.02, 1.04, '(c)', transform=ax.transAxes, fontweight='bold', fontsize=8.5, ha='left', va='bottom')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
h = [Line2D([], [], color=COL[n], ls=LS[n], lw=1.0, marker=MK[n], ms=3, mfc='white' if n != 'NSGA-II' else 'black', mew=0.6, label=n) for n in NAMES]
h.append(Line2D([], [], color='0.5', ls=(0, (4, 2)), lw=0.6, label='Pooled reference HV (a, b)'))
h.append(Line2D([], [], color='0.62', marker='o', ls='', ms=2.6, label='Pooled reference front (c)'))
fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.1, handletextpad=0.5)
fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.31, wspace=0.32)
fig.savefig('fig_conv.png', dpi=300); fig.savefig('fig_conv.pdf'); plt.close(fig); print('ok')
