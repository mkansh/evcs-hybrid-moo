"""Figures 12 and 13 of the paper: hypervolume distributions (box plots with individual runs)."""
import pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'font.size': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'mathtext.fontset': 'stix'})
CONT, TUNED, INT = '#D55E00', '#009E73', '#0072B2'            # Okabe-Ito (colour-blind safe)
NAMES = ['NSGA-II', 'HNSGA-PSO', 'MOEA/D', 'NSGA-III', 'MOPSO']
TICK = ['NSGA-II', 'HNSGA-\nPSO', 'MOEA/D', 'NSGA-III', 'MOPSO']
rng = np.random.default_rng(7)

def boxes(ax, data, pos, color, width, horizontal=False, hatch=None):
    for d, p in zip(data, pos):
        bp = ax.boxplot([d], positions=[p], widths=width, orientation=('horizontal' if horizontal else 'vertical'), patch_artist=True, showfliers=False, manage_ticks=False,
                        medianprops=dict(color='black', lw=1.0), whiskerprops=dict(color=color, lw=0.7), capprops=dict(color=color, lw=0.7),
                        boxprops=dict(facecolor=color, alpha=0.30, edgecolor=color, lw=0.8))
        j = rng.uniform(-width*0.32, width*0.32, len(d))
        if horizontal: ax.scatter(d, p+j, s=3.2, color=color, alpha=0.85, lw=0, zorder=3)
        else: ax.scatter(p+j, d, s=3.2, color=color, alpha=0.85, lw=0, zorder=3)

def label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=8.5, ha='left', va='bottom')

def clean(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(axis='x', length=0)

# ----------------------------------------------------------------------------------- Fig. 12
A = {tag: pickle.load(open(f'analysis_extra{tag}.pkl', 'rb')) for tag in ('', '_daily')}
H69 = pickle.load(open('hv69.pkl', 'rb')); REF69 = 0.847
def xt(ax, x):
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=32, ha='right', rotation_mode='anchor'); ax.set_xlim(-0.6, 4.6)
fig, axs = plt.subplots(1, 3, figsize=(6.95, 2.75), sharey=True)
panels = [('(a)', 'Published scenarios', {e: A['']['HV'][e] for e in ('real', 'discrete')}, A['']['hvref']),
          ('(b)', 'Daily-feature scenarios', {e: A['_daily']['HV'][e] for e in ('real', 'discrete')}, A['_daily']['hvref']),
          ('(c)', 'IEEE 69-bus, storage, 4 obj.', {e: {n: H69[(e, n)] for n in NAMES} for e in ('real', 'discrete')}, REF69)]
x = np.arange(5)
for ax, (lab, ttl, HV, ref) in zip(axs, panels):
    boxes(ax, [HV['real'][n] for n in NAMES], x-0.19, CONT, 0.32)
    boxes(ax, [HV['discrete'][n] for n in NAMES], x+0.19, INT, 0.32)
    ax.axhline(ref, ls=(0, (4, 2)), color='0.35', lw=0.7)
    xt(ax, x); ax.set_ylim(0, 0.92); clean(ax); label(ax, lab); ax.set_title(ttl, fontsize=7.5, pad=3)
    ax.yaxis.grid(True, lw=0.3, color='0.85'); ax.set_axisbelow(True)
axs[0].set_ylabel('Normalized hypervolume (HV)')
fig.legend(handles=[Patch(facecolor=CONT, alpha=0.5, edgecolor=CONT, label='Continuous siting genes'), Patch(facecolor=INT, alpha=0.5, edgecolor=INT, label='Integer siting genes'),
                    Line2D([], [], color='0.35', ls=(0, (4, 2)), lw=0.7, label='Pooled reference HV')], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.30, wspace=0.07)
fig.savefig('fig12.png', dpi=300); fig.savefig('fig12.pdf'); plt.close(fig)

# ----------------------------------------------------------------------------------- Fig. 13
T = pickle.load(open('tuned_all.pkl', 'rb')); AB = pickle.load(open('ablate.pkl', 'rb'))
fig = plt.figure(figsize=(6.95, 2.7))
gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 2.7], wspace=0.62)
ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
boxes(ax1, [T[n]['default_cont'] for n in NAMES], x-0.27, CONT, 0.24)
boxes(ax1, [T[n]['tuned_cont'] for n in NAMES], x, TUNED, 0.24)
boxes(ax1, [T[n]['default_int'] for n in NAMES], x+0.27, INT, 0.24)
ax1.axhline(0.741, ls=(0, (4, 2)), color='0.35', lw=0.7)
xt(ax1, x); ax1.set_ylim(0, 0.86); ax1.set_ylabel('Normalized hypervolume (HV)'); clean(ax1); label(ax1, '(a)')
ax1.set_title('Tuning, published scenarios (seeds 800–829)', fontsize=7.5, pad=3)
order = ['continuous (published)', 'integer init only', 'init + crossover', 'init + mutation', 'crossover + mutation (continuous init)', 'full integer encoding']
tl = ['Continuous (published)', 'Integer init. only', 'Init. + crossover', 'Init. + mutation', 'Crossover + mutation', 'Full integer encoding']
y = np.arange(len(order))[::-1]
for k, (nm, yy) in enumerate(zip(order, y)):
    boxes(ax2, [AB[nm]], [yy], CONT if k == 0 else INT, 0.55, horizontal=True)
ax2.axvline(0.741, ls=(0, (4, 2)), color='0.35', lw=0.7)
ax2.set_yticks(y); ax2.set_yticklabels(tl); ax2.set_ylim(-0.6, len(order)-0.4); ax2.set_xlim(0.2, 0.8); ax2.set_xlabel('Normalized hypervolume (HV)'); ax2.tick_params(axis='y', length=0)
for sp in ('top', 'right'): ax2.spines[sp].set_visible(False)
ax2.xaxis.grid(True, lw=0.3, color='0.85'); ax2.set_axisbelow(True); ax1.yaxis.grid(True, lw=0.3, color='0.85'); ax1.set_axisbelow(True)
ax2.text(-0.62, 1.04, '(b)', transform=ax2.transAxes, fontweight='bold', fontsize=8.5, ha='left', va='bottom')
ax2.set_title('Ablation, NSGA-II (seeds 700–729)', fontsize=7.5, pad=3, loc='left', x=-0.4)
fig.legend(handles=[Patch(facecolor=CONT, alpha=0.5, edgecolor=CONT, label='Continuous genes, default parameters'), Patch(facecolor=TUNED, alpha=0.5, edgecolor=TUNED, label='Continuous genes, tuned'),
                    Patch(facecolor=INT, alpha=0.5, edgecolor=INT, label='Integer genes, default parameters'), Line2D([], [], color='0.35', ls=(0, (4, 2)), lw=0.7, label='Pooled reference HV')],
           loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.2, handletextpad=0.5)
fig.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.34)
fig.savefig('fig13.png', dpi=300); fig.savefig('fig13.pdf'); plt.close(fig)
print('figures written')
