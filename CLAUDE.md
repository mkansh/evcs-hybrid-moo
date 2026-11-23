# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

Reproducible benchmark code for the review paper *"Hybrid Metaheuristics for
Renewable-Integrated EV Charging Station Planning: A Critical Review,
Reproducible Benchmark and Implications for Iraq and the MENA Region"*
(Section 7, Sections 7.5-7.6 and the Appendix). It is an independent
re-implementation written for the review, not the code of the surveyed studies.
Model assumptions are listed in Table 10 of the paper.

It is a flat collection of research scripts (no package, no build step, no test
framework). Results (`*.pkl`, `*.npy`, `*.npz`, `*_output.txt`, `*.log`,
`stats.json`) and figures (`fig*.png/pdf`) are committed so the paper's numbers
can be checked without re-running experiments.

## Environment

```bash
pip install -r requirements.txt   # numpy, scipy, scikit-learn, matplotlib
```

Reference versions (stated in README; these match the sandbox):
Python 3.13, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2.
Results depend on library versions and NumPy's RNG, so small numerical
differences are expected with other versions.

All scripts are run from the repo root and read/write files in the current
directory.

## Core modules

| File | Role |
|---|---|
| `model.py` | IEEE 33-bus data [1], backward/forward-sweep (BFS) power flow, Beta/Weibull Monte Carlo scenarios + k-means reduction (`make_scenarios`, fixed seed 2024), PV/WT models, `Problem` (3 objectives, 6 genes: hub1, hub2, PV bus, WT bus, 2 sizes). |
| `algos.py` | NSGA-II [3], HNSGA-PSO hybrid (Algorithm 1 in the paper; `nsga2(..., hybrid=True)`), MOEA/D [4], Deb's constrained domination, `nds`, `crowding`. Global config: `POP=GENS=60` (3600 evaluations), `ENC`, `NBG`, `NCAND`, `PARTS`, `PARAMS`. |
| `algos_extra.py` | NSGA-III [5] and MOPSO [6]. |
| `model69.py` | IEEE 69-bus feeder [2] (`bus69.npy`, `br69.npy`, from MATPOWER case69) with rule-based BESS and 4 objectives (`Problem69`). |
| `hvtools.py` | Shared `feas_nd` and normalised 3-D hypervolume using the fixed normalisation in `norm_legacy.npz`. |

Every solver returns `(X, F, cv, trace)`; feasible means `cv <= 1e-12`.
All objectives are minimised.

### Configuration knobs (module globals / env vars)

- `EVCS_ENC=real|discrete` — `real` is the published continuous encoding;
  `discrete` uses integer bus genes. Read once at `import algos`, so set it
  before importing (or assign `algos.ENC` directly, as `tune.py`/`ablate.py` do).
- `EVCS_ENC_PARTS` — subset of `init,cross,mut` for the discrete encoding (ablation).
- `algos.PARAMS` — SBX/PM/PSO parameters; scripts reset it from a deep copy of
  the defaults after each run.
- `algos.NBG`, `algos.NCAND` — must be set to `5` and `68` for the 69-bus model (see `run69.py`).
- `NR` (env, `run69.py` only) — number of runs, default 30.
- `EXTRA_TAG` (env, `analyze_extra.py`) — `''` for legacy scenarios, `_daily` for daily-feature scenarios.

**Invariant:** defaults in `algos.py` must stay bit-identical to the published
code. `regress.py` checks this (max abs difference must be 0.0).

## Commands

There is no unit-test suite. "Testing" means the smoke test and the
regression check below.

### Quick smoke test (~5 s)

```bash
python3 -c "from model import *; from algos import nsga2; p=Problem(make_scenarios()); X,F,cv,_=nsga2(p,300,False); print(F.shape, p.nfe)"
# expected: (60, 3) 3600
```

### Regression check against a previous version

```bash
# regress.py <dir_of_code_version> <real|discrete> <out.npy>
python3 regress.py /path/to/old_version real old.npy
python3 regress.py .                    real new.npy
python3 -c "import numpy as np; print(np.abs(np.load('old.npy')-np.load('new.npy')).max())"   # must be 0.0
```

Run `regress.py` for both encodings after any change to `model.py`, `algos.py`
or `algos_extra.py`.

### Original benchmark (33-bus, Table 11-12, Figs. 6-11)

```bash
python3 run_all.py    # ~7 min, seeds 100-129 -> results.pkl
python3 analyze.py    # -> metrics.pkl, stats.json
python3 sens.py       # Table 12, seeds 101-103 -> sens.pkl
python3 figs.py       # Figs. 6-11 -> figs/ (created automatically)
python3 figs_pub.py   # publication versions -> figs_pub/ (png + pdf)
```

### Extended experiments (Sections 7.5-7.6, Figs. 12-14)

```bash
EVCS_ENC=real     python3 run_extra.py res_extra_real.pkl       # seeds 300-329
EVCS_ENC=discrete python3 run_extra.py res_extra_discrete.pkl
python3 analyze_extra.py                       # -> analysis_extra.pkl
EXTRA_TAG=_daily python3 analyze_extra.py      # -> analysis_extra_daily.pkl
python3 analyze_equiv.py                       # IGD+, epsilon, bootstrap equivalence (margin 0.02)

python3 tune.py NSGA-II HNSGA-PSO MOEA/D NSGA-III MOPSO        # seeds 500-509 -> tune_*.pkl
python3 test_tuned.py NSGA-II HNSGA-PSO MOEA/D NSGA-III MOPSO  # seeds 800-829 -> test_*.pkl
python3 analyze_tuned.py                                       # -> tuned_all.pkl

python3 ablate.py      # seeds 700-729, ~15 min -> ablate.pkl, neutrality.pkl
python3 landscape.py   # exhaustive 540 672 plans, ~13 min -> landscape.pkl, landscape_stats.pkl

EVCS_ENC=real     python3 run69.py r69_real.pkl       # 69-bus, seeds 400-429
EVCS_ENC=discrete python3 run69.py r69_discrete.pkl
python3 analyze69.py   # -> analysis69.pkl, hv69.pkl

EVCS_ENC=real     python3 run_trace.py trace_real.pkl       # HV vs evaluations
EVCS_ENC=discrete python3 run_trace.py trace_discrete.pkl
python3 make_fig_conv.py   # Fig. 12 (fig_conv.*)
python3 make_figs.py       # Figs. 13-14 (fig12.*, fig13.*)
```

Each solver run takes about 4-9 s on one core; a 5-solver x 30-seed script
takes 10-20 min. Run long jobs in the background and capture stdout to a
`*_output.txt` / `*.log` file, as the existing outputs do.

## Seeds (do not reuse across experiments)

| Seeds | Experiment |
|---|---|
| 100-129 | original benchmark (`run_all.py`) |
| 101-103 | sensitivity (`sens.py`) |
| 200-229 | earlier set listed in README (no script in repo uses it) |
| 300-329 | five-solver confirmation, traces, equivalence |
| 400-429 | 69-bus + BESS |
| 500-509 | tuning |
| 700-729 | ablation |
| 800-829 | tuned test |

Scenario generation always uses seed 2024. New experiments need a fresh,
non-overlapping seed block; record it in the README.

## Code style

Match the existing style rather than reformatting:

- Compact, NumPy-vectorised research code; short names (`F`, `X`, `cv`, `hv`,
  `prob`, `rng`); multiple statements per line with `;` are accepted.
- `np.random.default_rng(seed)` everywhere; never use the global NumPy RNG.
- Module docstring at the top of each script stating purpose, seeds and usage.
- Inline `#` comments only for non-obvious choices (units, encodings, paper references).
- Pickle for results, `matplotlib.use('Agg')` for figures; save figures as
  PNG (300-600 dpi) and PDF with `pdf.fonttype = 42`.
- No formatter or linter is configured; do not run one over the whole repo, as it
  would create large diffs in files tied to published results.
- Helper functions (`feas_nd`, `hv2d`, `hv3d`, `a12`, `holm`) are duplicated in
  several analysis scripts on purpose so each script reproduces its table
  independently. Do not refactor them into a shared module without asking.

## Rules for changes

1. Never overwrite a committed result file (`*.pkl`, `*.npy`, `*.npz`,
   `stats.json`, `*_output.txt`, figures) unless asked; write new outputs to new
   names or to a scratch directory.
2. Keep defaults bit-identical (run `regress.py`); add new behaviour behind a
   flag or env var.
3. Do not change `make_scenarios` defaults, seeds, `POP/GENS` or the
   normalisation in `norm_legacy.npz`: the paper's numbers depend on them.
4. Statistical reporting follows the paper: Kruskal-Wallis and Friedman
   omnibus tests, pairwise Wilcoxon rank-sum with Holm correction [7], and
   Vargha-Delaney A12 effect size [8].
5. Update `README.md` / `README_extended.md` when adding scripts or seeds.

## Writing conventions (paper text, README, reports)

- Academic, professional tone.
- IEEE citation style with numbered in-text citations, e.g. "NSGA-II [3]".
- Cite peer-reviewed journal sources and verify each reference (DOI) before using it.
- Use the paper's numbering for tables and figures (Table 10-12, Figs. 6-14).

## Known issues

- The "daily-feature" scenario set used for `res_extra_daily_*.pkl` is not
  implemented in the committed `model.py` (no `SCEN_MODE`), so the daily results
  cannot be regenerated. `run_extra.py` runs the legacy set only; it reproduces
  `res_extra_{real,discrete}.pkl` exactly (checked on NSGA-II, seed 300: max abs
  difference 0.0).
- `analyze69.py` and `analyze_equiv.py` write `analysis69.pkl` and `equiv.pkl`,
  which are not committed.
- No `.gitignore`; be careful not to commit scratch outputs.

## Open questions for the maintainer

Fill these in when known, so they don't have to be asked each session:

- Where is the daily-feature scenario code (`SCEN_MODE`) that produced `res_extra_daily_*.pkl`?
- Which seed block (200-229) maps to which experiment?
- Preferred workflow: commit directly to the branch, or open pull requests?

## References

[1] M. E. Baran and F. F. Wu, "Network reconfiguration in distribution systems
for loss reduction and load balancing," *IEEE Trans. Power Del.*, vol. 4, no. 2,
pp. 1401-1407, Apr. 1989.

[2] M. E. Baran and F. F. Wu, "Optimal capacitor placement on radial
distribution systems," *IEEE Trans. Power Del.*, vol. 4, no. 1, pp. 725-734,
Jan. 1989.

[3] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist
multiobjective genetic algorithm: NSGA-II," *IEEE Trans. Evol. Comput.*,
vol. 6, no. 2, pp. 182-197, Apr. 2002.

[4] Q. Zhang and H. Li, "MOEA/D: A multiobjective evolutionary algorithm based
on decomposition," *IEEE Trans. Evol. Comput.*, vol. 11, no. 6, pp. 712-731,
Dec. 2007.

[5] K. Deb and H. Jain, "An evolutionary many-objective optimization algorithm
using reference-point-based nondominated sorting approach, Part I: Solving
problems with box constraints," *IEEE Trans. Evol. Comput.*, vol. 18, no. 4,
pp. 577-601, Aug. 2014.

[6] C. A. Coello Coello, G. T. Pulido, and M. S. Lechuga, "Handling multiple
objectives with particle swarm optimization," *IEEE Trans. Evol. Comput.*,
vol. 8, no. 3, pp. 256-279, Jun. 2004.

[7] S. Holm, "A simple sequentially rejective multiple test procedure,"
*Scand. J. Statist.*, vol. 6, no. 2, pp. 65-70, 1979.

[8] A. Vargha and H. D. Delaney, "A critique and improvement of the CL common
language effect size statistics of McGraw and Wong," *J. Educ. Behav. Stat.*,
vol. 25, no. 2, pp. 101-132, 2000.
