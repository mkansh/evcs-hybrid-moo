# evcs_hybrid_moo

Benchmark code for the review "Hybrid Metaheuristics for Renewable-Integrated EV Charging Station Planning: A Critical Review, Reproducible Benchmark and Implications for Iraq and the MENA Region" (Section 7 and Appendix).

This is an independent re-implementation written for the review. It is not the code of the surveyed studies. Model assumptions are listed in Table 10 of the paper.

## Contents

## Extended experiments (v2)

Scripts added for the revised paper (Sections 7.5-7.6):
- run_extra.py, analyze_extra.py: five solvers (NSGA-II, HNSGA-PSO, MOEA/D, NSGA-III, MOPSO), two encodings, two scenario sets (seeds 300-329)
- tune.py, test_tuned.py, analyze_tuned.py: tuning on seeds 500-509, test on seeds 800-829
- ablate.py, landscape.py, analyze_equiv.py: ablation, landscape and equivalence tests
- model69.py, run69.py, analyze69.py: IEEE 69-bus feeder with storage and four objectives (seeds 400-429)
- make_figs.py, make_fig_conv.py: figures 12-14

Seeds used overall: 100-129, 200-229, 300-329, 400-429, 500-509, 700-729 and 800-829.
See README_extended.md for details.
- `model.py`: IEEE 33-bus data, backward/forward-sweep (BFS) power flow, Beta/Weibull Monte Carlo scenarios with k-means reduction, PV thermal derating with a constant soiling factor, objectives and constraints.
- `algos.py`: NSGA-II, the HNSGA-PSO hybrid (Algorithm 1) and MOEA/D, with Deb's constrained-domination selection.
- `run_all.py`: 30 runs x 3 solvers, seeds 100-129 -> `results.pkl`.
- `analyze.py`: hypervolume, generational distance, spacing; Kruskal-Wallis and Friedman omnibus tests; pairwise Wilcoxon rank-sum with Holm correction; Vargha-Delaney A12 -> `metrics.pkl`, `stats.json`.
- `sens.py`: sensitivity analysis (Table 12, seeds 101-103).
- `figs.py`, `figs_pub.py`: figures (`figs_pub.py` produces the publication versions of Figs. 6-11).
- `results.pkl`, `metrics.pkl`, `stats.json`, `sens.pkl`: precomputed outputs, so that the statistics can be checked without re-running the 7-minute experiment.

## Requirements
Python 3 with NumPy, SciPy, scikit-learn and matplotlib (see `requirements.txt`; tested with Python 3.13, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2).

## Run order
    python3 run_all.py   # about 7 min on one CPU core
    python3 analyze.py   # needs results.pkl
    python3 sens.py      # needs metrics.pkl
    python3 figs.py      # needs results.pkl, metrics.pkl, stats.json

Seeds 100-129 reproduce Table 11 and Figs. 6-11. Seeds 101-103 reproduce Table 12. Results depend on library versions and on NumPy's random generator, so small numerical differences are possible with other versions.

## License
MIT (see `LICENSE`).
