# Extended experiments (Sections 7.5-7.6 of the paper)

Patched files: `algos.py` (parameters, candidate count and integer-encoding components are configurable; defaults are bit-identical to the
published code: regress.py compares all five solvers, both encodings, with the previous version, max abs difference 0.0), `algos_extra.py` (NSGA-III, MOPSO).

| Script | Purpose | Seeds |
|---|---|---|
| run_extra.py, analyze_extra.py | five solvers x two encodings x two scenario sets | 300-329 |
| analyze_equiv.py | IGD+, additive epsilon, bootstrap equivalence (margin 0.02) | 300-329 |
| tune.py, test_tuned.py, analyze_tuned.py | random-search tuning (8 configs + default, continuous encoding) and test | 500-509 / 800-829 |
| ablate.py | ablation of the integer-encoding components (NSGA-II) and mutation step sizes | 700-729 |
| landscape.py | exhaustive siting landscape (540 672 plans) | - |
| model69.py, run69.py, analyze69.py | IEEE 69-bus + rule-based BESS + 4 objectives | 400-429 |

69-bus data: Baran & Wu (1989) as distributed with MATPOWER case69 (`bus69.npy`, `br69.npy`); the BFS reproduces 224.99 kW loss and Vmin 0.9092 pu at bus 65.
Storage cost, tariff and emission factor in model69.py are illustrative planning assumptions.

Additional scripts: `run_trace.py` (HV versus evaluations and final fronts, seeds 300-329, legacy scenario; reproduces the HV of run_extra.py exactly),
`make_fig_conv.py` (Fig. 12), `make_figs.py` (Figs. 13-14).
