---
title: Reproducing the figures
---

# Reproducing the figures

## Install

```bash
git clone {{ site.repo_url }}.git
cd power-law-activations-are-preserved
pip install -r requirements.txt        # numpy, matplotlib, torch
```

All commands run from the repository root, as Python modules (`python -m ...`). `torch` is needed for
the experiments with neural networks. Elsewhere it is optional; if it is present and a GPU is available,
the larger computations run on the GPU in double precision.

**Data.** Only the experiments that use CIFAR-10 need data. They read
`data/cifar-10-batches-py` (the "python version" of CIFAR-10) and download it there (163 MB) if it is
missing; `--data DIR` points elsewhere. Everything else is synthetic and generated from explicit seeds.

## Two configurations

Every experiment has a `quick` configuration, which runs on a laptop in seconds to a few minutes at
smaller sizes and with fewer random draws, and a `paper` configuration, which produces the figures on
this site. `run.py` computes and saves every eigenvalue to `results/`; `plot.py` and `summarize.py` read
those files and never recompute. So the figures can be redrawn, and the tables reprinted, without
rerunning anything.

| Figure | Commands (`paper`) | Time (`paper`, one GPU) | Output |
|---|---|---|---|
| [Main theorem](experiments/fig2-theorem.html) | `python -m experiments.fig2_theorem.run --config paper`, then `.plot`, `.summarize` | ~10 min | `figures/fig2_theorem_paper.*` |
| [Other activations](experiments/fig2-activations.html), Figure 1 | `python -m experiments.fig2_activations.run --config paper`, then `.plot`, `.summarize` | ~25 min | `figures/fig2_activations_paper.*` |
| [Other activations](experiments/fig2-activations.html), Figure 2 | `python -m experiments.fig2_activations.floor --config paper` (after `run.py`) | ~10 min (CPU) | `figures/fig2_activations_floor_paper.*` |
| [Other input distributions](experiments/fig3-universality.html) | `python -m experiments.fig3_universality.run --config paper`, then `.plot`, `.summarize` | ~25 min | `figures/fig3_universality_paper.*`, `figures/fig3_input_spectra_paper.*` |
| [Depth and training](experiments/sm5-depth.html) | `python -m experiments.sm5_depth.run --config paper --part all`, then `.plot`, `.summarize` | ~40 min | `figures/sm5_*_paper.*`, `figures/sm5_table_paper.tex` |
| [$$\alpha < 1$$](experiments/sm6-alpha.html), Figures 1–2 | `python -m experiments.sm6_alpha.run --config paper`, then `.plot` | ~45 min | `figures/sm6_principal_paper.*`, `figures/sm6_exact_paper.*` |
| [$$\alpha < 1$$](experiments/sm6-alpha.html), Figure 3 | `python -m experiments.sm6_alpha.staircase --config paper` | ~2 min (CPU) | `figures/sm6_staircase_paper.*` |

Every `run.py` accepts `--device numpy|cpu|cuda` (default: a GPU if one is available, otherwise numpy).
On a CPU the `paper` configurations take hours, dominated by eigendecompositions of
$$10^4 \times 10^4$$ matrices.
{: .caption}

## Tests

```bash
python -m tests.test_exact         # polynomial K_W vs Monte Carlo                          (~10 s)
python -m tests.test_gaussian      # non-polynomial activations: closed forms, Mehler series (~5 s)
python -m tests.test_iid           # non-Gaussian inputs: cumulant formula vs Monte Carlo     (~10 s)
python -m tests.test_population    # population spectrum vs brute force, large d, truncation (~3 min)
```

What each test checks, and to what accuracy, is listed under [Methods](methods.html#checks).

## Environment used for the figures

| | |
|---|---|
| GPU runs | NVIDIA RTX PRO 6000 Blackwell (96 GB); Python 3.13.7, numpy 2.2.6, torch 2.8.0 (CUDA 12.8), matplotlib 3.10.7 |
| CPU runs and plots | Apple silicon laptop; Python 3.14.3, numpy 2.4.2, torch 2.11.0, matplotlib 3.10.8 |
| Precision | float64 throughout; for networks, cuDNN deterministic and TF32 off |

Random sketches and synthetic inputs are drawn with numpy from explicit integer seeds, so they do not
depend on the machine. Linear algebra on different hardware can differ at the level of rounding; the
networks' training can differ slightly more, which is why their results are reported over several
seeds.

## Output files

Each `results/<experiment>_<config>.npz` holds a `meta` entry, a JSON string recording the
configuration, the seed rule and the key format, plus one array per spectrum. Arrays have shape
(draws, number of eigenvalues), with eigenvalues in decreasing order. For example:

```python
import json, numpy as np
R = np.load("results/fig2_theorem_paper.npz")
meta = json.loads(str(R["meta"]))
lam = R["lam_y3_d10000"]          # (20, 10000): lambda_j(K_W) for f(y) = y^3, d = 10^4, 20 sketches
pop = R["pop_y3_d10000"]          # (10000,):   the population spectrum
```

Each experiment page lists its key format.
