# Power-law spectrum of the random feature model: experiments

Reproducible numerical experiments for
[*Power-Law Spectrum of the Random Feature Model*](https://arxiv.org/abs/2603.14578)
(E. Paquette, K. L. Xiao, Y. Zhu).

The [documentation site](https://mlexpos.github.io/power-law-activations-are-preserved/) goes through each experiment: what it
computes, what it illustrates, and how to reproduce it. Start with `docs/index.md`.

```bash
git clone https://github.com/mlexpos/power-law-activations-are-preserved.git
cd power-law-activations-are-preserved
pip install -r requirements.txt
python -m tests.test_exact
python -m tests.test_population
python -m tests.test_iid
python -m tests.test_gaussian
python -m experiments.fig2_theorem.run --config quick     # or --config paper (GPU recommended)
python -m experiments.fig2_theorem.plot --config quick
python -m experiments.fig2_activations.run --config quick
python -m experiments.fig2_activations.plot --config quick
python -m experiments.fig3_universality.run --config quick   # CIFAR-10 is downloaded to data/ if missing
python -m experiments.fig3_universality.plot --config quick
python -m experiments.sm5_depth.run --config quick --part all
python -m experiments.sm5_depth.plot --config quick
```

| Path | Contents |
|---|---|
| `plrf/` | Library: activations, exact second-moment matrices, population spectra |
| `experiments/` | One folder per figure: `run.py` computes, `plot.py` draws |
| `tests/` | Checks of the closed-form formulas and of the population spectrum |
| `data/` | CIFAR-10 (downloaded; not tracked) |
| `results/`, `figures/` | Outputs (results are not tracked; rerun to regenerate) |
| `docs/` | GitHub Pages site |
