---
title: Overview
---

# Power-law spectra of random features: numerical experiments

This site accompanies the paper
[*Power-Law Spectrum of the Random Feature Model*]({{ site.paper_url }}) (E. Paquette, K. L. Xiao,
Y. Zhu). It explains each numerical experiment in the paper: what is computed, how, what it shows, and
how to reproduce it from the code in [the repository]({{ site.repo_url }}). It is meant to be readable
without the paper at hand.

## The question

Natural data often have power-law covariance spectra. For CIFAR-10 images, for example, the $$j$$-th
eigenvalue of the covariance decays roughly like $$j^{-1.3}$$. Such exponents control how fast learning
algorithms make progress, and so how performance scales with data and compute. Machine-learning models
pass data through random or learned nonlinear maps. **Does the power law survive such a map, and with
what exponent?**

The paper answers this for the simplest such map, a layer of random features. Data
$$x \in \mathbb R^v$$ with covariance $$H$$, $$\lambda_j(H) \asymp j^{-\alpha}$$ and $$\alpha > 1$$, are sketched
by a Gaussian matrix $$W \in \mathbb R^{v\times d}$$, and a polynomial activation $$f$$ of degree $$p$$ is
applied entrywise. The main theorem states that the second-moment matrix of the features,

$$
K_W = \frac1d\, \mathbb E_x\big[f(W^\top x)\, f(W^\top x)^\top\big],
$$

satisfies, with high probability, for $$1 \le j \le c_1 d \log^{-(p+1)} d$$,

$$
\lambda_j(K_W) \asymp \Big(\frac{\log^{p-1}(j+1)}{j}\Big)^{\alpha}.
$$

So the exponent $$\alpha$$ of the data is inherited, up to a logarithmic correction that depends on the
degree. (It assumes $$v \ge d$$ if $$p = 1$$ and $$v \ge d^{1/p+\epsilon}$$ if $$p \ge 2$$.) The theorem is
really a comparison with an explicit deterministic spectrum; [Methods](methods.html#what-the-main-theorem-says)
gives the precise form.

## The experiments

| | Experiment | In the paper | Finding |
|---|---|---|---|
| 1 | [The main theorem](experiments/fig2-theorem.html) | Figure 2, left | For polynomial activations, the spectrum of $$K_W$$ matches the exactly computed population spectrum over a range of indices that grows with $$d$$. |
| 2 | [Other activations](experiments/fig2-activations.html) | Figure 2, right | ReLU and $$\tanh$$ give power laws close to $$j^{-\alpha}$$. The flat spectrum of $$y^2e^{-y^2}$$ is a $$1/d$$ floor from high Hermite degrees, predicted without fitting. |
| 3 | [Other input distributions](experiments/fig3-universality.html) | Figure 3 | Changing the input distribution, with the covariance fixed, leaves the decay unchanged but can change the head of the spectrum. |
| 4 | [Depth and training](experiments/sm5-depth.html) | Supplement, SM5 | Through the layers of MLPs, CNNs and ResNets on CIFAR-10, the spectra stay close to power laws, but the exponents drift, in ways that depend on the architecture, the initialization and training. |
| 5 | [$$\alpha < 1$$](experiments/sm6-alpha.html) | Supplement, SM6 | When the data's tail is not summable, the exponent still appears to be inherited, but a new effect appears: a staircase, with a step where each Hermite component of the activation runs out. |

The first figure of the paper (iterated linear sketches, and a $$\tanh$$ network on CIFAR-10) is not yet
on this site.

## How to read the pages

Each experiment page has the same structure: the question, a short answer, the figures with full
captions, the setup, how each quantity is computed, the results with numbers, and the commands that
reproduce everything. Three pages are shared:

- [Methods](methods.html): the model and notation, how $$K_W$$ and its population spectrum are computed
  exactly, when Monte Carlo is used, how exponents are fitted, and how uncertainty is reported.
- [Reproducing](reproduce.html): installation, the command for each figure, running times, the
  environment, the tests and the output formats.
- [Revision notes](revisions.html): what changed from the figures in the first version of the paper,
  and why.

## Principles

- **Exact where possible.** Most matrices are computed in closed form, with no sampling over the data.
  Where sampling is needed, every compared quantity is sampled the same way.
- **Compare, don't fit.** Fitted exponents are unreliable at computable sizes. So the experiments on
  random features compare spectra with exact references; exponents are fitted only for neural networks,
  always over a stated window.
- **Report the spread.** Every result is computed for many independent random draws, with explicit
  seeds, and the plots show the spread.

## Repository layout

| Path | Contents |
|---|---|
| `plrf/activations.py` | Activations and their Wick (Hermite) expansions |
| `plrf/exact.py`, `plrf/gaussian.py` | Exact $$K_W$$ for Gaussian inputs: polynomials; ReLU, Heaviside, $$y^2e^{-y^2}$$ in closed form; any activation by the Mehler series |
| `plrf/iid.py`, `plrf/inputs.py` | Exact $$K_W$$ for polynomial activations and inputs with independent non-Gaussian coordinates; input distributions; CIFAR-10 |
| `plrf/population.py` | The population spectrum (the $$d\to\infty$$ limit) for polynomials of degree $$\le 3$$ |
| `plrf/montecarlo.py`, `plrf/fit.py` | Monte Carlo estimates of $$K_W$$; power-law fits over an explicit window |
| `experiments/<name>/` | One folder per experiment: `run.py` computes, `plot.py` draws, `summarize.py` prints the tables |
| `tests/` | Checks of every exact formula |
| `docs/` | This site |
