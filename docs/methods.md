---
title: Methods and conventions
---

# Methods and conventions

This page collects the model, the notation and the numerical methods that all the experiments share.
Each experiment page describes only what is specific to it and links back here.

- [The model](#the-model)
- [What the main theorem says](#what-the-main-theorem-says)
- [Computing $$K_W$$ exactly](#computing-k_w-exactly)
- [The population spectrum](#the-population-spectrum)
- [Hermite degrees and the $$1/d$$ floor](#hermite-degrees-and-the-1d-floor)
- [When Monte Carlo is used](#when-monte-carlo-is-used)
- [Fitting exponents](#fitting-exponents)
- [Uncertainty and seeds](#uncertainty-and-seeds)

## The model

| Symbol | Meaning |
|---|---|
| $$x \in \mathbb R^v$$ | an input, with covariance $$H$$ (diagonal without loss of generality; see below) |
| $$H = \operatorname{diag}(H_1, \dots, H_v)$$ | a power-law spectrum, $$H_j \propto j^{-\alpha}$$, with $$\alpha = 1.31$$ throughout (close to the exponent of CIFAR-10) |
| $$W \in \mathbb R^{v\times d}$$ | the sketch: i.i.d. $$N(0,1)$$ entries; its columns are $$w_1, \dots, w_d$$ |
| $$y = W^\top x \in \mathbb R^d$$ | the pre-activations; $$y_i = w_i^\top x$$ |
| $$f$$ | the activation, applied entrywise |
| $$K_W = \frac1d\,\mathbb E_x\big[f(y) f(y)^\top\big]$$ | the second-moment matrix of the random features (the object of the main theorem) |
| $$\mu = \mathbb E_x f(y)$$, $$\;C_W = K_W - \frac1d \mu\mu^\top$$ | the mean and the covariance of the features |
| $$\lambda_j(\cdot)$$ | eigenvalues in decreasing order |

**Second moment or covariance.** The main theorem is about $$K_W$$. The covariance $$C_W$$ differs from it
by a matrix of rank one, so by Cauchy interlacing $$\lambda_{j+1}(K_W) \le \lambda_j(C_W) \le \lambda_j(K_W)$$:
the two spectra are the same up to a shift of one index. Only $$\lambda_1$$ can differ substantially. For
an activation with nonzero mean (for example $$y^2$$ or ReLU), $$\lambda_1(K_W)$$ is dominated by the mean
term. The experiments on random features report $$K_W$$, and they store $$C_W$$ too. The experiments on
neural networks report the covariance, which is the usual object there.

**Scale.** For a fixed sketch, $$y_i$$ has variance $$s_i = w_i^\top H w_i$$, with mean
$$\mathbb E\, s_i = \operatorname{tr} H$$. For homogeneous activations (monomials, ReLU, Heaviside) the scale
only multiplies the spectrum by a constant. For $$\tanh$$ or $$y^2e^{-y^2}$$ it decides how nonlinear the
activation is. All the random-feature experiments therefore normalize $$\operatorname{tr} H = 1$$, so
that the pre-activations have mean square 1. (The page on [other activations](experiments/fig2-activations.html)
also uses a second, larger scale for comparison.) The exception is the page on
[$$\alpha < 1$$](experiments/sm6-alpha.html). There $$\operatorname{tr} H$$ grows with $$v$$, and that growth
is the point.

**Why diagonal $$H$$ loses nothing.** If $$x = U H^{1/2}\xi$$ for a rotation $$U$$, then
$$W^\top x = (U^\top W)^\top H^{1/2}\xi$$, and $$U^\top W$$ has the same law as $$W$$. So every
statement about the law of the spectrum of $$K_W$$ is the same for $$H$$ and for $$U H U^\top$$.

**The effective dimension is small.** With a power-law $$H$$ and $$\operatorname{tr} H = 1$$, the
effective rank $$(\operatorname{tr} H)^2 / \operatorname{tr} H^2$$ is about 10. So $$y_i$$ depends mostly on
a few coordinates of $$x$$, and two pre-activations are substantially correlated:
$$\rho_{ij} = G_{ij}/\sqrt{s_is_j}$$, with $$G = W^\top H W$$, has typical size 0.2 and reaches about 0.9
over all pairs. This matters twice below. The central limit theorem does not make $$y_i$$ Gaussian when
$$x$$ is not (see [Other input distributions](experiments/fig3-universality.html)), and the
Hermite series of the next section converges quickly but not trivially.

## What the main theorem says

Let $$f$$ be a polynomial of degree $$p$$, and suppose $$v \ge d$$ if $$p = 1$$ and
$$v \ge d^{1/p + \epsilon}$$ if $$p \ge 2$$. The main theorem of the paper (Theorem 2.3) states two things.

1. **Comparison with a deterministic spectrum.** With high probability over $$W$$,
   $$\lambda_j(K_W)$$ is within constant factors of the $$j$$-th largest product
   $$H_{s_1} H_{s_2} \cdots H_{s_p}$$ over index sets $$s_1 < s_2 < \dots < s_p$$, for
   $$1 \le j \le c_1 d \log^{-(p+1)} d$$. In the paper these products are the eigenvalues of a matrix
   written $$H^{\mathbf 1_p}$$.
2. **Lattice counting.** The $$j$$-th largest such product behaves like
   $$\big(\log^{p-1}(j+1)/j\big)^\alpha$$. So the exponent $$\alpha$$ of the data is inherited by the
   random features, up to a logarithmic correction that depends on the degree.

The experiments illustrate the first statement directly, by comparing $$\lambda_j(K_W)$$ with an exactly
computed deterministic spectrum. They do not try to read off the second one from fitted slopes; see
[Fitting exponents](#fitting-exponents) for why.

## Computing $$K_W$$ exactly

Most experiments compute $$K_W$$ in closed form, with no sampling over $$x$$. This removes Monte Carlo
error, which otherwise distorts the small eigenvalues, and it computes exactly the matrix the theorem is
about. Code: [`plrf/`]({{ site.repo_url }}/tree/main/plrf).

### Gaussian inputs: the Hermite expansion

For a fixed sketch, $$(y_i, y_j)$$ is a centered Gaussian pair with variances $$s_i, s_j$$ and correlation
$$\rho_{ij}$$. Expand the activation in Hermite polynomials at the scale of each neuron,

$$
f(\sqrt{s}\,z) = \sum_{k \ge 0} a_k(s)\, \frac{\mathrm{He}_k(z)}{\sqrt{k!}}, \qquad
a_k(s) = \mathbb E\Big[f(\sqrt s Z)\,\frac{\mathrm{He}_k(Z)}{\sqrt{k!}}\Big],\quad Z \sim N(0,1).
$$

Mehler's formula then gives every entry of $$K_W$$:

$$
d\,(K_W)_{ij} = \mathbb E[f(y_i) f(y_j)] = \sum_{k\ge 0} a_k(s_i)\, a_k(s_j)\, \rho_{ij}^{\,k}.
\qquad (\ast)
$$

Here $$E_k = \overline{a_k(s)^2}$$ (averaged over neurons) is the share of $$\mathbb E f(y)^2$$ carried by
Hermite degree $$k$$. Three cases:

- **Polynomials.** The sum $$(\ast)$$ is finite. Equivalently, in terms of Wick powers
  $${:}y^k{:} = s^{k/2}\mathrm{He}_k(y/\sqrt s)$$, which satisfy $$\mathbb E[{:}y_i^k{:}\,{:}y_j^l{:}] = \delta_{kl}\,k!\,G_{ij}^k$$:
  $$f(y) = \sum_k b_k(s)\,{:}y^k{:}$$ gives $$d\,(K_W)_{ij} = \sum_k k!\, b_k(s_i) b_k(s_j)\, G_{ij}^k$$.
  For example, $$y^2 = {:}y^2{:} + s$$ and $$y^3 = {:}y^3{:} + 3s\,y$$, so
  $$d\,K_W = 6\,G^{\circ 3} + 9\,(ss^\top)\circ G$$ for $$f(y) = y^3$$ (with $$\circ$$ the entrywise product).
  Code: `plrf/activations.py`, `plrf/exact.py`.
- **Closed forms.** These are Gaussian integrals:
  - ReLU (the arc-cosine kernel):
    $$\frac{\sqrt{s_is_j}}{2\pi}\big(\sin\theta + (\pi-\theta)\cos\theta\big)$$, with $$\cos\theta = \rho_{ij}$$;
  - Heaviside $$\mathbf 1\{y > 0\}$$: $$(\pi - \theta)/(2\pi)$$;
  - $$y^2 e^{-y^2}$$: $$D^{-1/2}(S_{11}S_{22} + 2S_{12}^2)$$. Here $$\Sigma$$ is the $$2\times 2$$ covariance of
    $$(y_i,y_j)$$, $$M = I + 2\Sigma$$, $$D = \det M$$ and $$S = \Sigma M^{-1}$$.

  Code: `plrf/gaussian.py`.
- **Anything else** ($$\tanh$$ here): the series $$(\ast)$$, truncated at the degree $$L$$ for which
  $$\max_{i\ne j}\lvert\rho_{ij}\rvert^L < 10^{-14}$$ ($$L \approx 210$$–$$340$$ for the sketches used). The
  diagonal, where $$\rho = 1$$, is set to the exact $$\mathbb E f(y_i)^2$$. The coefficients $$a_k(s)$$ are
  computed by Gauss–Hermite quadrature with 1000 nodes. The weights come from the Christoffel formula,
  because the usual eigenvector formula is accurate only to $$10^{-16}$$ in absolute terms, and the outer
  weights (down to $$10^{-700}$$) multiply very large Hermite values.

### Non-Gaussian inputs with independent coordinates

Let $$x = H^{1/2}\xi$$, where the $$\xi_k$$ are i.i.d., symmetric and standardized
($$\mathbb E\xi_k^2 = 1$$), with cumulants $$\kappa_r$$. The joint cumulants of a pair $$(y_i, y_j)$$ are
sums over coordinates:

$$
\operatorname{cum}(\underbrace{y_i,\dots,y_i}_{a},\underbrace{y_j,\dots,y_j}_{b})
= \kappa_{a+b} \sum_{k=1}^v H_k^{(a+b)/2}\, W_{ki}^{a}\, W_{kj}^{b}.
$$

Each of these is a $$d\times d$$ matrix computed by one matrix product. The joint moments
$$\mathbb E[y_i^a y_j^b]$$ follow by exponentiating the bivariate cumulant generating function, and $$K_W$$
for a polynomial $$f$$ follows from them. For $$f$$ of degree $$p$$ this needs cumulants of order up to
$$2p$$. That is exactly the condition for $$K_W$$ to exist; for Student-$$t_\nu$$ it is $$\nu > 2p$$.
Code: `plrf/iid.py`, `plrf/inputs.py`.

### Checks

| Test | What it checks | Agreement |
|---|---|---|
| `tests/test_exact.py` | polynomial $$K_W$$ (Gaussian inputs) against Monte Carlo over $$x$$, $$4\times10^5$$ samples | 0.5–1.4% (Frobenius), the size of the sampling error |
| `tests/test_gaussian.py` | the Mehler series against the closed forms, for ReLU, Heaviside, $$y^2e^{-y^2}$$, $$y^2$$ and $$y^3$$, at two scales | $$10^{-15}$$ |
| | $$\tanh$$ coefficients with 1000 against 2000 quadrature nodes | $$10^{-13}$$ |
| | all four non-polynomial activations against Monte Carlo | 0.2–0.5% |
| `tests/test_iid.py` | the cumulant formula, for Gaussian inputs, against `plrf/exact.py` | $$10^{-16}$$ |
| | against Monte Carlo for Rademacher, $$t_9$$ and $$t_{15}$$ inputs | 0.2–1.4% (the Gaussian formula is off by 8–216% on the same inputs) |
| `tests/test_population.py` | the population spectrum (next section) against brute force (monomials, Wick powers and general cubics), against large-$$d$$ $$K_W$$, and its truncation | $$10^{-13}$$; a few %; $$<10^{-3}$$ |

## The population spectrum

The columns $$w_i$$ of $$W$$ are i.i.d. $$N(0, I_v)$$, and $$(K_W)_{ij} = \kappa(w_i, w_j)/d$$ with the kernel
$$\kappa(w, w') = \mathbb E_x[f(w^\top x)\, f(w'^\top x)]$$. As $$d \to \infty$$ with $$v$$ fixed, the
eigenvalues of $$K_W$$ converge to those of the integral operator

$$
(Tg)(w) = \mathbb E_{w'}\big[\kappa(w, w')\, g(w')\big] \quad \text{on } L^2(N(0, I_v)).
$$

This deterministic spectrum is the *population spectrum*. It is the natural reference for
$$\lambda_j(K_W)$$: comparing the two shows directly where a sketch of size $$d$$ resolves the spectrum and
where it does not.

For a polynomial $$f$$ of degree $$p$$, $$\kappa$$ is a polynomial of degree at most $$p$$ in each argument, so
$$T$$ has finite rank and its spectrum can be computed exactly. The page on
[the main theorem](experiments/fig2-theorem.html#how-the-population-spectrum-is-computed) explains the
computation step by step; in summary: In the basis $$\phi$$ of
monomials $$w^n$$ of degree at most $$p$$, write $$\kappa(w, w') = \phi(w)^\top C\, \phi(w')$$. The nonzero eigenvalues
of $$T$$ are those of $$CM$$, where $$M = \mathbb E[\phi\phi^\top]$$ is the Gaussian Gram matrix of the
monomials. Both $$C$$ and $$M$$ are block diagonal by the set of coordinates with odd exponent:

- **Squarefree monomials** $$w_{s_1}\cdots w_{s_p}$$ ($$s_1 < \dots < s_p$$) are orthonormal and decouple.
  They contribute the eigenvalues $$(p!)^2 H_{s_1}\cdots H_{s_p}$$: the deterministic spectrum of the
  main theorem, up to the constant $$(p!)^2$$.
- **The other blocks** come from repeated indices and from the lower Wick components of $$f$$: one block
  with monomials $$1, w_a^2$$, and one block per index $$t$$ with monomials $$w_t, w_a^2 w_t, w_t^3$$.
  Because $$\mathbb E w_a^2 = 1$$ couples these monomials, the blocks are dense. For $$y^3$$ each $$t$$-block
  has one large eigenvalue, of order $$9(\operatorname{tr}H)^2 H_t$$. It comes from the linear component
  $$3s\,{:}y{:}$$ and dominates the head of the spectrum at accessible sizes.

These blocks are solved exactly, with the coordinates beyond the first 150 entering through one
aggregated direction. Raising 150 to 400 changes the top 2000 eigenvalues by less than
$$10^{-3}$$. Code: `plrf/population.py` (degrees $$p \le 3$$, any polynomial).

## Hermite degrees and the $$1/d$$ floor

By $$(\ast)$$, $$K_W = \sum_k K^{(k)}$$ with $$d\,K^{(k)}_{ij} = a_k(s_i)\,a_k(s_j)\,\rho_{ij}^{\,k}$$. The
degree-$$k$$ part behaves like the random-feature matrix of the Wick power $${:}y^k{:}$$. Its population
spectrum decays like $$\big(\log^{k-1}(j+1)/j\big)^\alpha$$, but with a constant that shrinks fast with
$$k$$, and with a very large number of comparable eigenvalues.

A sketch of size $$d$$ cannot resolve that. For large $$k$$ the off-diagonal entries $$\rho_{ij}^{\,k}$$ are
negligible, so $$K^{(k)} \approx \frac1d \operatorname{diag}\big(a_k(s_i)^2\big)$$: white noise of height
about $$E_k/d$$. An activation that puts much of its variance into high degrees therefore shows a flat
stretch in its spectrum, a *floor*, at height about $$\big(\sum_{k\,\text{high}} E_k\big)/d$$.

The diagnostic is to plot $$d\,\lambda_j$$ for several $$d$$. A floor collapses onto one curve, while the
resolved part of the spectrum, which does not depend on $$d$$, moves up with $$d$$. See
[Other activations](experiments/fig2-activations.html).

## When Monte Carlo is used

Monte Carlo over $$x$$ is used only where no closed form exists:

- non-polynomial activations with non-Gaussian inputs;
- real data: $$K_W$$ is then the average over the dataset, exact for its empirical distribution;
- trained networks.

In comparisons between input distributions, every distribution is estimated with the same number of
samples, and the sketch is shared, so the finite-sample distortions largely cancel in the ratios.
Wherever a closed form exists for the Gaussian case, the Monte Carlo estimate is checked against it.
Code: `plrf/montecarlo.py`.

## Fitting exponents

Fitted power-law exponents are unreliable at the sizes that can be computed, for three reasons.

1. **The lowest Hermite components dominate the head.** In $$y^3 = {:}y^3{:} + 3s\,{:}y{:}$$, the linear
   part gives population eigenvalues of order $$9(\operatorname{tr}H)^2 H_t$$, and the cubic part gives
   $$36\,H_{s_1}H_{s_2}H_{s_3}$$. At $$\operatorname{tr}H = 1$$ the linear part supplies 88 of the top 100
   population eigenvalues and half of the top $$10^4$$. These decay like $$j^{-\alpha}$$, so the
   $$\log^{p-1}$$ correction appears only much further out.
2. **The theorem's range is short.** It is $$j \le c_1 d \log^{-(p+1)} d$$, and $$\log^4(10^4) \approx 7000$$.
3. **Finite-$$d$$ floors and the edge of the spectrum** flatten or bend the tail.

So the experiments on random features compare with exact references (the population spectrum, or a
Gaussian with the same covariance) instead of fitting.

Where an exponent is reported anyway (for neural networks, which the theory does not cover), it is
the slope of an ordinary least-squares fit of $$\log_{10}\lambda_j$$ on $$\log_{10} j$$ over an explicit window
of indices. The default window is $$1 \le j \le 100$$, and every reported slope states its window. Each
such experiment also reports how the slope changes over the windows 10–100, 1–30 and 10–300. Code:
`plrf/fit.py`, the only fitting routine used.

## Uncertainty and seeds

- **Independent draws.** Every random-feature result is computed for several independent sketches $$W$$
  (20 in the `paper` configurations). Every network result is computed for several initializations
  (5 random, 3 trained). Plots show the median over draws with the 10–90% band (random features), or the
  median with the range (networks).
- **Paired ratios.** A ratio of two spectra is computed draw by draw, with the same $$W$$ in numerator and
  denominator, and then summarized.
- **Seeds.** Every seed is an explicit integer, or a list of integers passed to
  `numpy.random.default_rng`. Sketches and synthetic inputs are always drawn with numpy, so results do
  not depend on whether the linear algebra runs on a CPU or a GPU. Each experiment page states its seed
  rule.
- **Determinism.** The computations are in float64. Network experiments set cuDNN to deterministic and
  turn TF32 off. Bitwise identical results across different GPU models are still not guaranteed;
  differences show up at the level of rounding.
