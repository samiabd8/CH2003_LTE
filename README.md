# Laplace-type (quasi-Bayesian) estimation via ensemble MCMC

A Python/PyTorch implementation of the Chernozhukov–Hong (2003) Laplace-type
estimator (LTE). Instead of optimizing a possibly non-smooth, non-convex
extremum criterion, you **sample** the quasi-posterior

```
p(theta) ∝ exp( L_n(theta) ) · prior(theta)
```

with a (vectorized) **ensemble of random-walk Metropolis chains**, and report the
posterior mean as the point estimate. This sidesteps optimization entirely and
yields finite-sample uncertainty from the draws.

Why an ensemble? A single Markov chain must run sequentially (each step depends on the previous one), so it cannot be sped up by a GPU. However, you can run many independent chains at the exact same time, as we do in this implementation. The scripts
select CUDA automatically when a GPU is visible and falls back to CPU otherwise (identical procedures).

## Install

```bash
pip install -e ".[dev]"      # torch + numpy, plus matplotlib/scipy/pytest
```

`scipy` is optional (used only for the Nelder-Mead posterior-mode);
`matplotlib` is optional (plots only).

## Quickstart

```python
from lte import fit
from lte.examples import QuantileIVRegression, simulate_quantile_iv

Y, Z, truth = simulate_quantile_iv()             # CH2003 Example 2 DGP (tau=0.5)
model = QuantileIVRegression(Y, Z, tau=0.5)
result = fit(model)                              # default chains / burn-in / keep
result.summary(truth=truth)                      # mean / sd / CI / ESS / R-hat
result.plot(truth=truth, show=True, path="posteriors.png")   # display AND save
```

`fit` uses intuitive defaults (`n_chains=8, n_burnin=10_000, n_keep=10_000`);
pass overrides only when you want them. 

### Across a sequence of quantiles

```python
from lte.examples import simulate_quantile_iv, fit_quantile_process
from lte.plotting import plot_quantile_coefficients

Y, Z, _ = simulate_quantile_iv()
results = fit_quantile_process(Y, Z, taus=[0.1, 0.5, 0.9])    # uses fit defaults
for tau, res in results.items():
    res.summary(truth=res.info["truth"])         # per-tau population truth
    res.plot(truth=res.info["truth"], show=True, path=f"post_{tau}.png")

plot_quantile_coefficients(results, show=True, path="qr_coefficients.png")
```

Or from the example scripts:

```bash
python examples/run_quantile_iv.py --tau 0.1 0.5 0.9 --show
python examples/run_censored_median.py --n 2000 --show
```

## Specifying your own model

Two entry points, depending on how your estimator is defined.

**1. Any extremum criterion** — subclass `LTEModel`, implement `criterion`,
the quasi-log-likelihood `L_n(theta)` to be maximized, batched over chains:

```python
import torch
from lte import LTEModel, fit

class CensoredMedian(LTEModel):
    def criterion(self, theta):                  # theta: (C, k) -> (C,)
        fit_ = torch.clamp(theta @ self.Z.t(), min=0.0)
        return -(self.Y.unsqueeze(0) - fit_).abs().sum(dim=1)

model = CensoredMedian(["b0", "b1", "b2", "b3"], Y=Y, Z=Z)
result = fit(model)
```

**2. A GMM model** — subclass `GMMModel`, implement `moment_contributions`,
the per-observation moment vectors `psi_i(theta)` shaped `(C, n, m)`. The
criterion `-(n/2) gbar' W gbar` with continuously-updated weighting is
constructed as usual:

```python
from lte import GMMModel, fit

class QuantileIV(GMMModel):
    def moment_contributions(self, theta):       # -> (C, n, m)
        q = theta @ self.Z.t()
        w = self.tau - (self.Y.unsqueeze(0) <= q).to(theta.dtype)
        return w.unsqueeze(-1) * self.Z.unsqueeze(0)

model = QuantileIV(["alpha", "b1", "b2", "b3"], Y=Y, Z=Z)
model.tau = 0.5
result = fit(model)
```

## API

| object | purpose |
|---|---|
| `fit(model, n_chains, n_burnin, n_keep, scale, spread, seed, refine_mode)` | run the ensemble sampler → `LTEResult` |
| `LTEModel` | base class; implement `criterion(theta)` |
| `GMMModel` | implement `moment_contributions(theta)`; CUE or fixed weighting |
| `LTEResult` | `.post_mean`, `.post_sd`, `.rhat`, `.ess`, `.quantiles()`, `.draws`, `.accept`, `.summary()`, `.plot()` |
| `split_rhat`, `effective_sample_size` | convergence / mixing diagnostics |
| `lte.plotting` | `plot_objective_profile`, `plot_posteriors`, `plot_quantile_coefficients` (all take `show=`) |
| `lte.examples` | `fit_quantile_process(Y, Z, taus, **fit_kwargs)`, `population_quantile_params(tau)` |

`summary()` reports the posterior mean and SD, a credible interval (`q5`/`q95`
by default), the effective sample size, and split-R-hat per parameter. 



## Notes for time-series / serially-dependent use

For i.i.d. cross-sections the CUE weighting `W = S⁻¹` with
`S = (1/n) Σ ψ_i ψ_i'` makes the quasi-posterior's credible sets valid
confidence sets (generalized information equality). With **serially dependent**
data, replace `S` with a HAC / long-run-variance estimate (Newey–West). 
Under inefficient weighting or misspecification, the
quasi-posterior covariance needs a sandwich correction (Müller, 2013).


## References 

- Chernozhukov, V. and Hong, H. (2003). An MCMC Approach to Classical
  Estimation. *Journal of Econometrics* 115(2), 293–346.
- Müller, U. K. (2013). Risk of Bayesian Inference in Misspecified Models, and
  the Sandwich Covariance Matrix. *Econometrica* 81(5), 1805–1849.
- Newey, W. and West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
- O'Hara, K. (2014). Chernozhukov-Hong [Computer software (R Code)]. GitHub. https://github.com/kthohr/Chernozhukov-Hong
