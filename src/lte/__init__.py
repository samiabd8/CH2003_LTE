"""lte -- Laplace-type (quasi-Bayesian) estimation via ensemble MCMC.

Implements the Chernozhukov-Hong (2003) estimator: sample the quasi-posterior
proportional to exp(L_n(theta)) * prior(theta) with a vectorized ensemble of
random-walk Metropolis chains (GPU-aware), and report the posterior mean as
the point estimate.

Quickstart
----------
    from lte import fit
    from lte.examples import QuantileIVRegression, simulate_quantile_iv

    Y, Z, truth = simulate_quantile_iv()
    model = QuantileIVRegression(Y, Z)
    result = fit(model, n_chains=8, n_burnin=10_000, n_keep=10_000)
    result.summary(truth=truth)

Define your own model by subclassing ``LTEModel`` (implement ``criterion``)
or ``GMMModel`` (implement ``moment_contributions``).
"""
from .core import (
    LTEResult,
    effective_sample_size,
    fit,
    pick_device,
    run_ensemble_mh,
    split_rhat,
)
from .models import GMMModel, LTEModel, flat_log_prior

__all__ = [
    "fit",
    "run_ensemble_mh",
    "LTEResult",
    "split_rhat",
    "effective_sample_size",
    "pick_device",
    "LTEModel",
    "GMMModel",
    "flat_log_prior",
]

__version__ = "0.1.0"
