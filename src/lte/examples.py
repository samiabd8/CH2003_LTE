"""The two Chernozhukov-Hong (2003) reference models, ported from Keith
O'Hara's R code, plus their data-generating processes. Use them directly or
as templates for your own model.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .core import fit, pick_device
from .models import GMMModel, LTEModel


# ---------------------------------------------------------------------------
# Example 1: censored median (CLAD) regression
#   criterion(theta) = -sum_i | y_i - max(0, z_i' theta) |
# ---------------------------------------------------------------------------
class CensoredMedianRegression(LTEModel):
    def __init__(self, Y, Z, **kwargs):
        names = [f"beta{j}" for j in range(Z.shape[1])]
        super().__init__(names, Y=Y, Z=Z, **kwargs)

    def criterion(self, theta: torch.Tensor) -> torch.Tensor:
        fit = torch.clamp(theta @ self.Z.t(), min=0.0)   # (C, n) max(0, Z theta)
        resid = self.Y.unsqueeze(0) - fit                # (C, n)
        return -resid.abs().sum(dim=1)                   # (C,)


def simulate_censored_median(n=200, seed=8981, device=None,
                             dtype=torch.float64,
                             beta0=(-6.0, 3.0, 3.0, 3.0)
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = device or pick_device()
    g = torch.Generator(device=device).manual_seed(seed)
    b0 = torch.tensor(beta0, device=device, dtype=dtype)
    X = torch.randn(n, 3, generator=g, device=device, dtype=dtype)
    Z = torch.cat([torch.ones(n, 1, device=device, dtype=dtype), X], dim=1)
    eps = torch.randn(n, generator=g, device=device, dtype=dtype)
    u = (X[:, 0] ** 2) * eps                             # heteroskedastic
    Ys = Z @ b0 + u
    Y = torch.clamp(Ys, min=0.0)                         # censoring at 0
    return Y, Z, b0


# ---------------------------------------------------------------------------
# Example 2: quantile IV regression (median, tau = 0.5)
#   psi_i(theta) = (tau - 1{y_i <= z_i' theta}) z_i
# ---------------------------------------------------------------------------
class QuantileIVRegression(GMMModel):
    def __init__(self, Y, Z, tau=0.5, **kwargs):
        names = ["alpha"] + [f"beta{j}" for j in range(1, Z.shape[1])]
        super().__init__(names, Y=Y, Z=Z, **kwargs)
        self.tau = tau

    def moment_contributions(self, theta: torch.Tensor) -> torch.Tensor:
        q = theta @ self.Z.t()                           # (C, n)
        ind = (self.Y.unsqueeze(0) <= q).to(theta.dtype)
        w = self.tau - ind                               # (C, n)
        return w.unsqueeze(-1) * self.Z.unsqueeze(0)     # (C, n, k)


def simulate_quantile_iv(n=200, seed=8981, device=None, dtype=torch.float64
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = device or pick_device()
    g = torch.Generator(device=device).manual_seed(seed)
    b0 = torch.zeros(4, device=device, dtype=dtype)       # alpha0=0, beta0=0_3
    D = torch.randn(n, 3, generator=g, device=device, dtype=dtype)
    Z = torch.cat([torch.ones(n, 1, device=device, dtype=dtype), D], dim=1)
    eps = torch.randn(n, generator=g, device=device, dtype=dtype)
    sigma = (1.0 + D.sum(dim=1)) / 5.0                    # heteroskedastic
    Y = sigma * eps                                       # median(Y|D)=0
    return Y, Z, b0


def population_quantile_params(tau, n=300_000, seed=0, device=None,
                               dtype=torch.float64) -> Optional[torch.Tensor]:
    """Large-sample linear quantile-regression truth for the Example-2 DGP.

    The DGP is heteroskedastic, so the true linear-QR coefficients are zero
    only at tau=0.5; for other tau they are nonzero. Computed by minimizing
    the pinball loss on a large simulated sample. Returns None if SciPy is
    unavailable. (At tau=0.5 returns ~0 without the optimization.)
    """
    device = device or pick_device()
    if abs(tau - 0.5) < 1e-12:
        return torch.zeros(4, device=device, dtype=dtype)
    try:
        from scipy.optimize import minimize
        from scipy.stats import norm
    except Exception:
        return None
    Y, Z, _ = simulate_quantile_iv(n=n, seed=seed, device="cpu", dtype=dtype)
    Yn, Zn = Y.numpy(), Z.numpy()

    def loss(b):
        u = Yn - Zn @ b
        return float((u * (tau - (u < 0))).sum())

    b0 = [float(norm.ppf(tau)) / 5.0] * 4
    res = minimize(loss, b0, method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-7, maxiter=40_000))
    return torch.as_tensor(res.x, device=device, dtype=dtype)


def fit_quantile_process(Y, Z, taus=(0.1, 0.5, 0.9), truth=True, **fit_kwargs):
    """Fit the quantile-IV LTE at each tau and return an ordered dict
    ``{tau: LTEResult}``.

    Extra keyword arguments pass through to ``fit`` -- omit them to use its
    defaults (n_chains=8, n_burnin=10_000, n_keep=10_000, ...). Each result
    carries ``info['tau']`` and, when ``truth`` is True, ``info['truth']``
    (the large-sample population QR coefficients for that tau).
    """
    results = {}
    for tau in taus:
        model = QuantileIVRegression(Y, Z, tau=tau)
        res = fit(model, **fit_kwargs)
        res.info["tau"] = tau
        res.info["truth"] = population_quantile_params(tau) if truth else None
        results[tau] = res
    return results
