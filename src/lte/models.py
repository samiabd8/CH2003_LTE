"""Model base classes for the Laplace-type / quasi-Bayesian estimator.

Two ways to specify a model:

1. Subclass ``LTEModel`` and implement ``criterion(theta) -> (C,)``, the
   CH2003 quasi-log-likelihood L_n(theta) (higher is better). This covers
   any extremum estimator -- M-estimators, GMM, MLE, ...

2. Subclass ``GMMModel`` and implement
   ``moment_contributions(theta) -> (C, n, m)``, the per-observation moment
   vectors psi_i(theta). The GMM criterion -(n/2) gbar' W gbar with a
   continuously-updated (CUE) weighting is built for you.

Both default to a flat U(prior_lo, prior_hi) prior (override ``log_prior``
for anything else) and, when ``self.Y`` / ``self.Z`` are present, to an
OLS-based starting point and proposal scale (override ``_default_init``
otherwise).
"""
from __future__ import annotations

import abc
from typing import Optional, Sequence, Tuple

import torch

from .core import pick_device


def flat_log_prior(theta: torch.Tensor, lo, hi) -> torch.Tensor:
    """0 inside the box [lo, hi]^k, -inf outside. lo/hi scalar or (k,)."""
    inside = ((theta >= lo) & (theta <= hi)).all(dim=-1)
    out = torch.zeros(theta.shape[0], dtype=theta.dtype, device=theta.device)
    out[~inside] = float("-inf")
    return out


class LTEModel(abc.ABC):
    """Base class. Implement ``criterion``; optionally set ``Y``/``Z``."""

    def __init__(self,
                 param_names: Sequence[str],
                 Y: Optional[torch.Tensor] = None,
                 Z: Optional[torch.Tensor] = None,
                 prior_lo=-10.0,
                 prior_hi=10.0,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float64):
        self.param_names = list(param_names)
        self.device = device or pick_device()
        self.dtype = dtype
        self.prior_lo = prior_lo
        self.prior_hi = prior_hi
        self.Y = None if Y is None else Y.to(self.device, dtype)
        self.Z = None if Z is None else Z.to(self.device, dtype)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    # --- the only thing a subclass must provide -------------------------
    @abc.abstractmethod
    def criterion(self, theta: torch.Tensor) -> torch.Tensor:
        """L_n(theta), batched: theta (C, k) -> (C,). Higher is better."""

    # --- prior / posterior ---------------------------------------------
    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        return flat_log_prior(theta, self.prior_lo, self.prior_hi)

    def log_posterior(self, theta: torch.Tensor) -> torch.Tensor:
        lp = self.criterion(theta) + self.log_prior(theta)
        return torch.nan_to_num(lp, nan=float("-inf"))

    # --- starting point and proposal scale ------------------------------
    def init(self, refine: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        center, cov = self._default_init()
        if refine:
            refined = self._refine_mode(center)
            if refined is not None:
                center = refined
        return center, cov

    def _default_init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.Y is None or self.Z is None:
            raise NotImplementedError(
                "No Y/Z available -- override _default_init() to supply a "
                "starting point (center) and proposal covariance.")
        Y, Z = self.Y, self.Z
        theta = torch.linalg.lstsq(Z, Y.unsqueeze(1)).solution.squeeze(1)
        resid = Y - Z @ theta
        sigsq = (resid ** 2).mean()
        cov = torch.linalg.inv(Z.t() @ Z) * sigsq ** 2
        return theta, cov

    def _refine_mode(self, center: torch.Tensor) -> Optional[torch.Tensor]:
        """Derivative-free (Nelder-Mead) posterior-mode polish, mirroring the
        R optim() step. Returns None if SciPy is unavailable."""
        try:
            from scipy.optimize import minimize
        except Exception:
            return None

        def neg_lp(x):
            t = torch.as_tensor(x, dtype=self.dtype,
                                device=self.device).unsqueeze(0)
            return float(-self.log_posterior(t).item())

        res = minimize(neg_lp, center.detach().cpu().numpy(),
                       method="Nelder-Mead",
                       options=dict(xatol=1e-6, fatol=1e-6, maxiter=20000))
        return torch.as_tensor(res.x, dtype=self.dtype, device=self.device)


class GMMModel(LTEModel):
    """GMM specialization. Implement ``moment_contributions``.

    weighting: 'cue' recomputes W(theta) = S(theta)^{-1} every evaluation
    (continuously updated). Pass a fixed (m, m) tensor for a frozen /
    two-step weighting. ``ridge`` stabilizes the inverse.
    """

    def __init__(self, *args, weighting="cue", ridge=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighting = weighting
        self.ridge = ridge

    @abc.abstractmethod
    def moment_contributions(self, theta: torch.Tensor) -> torch.Tensor:
        """psi_i(theta), batched: theta (C, k) -> (C, n, m)."""

    def criterion(self, theta: torch.Tensor) -> torch.Tensor:
        psi = self.moment_contributions(theta)        # (C, n, m)
        C, n, m = psi.shape
        gbar = psi.mean(dim=1)                         # (C, m)
        if isinstance(self.weighting, torch.Tensor):
            W = self.weighting.to(theta.device, theta.dtype).expand(C, m, m)
        else:                                          # CUE
            S = psi.transpose(1, 2) @ psi / n          # (C, m, m)
            if self.ridge:
                S = S + self.ridge * torch.eye(m, dtype=psi.dtype,
                                               device=psi.device)
            W = torch.linalg.inv(S)
        gWg = torch.einsum("ci,cij,cj->c", gbar, W, gbar)
        return -0.5 * n * gWg
