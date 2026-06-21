"""Matplotlib helpers: a 1-D criterion profile and marginal posteriors.

matplotlib is imported lazily so the core library has no hard plotting
dependency. The backend is left to matplotlib (Agg automatically when
headless); pass show=True to display interactively.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, path, show):
    if path:
        fig.savefig(path, dpi=140)
    if show:
        _plt().show()
    return fig


def plot_objective_profile(model, index: int = 0, lo: float = -3.0,
                           hi: float = 3.0, num: int = 400,
                           fixed: Optional[Sequence[float]] = None,
                           path: Optional[str] = None, show: bool = False):
    """Profile the objective -L_n(theta) along one coordinate, others held
    at ``fixed`` (defaults to zeros)."""
    plt = _plt()
    k = model.n_params
    base = torch.zeros(num, k, dtype=model.dtype, device=model.device)
    if fixed is not None:
        base[:] = torch.as_tensor(fixed, dtype=model.dtype, device=model.device)
    grid = torch.linspace(lo, hi, num, dtype=model.dtype, device=model.device)
    base[:, index] = grid
    obj = (-model.criterion(base)).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(grid.cpu().numpy(), obj, color="#185FA5", lw=1.6)
    ax.set_xlabel(model.param_names[index])
    ax.set_ylabel("objective  $-L_n$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_posteriors(result, truth: Optional[Sequence[float]] = None,
                    bins: int = 40, path: Optional[str] = None,
                    show: bool = False):
    """Grid of marginal posterior histograms with mean and optional truth."""
    plt = _plt()
    flat = result.flat.numpy()
    mean = result.post_mean.numpy()
    k = flat.shape[1]
    ncol = 2 if k > 1 else 1
    nrow = (k + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.0 * nrow),
                             squeeze=False)
    for j, ax in enumerate(axes.ravel()):
        if j >= k:
            ax.axis("off")
            continue
        ax.hist(flat[:, j], bins=bins, color="#185FA5", alpha=0.85)
        ax.axvline(mean[j], color="#185FA5", lw=1.4,
                   label="post. mean")
        if truth is not None:
            ax.axvline(float(truth[j]), color="#993C1D", lw=1.2, ls="--",
                       label="truth")
        ax.set_xlabel(result.param_names[j])
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_quantile_coefficients(results, probs=(0.05, 0.95),
                               path: Optional[str] = None, show: bool = False):
    """Coefficient-vs-tau plot from a ``{tau: LTEResult}`` mapping (as returned
    by ``fit_quantile_process``): one panel per parameter, posterior mean with
    a credible band, and the population truth overlaid when available."""
    plt = _plt()
    taus = sorted(results)
    names = results[taus[0]].param_names
    k = len(names)
    means = np.array([results[t].post_mean.numpy() for t in taus])       # (T,k)
    qs = np.stack([results[t].quantiles(probs).numpy() for t in taus])    # (T,2,k)
    have_truth = all(results[t].info.get("truth") is not None for t in taus)
    if have_truth:
        truths = np.array([results[t].info["truth"].cpu().numpy() for t in taus])
    ncol = 2 if k > 1 else 1
    nrow = (k + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.0 * nrow),
                             squeeze=False)
    band = f"{int(probs[0] * 100)}-{int(probs[1] * 100)}% CI"
    for j, ax in enumerate(axes.ravel()):
        if j >= k:
            ax.axis("off")
            continue
        ax.fill_between(taus, qs[:, 0, j], qs[:, 1, j],
                        color="#185FA5", alpha=0.18, label=band)
        ax.plot(taus, means[:, j], "-o", color="#185FA5", label="post. mean")
        if have_truth:
            ax.plot(taus, truths[:, j], "--s", color="#993C1D", label="truth")
        ax.set_xlabel(r"$\tau$")
        ax.set_title(names[j])
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return _finish(fig, path, show)
