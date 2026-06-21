"""Model-agnostic core: device selection, the vectorized ensemble
random-walk Metropolis sampler, split-R-hat, and a results container.

The sampler knows nothing about a particular model. It only needs an
object exposing ``log_posterior(theta) -> (C,)`` (batched over C chains)
and ``init(refine) -> (center (k,), cov (k, k))``. See ``lte.models``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch


def pick_device() -> torch.device:
    """CUDA if a GPU is visible, otherwise CPU. The same code runs on both."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_rhat(draws: torch.Tensor) -> torch.Tensor:
    """Gelman split-R-hat per parameter.

    draws: (n_keep, C, k) -> (k,). Each chain is split in half so a single
    long chain still yields a meaningful statistic.
    """
    n_keep, C, k = draws.shape
    m = n_keep // 2
    x = (draws[: 2 * m]
         .reshape(2, m, C, k)
         .permute(0, 2, 1, 3)
         .reshape(2 * C, m, k))
    means = x.mean(dim=1)                       # (2C, k)
    vars = x.var(dim=1, unbiased=True)          # (2C, k)
    W = vars.mean(dim=0)                         # within-chain
    B = m * means.var(dim=0, unbiased=True)      # between-chain
    var_hat = (m - 1) / m * W + B / m
    return torch.sqrt(var_hat / W)


def effective_sample_size(draws: torch.Tensor) -> torch.Tensor:
    """Per-parameter effective sample size, summed over chains.

    draws: (n_keep, C, k) -> (k,). Uses an FFT autocorrelation with Geyer's
    initial positive-sequence truncation (integrated autocorrelation time)
    per chain.
    """
    x = draws.detach().cpu().numpy()
    N, C, K = x.shape
    out = np.empty(K)
    for kk in range(K):
        ess_k = 0.0
        for c in range(C):
            y = x[:, c, kk] - x[:, c, kk].mean()
            v = float(np.dot(y, y) / N)
            if v <= 0:
                ess_k += N
                continue
            f = np.fft.rfft(y, n=2 * N)
            ac = np.fft.irfft(f * np.conj(f), n=2 * N)[:N].real / (N * v)
            s, t = 0.0, 1
            while t + 1 < N:                      # Geyer positive sequence
                pair = ac[t] + ac[t + 1]
                if pair < 0:
                    break
                s += pair
                t += 2
            ess_k += N / max(1.0 + 2.0 * s, 1e-8)
        out[kk] = ess_k
    return torch.tensor(out)


@dataclass
class LTEResult:
    """Container for posterior draws plus convenience diagnostics."""
    draws: torch.Tensor                  # (n_keep, C, k), on CPU
    param_names: Sequence[str]
    accept: torch.Tensor                 # (C,) per-chain acceptance rate
    info: dict = field(default_factory=dict)

    @property
    def flat(self) -> torch.Tensor:
        return self.draws.reshape(-1, self.draws.shape[-1])

    @property
    def post_mean(self) -> torch.Tensor:
        return self.flat.mean(dim=0)

    @property
    def post_sd(self) -> torch.Tensor:
        return self.flat.std(dim=0)

    @property
    def rhat(self) -> torch.Tensor:
        return split_rhat(self.draws)

    def quantiles(self, probs=(0.05, 0.95)) -> torch.Tensor:
        q = torch.tensor(probs, dtype=self.flat.dtype)
        return torch.quantile(self.flat, q, dim=0)        # (len(probs), k)

    @property
    def ess(self) -> torch.Tensor:
        return effective_sample_size(self.draws)

    def summary(self, truth: Optional[Sequence[float]] = None,
                probs=(0.05, 0.95)) -> str:
        mean, sd, rhat, ess = self.post_mean, self.post_sd, self.rhat, self.ess
        lo, hi = self.quantiles(probs)
        n_keep, C, _ = self.draws.shape
        plo, phi = int(probs[0] * 100), int(probs[1] * 100)
        cols = (f"{'param':>8} {'post.mean':>11} {'post.sd':>9} "
                f"{f'q{plo}':>9} {f'q{phi}':>9} {'ess':>8} {'rhat':>7}")
        if truth is not None:
            cols += f" {'truth':>8}"
        lines = [
            f"device: {self.info.get('device', '?')}   chains: {C}   "
            f"kept/chain: {n_keep}   draws: {n_keep * C}   "
            f"mean accept: {self.accept.mean():.3f}",
            cols,
        ]
        for j, nm in enumerate(self.param_names):
            row = (f"{nm:>8} {mean[j]:>+11.4f} {sd[j]:>9.4f} "
                   f"{lo[j]:>+9.4f} {hi[j]:>+9.4f} {ess[j]:>8.0f} "
                   f"{rhat[j]:>7.4f}")
            if truth is not None:
                row += f" {float(truth[j]):>+8.3f}"
            lines.append(row)
        out = "\n".join(lines)
        print(out)
        return out

    def plot(self, truth: Optional[Sequence[float]] = None,
             path: Optional[str] = None, show: bool = False):
        """Marginal posterior histograms (convenience wrapper)."""
        from .plotting import plot_posteriors
        return plot_posteriors(self, truth=truth, path=path, show=show)


def run_ensemble_mh(model,
                    n_chains: int = 8,
                    n_burnin: int = 10_000,
                    n_keep: int = 10_000,
                    scale: float = 1.0,
                    spread: float = 2.0,
                    seed: int = 0,
                    refine_mode: bool = True,
                    store: str = "device") -> LTEResult:
    """Run ``n_chains`` independent RWM chains in lockstep.

    A single RWM chain is sequential; parallelism is across chains. The
    proposal, the (model-defined) batched log-posterior evaluation, and the
    accept/reject are all vectorized over the leading chain dimension, which
    is what a GPU accelerates. Over-dispersed starts make split-R-hat
    meaningful.
    """
    device, dtype, k = model.device, model.dtype, model.n_params
    g = torch.Generator(device=device).manual_seed(seed)

    center, cov = model.init(refine=refine_mode)
    L = torch.linalg.cholesky(scale * cov)               # proposal factor

    z0 = torch.randn(n_chains, k, generator=g, device=device, dtype=dtype)
    theta = center.unsqueeze(0) + spread * (z0 @ L.t())
    lp = model.log_posterior(theta)                      # (C,)

    def step(theta, lp):
        noise = torch.randn(n_chains, k, generator=g, device=device, dtype=dtype)
        prop = theta + noise @ L.t()
        lp_prop = model.log_posterior(prop)
        logu = torch.log(torch.rand(n_chains, generator=g,
                                    device=device, dtype=dtype))
        acc = logu < (lp_prop - lp)
        theta = torch.where(acc.unsqueeze(1), prop, theta)
        lp = torch.where(acc, lp_prop, lp)
        return theta, lp, acc

    for _ in range(n_burnin):
        theta, lp, _ = step(theta, lp)

    keep_device = device if store == "device" else torch.device("cpu")
    draws = torch.empty(n_keep, n_chains, k, dtype=dtype, device=keep_device)
    accept = torch.zeros(n_chains, device=device)
    for i in range(n_keep):
        theta, lp, acc = step(theta, lp)
        accept += acc.to(accept.dtype)
        draws[i] = theta.to(keep_device)

    info = dict(device=str(device), center=center.detach().cpu(),
                cov=cov.detach().cpu())
    return LTEResult(draws.cpu(), list(model.param_names),
                     (accept / n_keep).cpu(), info)


# Friendly alias.
fit = run_ensemble_mh
