"""Smoke tests for the sampler and the two reference models.

Note on the censored example: at n=200 with this heteroskedastic,
heavily-censored DGP, very few observations are uncensored, so the
finite-sample CLAD/LTE estimate has large variance and need not sit near
the truth (and the U(-10,10) prior can bind). We therefore test
truth-recovery at a well-identified sample size, and test *sampler
correctness* (mixing + concentration on the criterion's preferred region)
at n=200.
"""
import torch

from lte import fit
from lte.examples import (
    CensoredMedianRegression,
    QuantileIVRegression,
    simulate_censored_median,
    simulate_quantile_iv,
)


def test_quantile_iv_recovers_zero():
    Y, Z, truth = simulate_quantile_iv(seed=8981)
    res = fit(QuantileIVRegression(Y, Z), n_chains=8, n_burnin=2_000,
              n_keep=4_000, seed=1)
    assert torch.allclose(res.post_mean, truth.cpu(), atol=0.1)
    assert (res.rhat < 1.1).all()
    assert 0.1 < res.accept.mean() < 0.8


def test_censored_median_recovers_truth_large_n():
    Y, Z, truth = simulate_censored_median(n=2_000, seed=8981)
    res = fit(CensoredMedianRegression(Y, Z), n_chains=8, n_burnin=4_000,
              n_keep=6_000, seed=2)
    assert torch.allclose(res.post_mean, truth.cpu(), atol=0.5)
    assert (res.rhat < 1.1).all()


def test_censored_sampler_concentrates_on_optimum():
    Y, Z, _ = simulate_censored_median(n=200, seed=8981)
    m = CensoredMedianRegression(Y, Z)
    res = fit(m, n_chains=8, n_burnin=4_000, n_keep=6_000, seed=2)
    assert (res.rhat < 1.2).all()   # near-boundary posterior mixes less cleanly
    mode = m._refine_mode(m._default_init()[0])
    crit_mean = float(m.criterion(res.post_mean.unsqueeze(0).to(m.device)))
    crit_mode = float(m.criterion(mode.unsqueeze(0)))
    assert crit_mode - crit_mean < 5.0


def test_custom_model_via_subclass():
    from lte import GMMModel

    class LinearMeanGMM(GMMModel):
        def moment_contributions(self, theta):
            resid = self.Y.unsqueeze(0) - theta @ self.Z.t()
            return resid.unsqueeze(-1) * self.Z.unsqueeze(0)

    g = torch.Generator().manual_seed(0)
    Z = torch.cat([torch.ones(300, 1),
                   torch.randn(300, 2, generator=g)], 1).double()
    b = torch.tensor([1.0, -2.0, 0.5]).double()
    Y = Z @ b + 0.5 * torch.randn(300, generator=g).double()
    res = fit(LinearMeanGMM(["b0", "b1", "b2"], Y=Y, Z=Z),
              n_chains=8, n_burnin=2_000, n_keep=4_000, seed=3)
    assert torch.allclose(res.post_mean, b, atol=0.15)


def test_quantile_process_and_plot(tmp_path):
    from lte.examples import fit_quantile_process, simulate_quantile_iv
    from lte.plotting import plot_quantile_coefficients

    Y, Z, _ = simulate_quantile_iv(seed=8981)
    results = fit_quantile_process(Y, Z, taus=[0.25, 0.5, 0.75],
                                   n_burnin=1_000, n_keep=2_000, seed=1)
    assert set(results) == {0.25, 0.5, 0.75}
    # median truth is ~0; tails are nonzero and opposite-signed intercepts
    assert abs(float(results[0.5].info["truth"][0])) < 1e-6
    a_lo = float(results[0.25].info["truth"][0])
    a_hi = float(results[0.75].info["truth"][0])
    assert a_lo < 0 < a_hi
    out = tmp_path / "coef.png"
    plot_quantile_coefficients(results, path=str(out))
    assert out.exists()
