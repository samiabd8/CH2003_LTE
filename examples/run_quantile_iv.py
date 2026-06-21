"""Quantile-IV LTE across one or more quantiles (CH2003 Example 2).

Sampler settings use the defaults of ``fit`` (n_chains=8, n_burnin=10_000,
n_keep=10_000); edit the ``fit_quantile_process`` call to change them.

    python examples/run_quantile_iv.py                       # tau = 0.1,0.5,0.9
    python examples/run_quantile_iv.py --tau 0.25 0.75 --show
"""
import argparse

from lte.examples import fit_quantile_process, simulate_quantile_iv
from lte.plotting import plot_quantile_coefficients


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=8981)
    p.add_argument("--outdir", default=".")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()

    Y, Z, _ = simulate_quantile_iv(n=a.n, seed=a.seed)
    results = fit_quantile_process(Y, Z, taus=a.tau, seed=a.seed)  # fit defaults

    for tau, res in results.items():
        print(f"\n=== tau = {tau} ===")
        res.summary(truth=res.info.get("truth"))
        res.plot(truth=res.info.get("truth"), show=a.show,
                 path=f"{a.outdir}/quantile_tau{tau}_posteriors.png")

    if len(results) > 1:
        plot_quantile_coefficients(
            results, show=a.show,
            path=f"{a.outdir}/quantile_coefficients.png")


if __name__ == "__main__":
    main()
