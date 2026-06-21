"""Reproduce Chernozhukov-Hong (2003) Example 1 (censored median regression).

    python examples/run_censored_median.py --n 2000 --show
"""
import argparse

from lte import fit
from lte.examples import CensoredMedianRegression, simulate_censored_median
from lte.plotting import plot_objective_profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--burnin", type=int, default=10_000)
    p.add_argument("--keep", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=8981)
    p.add_argument("--outdir", default=".")
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()

    Y, Z, truth = simulate_censored_median(n=a.n, seed=a.seed)
    model = CensoredMedianRegression(Y, Z)
    result = fit(model, n_chains=a.chains, n_burnin=a.burnin,
                 n_keep=a.keep, seed=a.seed)
    result.summary(truth=truth.cpu())

    if not a.no_plots:
        plot_objective_profile(model, index=0, lo=-12, hi=0,
                               fixed=truth.cpu().tolist(),
                               path=f"{a.outdir}/censored_objective.png",
                               show=a.show)
        result.plot(truth=truth.cpu(),
                    path=f"{a.outdir}/censored_posteriors.png", show=a.show)


if __name__ == "__main__":
    main()
