"""
Bayesian Fama-MacBeth
"""
from data import get_ffrf, get_ff5f, get_ff25p
from scipy.stats import invwishart, multivariate_normal
import pandas as pd
import numpy as np
from numpy.linalg import inv
from time import time
import matplotlib.pyplot as plt


class BFM:

    def __init__(self, assets, factors, n_draws=1000):
        # TODO Documentation
        self.assets = assets
        self.factors = factors
        self.n_draws = n_draws
        self.y = pd.concat([assets, factors], axis=1).dropna()

        self.t = self.y.shape[0]
        self.n = assets.shape[1]
        self.k = factors.shape[1]
        self.p = self.n + self.k

        self.mu_y = self.y.mean()
        self.Sigma_y = self.y.cov()

        # TODO organize these in DataFrames
        self.draws_mu_y, self.draws_Sigma_y = self._draw_mu_sigma()
        self.draws_betas, draws_lambdas = self._compute_beta_lambda()
        self.draws_lambdas = pd.DataFrame(
            data=draws_lambdas,
            columns=factors.columns
        )


    def _draw_mu_sigma(self):
        # TODO Documentation

        # Draws covariance from inverse wishart
        draws_sigma = invwishart.rvs(
            df=self.t - 1,
            scale=self.t * self.Sigma_y,
            size=self.n_draws,
        )

        # Draws mu from normal, conditional on previous covariances
        draws_mu = np.array(
            [
                multivariate_normal.rvs(
                    mean=self.mu_y,
                    cov=(1 / self.t) * cov,
                )
                for cov in draws_sigma
            ]
        )

        return draws_mu, draws_sigma

    def _compute_beta_lambda(self):
        # TODO Documentation
        # compute betas from sigma draws
        draws_beta = np.array(
            [
                cov[:self.n, -self.k:] @ inv(cov[-self.k:, -self.k:])
                for cov in self.draws_Sigma_y
            ]
        )

        # campute lambdas from betas and mus
        draws_lambda = np.array(
            [
                inv(b.T @ b) @ b.T @ mu[:self.n]
                for mu, b in zip(self.draws_mu_y, draws_beta)
            ]
        )

        return draws_beta, draws_lambda

    def plot_lambda(self):
        axes = self.draws_lambdas.hist(
            density=True,
            bins=int(np.sqrt(self.n_draws)),
            sharex=True,
            figsize=(10, 6)
        )

        for ax in axes.flatten():
            ax.axvline(0, color='tab:orange', lw=1)

        plt.tight_layout()
        plt.show()


class BFMGLS(BFM):
    pass


# TODO TESTING - ERASE THIS
# Fama-French Portfolios - Excess Returns
ports = get_ff25p()
rf = get_ffrf()
ports = ports.sub(rf, axis=0)
ports = ports.dropna()
ports.columns = [f"FF{(s - 1) * 5 + v}" for s, v in ports.columns]

facts = get_ff5f()

tic = time()
bfm = BFM(
    assets=ports,
    factors=facts,
    n_draws=1000,
)
bfm.plot_lambda()
print(time() - tic)
