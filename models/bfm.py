"""
Bayesian Fama-MacBeth
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import get_ffrf, get_ff5f, get_ff25p
from numpy.linalg import inv, svd
from scipy.stats import invwishart, multivariate_normal
from time import time

# TODO add the R2 for each case


class BFM:

    def __init__(self, assets, factors, n_draws=1000):
        # TODO Documentation
        self.assets = assets
        self.factors = factors
        self.n_draws = n_draws
        self.y = pd.concat([assets, factors], axis=1).dropna()

        self.t = self.y.shape[0]
        self.n = assets.shape[1]  # Number of assets
        self.k = factors.shape[1]  # Number of factors

        self.mu_y = self.y.mean()
        self.Sigma_y = self.y.cov()

        self.draws_mu_y, self.draws_Sigma_y = self._draw_mu_sigma()
        self.draws_betas = self._compute_betas()

        draws_lambdas = self._compute_lambdas()
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

    def _compute_betas(self):
        # TODO Documentation
        draws_beta = np.array(
            [
                cov[:self.n, -self.k:] @ inv(cov[-self.k:, -self.k:])
                for cov in self.draws_Sigma_y
            ]
        )
        return draws_beta

    def _compute_lambdas(self):
        # TODO Documentation
        # campute lambdas from betas and mus
        draws_lambda = np.array(
            [
                inv(b.T @ b) @ b.T @ mu[:self.n]
                for mu, b in zip(self.draws_mu_y, self.draws_betas)
            ]
        )

        return draws_lambda

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

    def ci_table_lambda(self, cred=0.95):
        # TODO Documentation
        table = self.draws_lambdas.quantile(
            q=[
                (1 - cred) / 2,
                0.5,
                (1 + cred) / 2,
            ],
        )
        return table


class BFMGLS(BFM):
    # TODO Documentation

    def _compute_lambdas(self):
        # TODO Documentation
        # compute idiosyncratic error covariance
        draws_sige = np.array(
            [
                cov[:self.n, :self.n] - cov[:self.n, -self.k:] @ inv(cov[-self.k:, -self.k:]) @ cov[:self.n, -self.k:].T
                for cov in self.draws_Sigma_y
            ]
        )

        # campute lambdas from betas and mus
        draws_lambda = np.array(
            [
                inv(b.T @ inv(sige) @ b) @ b.T @ inv(sige) @ mu[:self.n]
                for mu, b, sige in zip(self.draws_mu_y, self.draws_betas, draws_sige)
            ]
        )

        return draws_lambda


class BFMOMIT(BFM):
    # TODO Documentation

    def __init__(self, assets, factors, n_draws=1000, p=5):
        self.p = p
        super().__init__(assets, factors, n_draws)

    def _compute_lambdas(self):
        # TODO Documentation
        def cov_svd(cov):
            u, s, _ = svd(cov)
            return (u @ np.diag(s))[:, :self.p]

        draws_beta_upsilon = np.array(
            [
                cov_svd(cov[:self.n, :self.n])
                for cov in self.draws_Sigma_y
            ]
        )

        draws_lambda_upsilon = np.array(
            [
                inv(b.T @ b) @ b.T @ mu[:self.n]
                for b, mu in zip(draws_beta_upsilon, self.draws_mu_y)
            ]
        )

        # TODO parei nos draws do lambda_f


# TODO TESTING - ERASE THIS
# Fama-French Portfolios - Excess Returns
ports = get_ff25p()
rf = get_ffrf()
ports = ports.sub(rf, axis=0)
ports = ports.dropna()
ports.columns = [f"FF{(s - 1) * 5 + v}" for s, v in ports.columns]

facts = get_ff5f()

tic = time()
bfm = BFMOMIT(
    assets=ports,
    factors=facts,
    n_draws=1000,
)
print(time() - tic)
print(bfm.ci_table_lambda())
bfm.plot_lambda()
