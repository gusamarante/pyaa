import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import get_ffrf, get_ff5f, get_ff25p
from numpy.linalg import inv, svd
from scipy.stats import invwishart, multivariate_normal
from time import time

# TODO add frequentist FM to charts


class FM:

    def __init__(self, assets, factors):

        self.beta = pd.DataFrame(
            data=(inv(factors.T @ factors) @ factors.T @ assets).values,
            columns=assets.columns,
            index=factors.columns,
        ).T

        mu = assets.mean()
        self.lambdas = pd.Series(
            data=(inv(self.beta.T @ self.beta) @ self.beta.T @ mu).values,
            index=factors.columns,
            name="Lambdas",
        )


class BFM:
    # TODO Documentation

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

        self.draws_lambdas = self._compute_lambdas()
        self.draws_r2 = self._compute_r2()

        # Reorganize in Pandas
        self.draws_lambdas = pd.DataFrame(
            data=self.draws_lambdas,
            columns=factors.columns
        )
        self.draws_r2 = pd.Series(
            data=self.draws_r2,
            name="R2",
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

    def _compute_r2(self):
        # TODO Documentation
        mu_r = self.mu_y[:self.n].values
        denom = (mu_r - mu_r.mean()) @ (mu_r - mu_r.mean())
        draws_r2 = [
            1 - ((mu_r - b @ l) @ (mu_r - b @ l)) / denom
            for b, l in zip(self.draws_betas, self.draws_lambdas)
        ]
        return draws_r2

    def plot_lambda(self, include_fm=False):
        axes = self.draws_lambdas.hist(
            density=True,
            bins=int(np.sqrt(self.n_draws)),
            sharex=True,
            figsize=(10, 6)
        )

        if include_fm:
            fm = FM(self.assets, self.factors)
            for ax in axes.flatten():
                try:
                    ax.axvline(
                        fm.lambdas[ax.title.get_text()],
                        color='tab:orange',
                        lw=2,
                    )
                except KeyError:
                    continue

        plt.tight_layout()
        plt.show()

    def plot_r2(self):
        # TODO Documentation
        axes = self.draws_r2.hist(
            density=True,
            bins=int(np.sqrt(self.n_draws)),
            figsize=(10, 6)
        )

        axes.axvline(0, color='tab:orange', lw=1)

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

    def ci_table_r2(self, cred=0.95):
        # TODO Documentation
        table = self.draws_r2.quantile(
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

    def _compute_r2(self):
        # TODO Documentation
        draws_sige = np.array(
            [
                cov[:self.n, :self.n] - cov[:self.n, -self.k:] @ inv(cov[-self.k:, -self.k:]) @ cov[:self.n, -self.k:].T
                for cov in self.draws_Sigma_y
            ]
        )

        mu_r = self.mu_y[:self.n].values
        mu_r_bar = mu_r.mean()

        draws_r2 = [
            1 - ((mu_r - b @ l) @ inv(sige) @ (mu_r - b @ l)) / ((mu_r - mu_r_bar) @ inv(sige) @ (mu_r - mu_r_bar))
            for b, l, sige in zip(self.draws_betas, self.draws_lambdas, draws_sige)
        ]
        return draws_r2


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

        draws_lambda_f = np.array(
            [
                lu.T @ inv(bu.T @ bu) @ bu.T @ cov[:self.n, -self.k:]
                for lu, bu, cov in zip(draws_lambda_upsilon, draws_beta_upsilon, self.draws_Sigma_y)
            ]
        )

        return draws_lambda_f

    def _compute_r2(self):
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

        mu_r = self.mu_y[:self.n].values
        mu_r_bar = mu_r.mean()
        draws_r2 = np.array(
            [
                1 - ((mu_r - bu @ lu) @ (mu_r - bu @ lu)) / ((mu_r - mu_r_bar) @ (mu_r - mu_r_bar))
                for lu, bu in zip(draws_lambda_upsilon, draws_beta_upsilon)
            ]
        )
        return draws_r2


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
    n_draws=10000,
    p=10,
)
print(time() - tic)
# print(bfm.ci_table_lambda())
# print(bfm.ci_table_r2())
bfm.plot_lambda(include_fm=True)
# bfm.plot_r2()
