from numpy.linalg import inv
import pandas as pd
import numpy as np


class BlackLitterman:

    def __init__(self, sigma, estimation_error, views_p, views_v, w_equilibrium=None, avg_risk_aversion=1.2,
                 mu_historical=None, mu_shrink=1, overall_confidence=1, relative_uncertainty=None):
        """
        Black-Litterman model for asset allocation. The model combines model estimates and views
        from the asset allocators. The views are expressed in the form of

            views_p @ mu = views_v

        where 'views_p' is a selection matrix and 'views_v' is a vector of values.

        Parameters
        ----------
        sigma : pandas.DataFrame
            robustly estimated covariance matrix of the assets

        estimation_error : float
            Uncertainty of the estimation. Recomended value is the inverse of the sample size
            used in the covariance matrix

        views_p : pandas.DataFrame
            selection matrix of the views

        views_v : pandas.DataFrame
            value matrix of the views

        w_equilibrium : pandas.DataFrame
            weights of each asset in the equilibrium

        avg_risk_aversion : float
            average risk aversion of the investors

        mu_historical : pandas.DataFrame
            historical returns of the asset class (can be interpreted as the target
            of the shrinkage estimate)

        mu_shrink : float
            between 0 and 1, shirinkage intensity. If 1 (default), best guess of mu
            is the model returns. If 0, bet guess of mu is 'mu_historical'

        overall_confidence : float
            the higher the number, the more weight the views have in te posterior

        relative_uncertainty : pandas.DataFrame
            the higher the value the less certain that view is
        """
        self.sigma = sigma.sort_index(axis=0).sort_index(axis=1)
        self.asset_names = list(self.sigma.index)
        self.n_assets = sigma.shape[0]
        self.estimation_error = estimation_error
        self.avg_risk_aversion = avg_risk_aversion
        self.mu_shrink = mu_shrink

        self.w_equilibrium = self._get_w_equilibrium(w_equilibrium)
        self.equilibrium_returns = self._get_equilibrium_returns()
        self.mu_historical = self._get_mu_historical(mu_historical)
        self.mu_best_guess = self._get_mu_best_guess()

        self.views_p = views_p.sort_index(axis=0).sort_index(axis=1)
        self.views_v = views_v.sort_index()
        self.n_views = views_p.shape[0]
        self.view_names = list(self.views_p.index)
        self.overall_confidence = overall_confidence
        self.relative_uncertainty = self._get_relative_uncertainty(relative_uncertainty)
        self.omega = self._get_views_covariance()

        self.mu_bl, self.sigma_mu_bl = self._get_mu_bl()
        self.sigma_bl = self.sigma + self.sigma_mu_bl

    def _get_w_equilibrium(self, w_equilibrium):
        """
        In case 'w_equilibrium' is not passed, assumes the equilibrium is equal weighted.
        """
        if w_equilibrium is None:
            w_equilibrium = (1 / self.n_assets) * np.ones(self.n_assets)
            w_equilibrium = pd.DataFrame(data=w_equilibrium,
                                         index=self.asset_names,
                                         columns=['Equilibrium Weights'])
        else:
            w_equilibrium = w_equilibrium.sort_index()

        return w_equilibrium

    def _get_equilibrium_returns(self):
        """
        Computes the equilibrium returns based on the equilibrium weights and
        average risk aversion.
        """
        sigma = self.sigma.values
        w_equilibrium = self.w_equilibrium.values
        pi = 2 * self.avg_risk_aversion * sigma @ w_equilibrium
        pi = pd.Series(data=pi,
                       index=self.asset_names,
                       name='Equilibrium Returns')
        return pi

    def _get_mu_historical(self, mu_historical):
        """
        In case 'mu_historical' is not passed, uses zeros as the shrinking target.
        """
        if mu_historical is None:
            mu_historical = np.zeros(self.n_assets)
            mu_historical = pd.DataFrame(data=mu_historical,
                                         index=self.asset_names,
                                         columns=['Historical Returns'])
        else:
            mu_historical = mu_historical.sort_index()

        return mu_historical

    def _get_mu_best_guess(self):
        """
        Uses shrinkage to estimate the best guess for mu by balancing between
        the model equilibrium returns and the historical returns.
        """
        best_guess = self.mu_shrink * self.equilibrium_returns + (1-self.mu_shrink) * self.mu_historical
        best_guess = best_guess.rename('Best Guess of mu')
        return best_guess

    def _get_relative_uncertainty(self, relative_uncertainty):
        """
        In case 'relative_uncertainty' is not passed, uses ones for every asset.
        """
        if relative_uncertainty is None:
            relative_uncertainty = np.ones(self.n_views)
            relative_uncertainty = pd.DataFrame(data=relative_uncertainty,
                                                columns=['Relative Uncertainty'],
                                                index=self.view_names)
        else:
            relative_uncertainty = relative_uncertainty.sort_index()

        return relative_uncertainty

    def _get_views_covariance(self):
        """
        Computes Omega, the covariance of the views.
        """
        c = self.overall_confidence
        u = np.diag(self.relative_uncertainty.values.flatten())
        P = self.views_p.values
        Sigma = self.sigma.values

        omega = (1 / c) * u @ P @ Sigma @ P.T @ u

        if np.linalg.det(omega) < np.finfo(float).eps:
            n, m = omega.shape
            omega = omega + 1e-16 * np.eye(n, m)

        omega = pd.DataFrame(data=omega,
                             index=self.view_names,
                             columns=self.view_names)

        return omega

    def _get_mu_bl(self):
        """
        Computes 'mu_bl', the vector of returns that combines the best guess
        for mu (equilibrium and empirical) with the views from the asset allocators
        """
        tau = self.estimation_error
        sigma = self.sigma.values
        P = self.views_p.values
        pi = self.mu_best_guess
        v = self.views_v.values
        omega = self.omega.values

        sigma_mu_bl = inv(inv(tau * sigma) + P.T @ inv(omega) @ P)
        B = inv(tau * sigma) @ pi + P.T @ inv(omega) @ v
        mu_bl = sigma_mu_bl @ B

        sigma_mu_bl = pd.DataFrame(data=sigma_mu_bl, index=self.asset_names, columns=self.asset_names)
        mu_bl = pd.Series(data=mu_bl, index=self.asset_names, name='Expected Returns')

        return mu_bl, sigma_mu_bl
