"""
Functions and classes for asset allocation models and portfolio construction methods
"""
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np


class MeanVar:

    def __init__(self, mu, cov, rf=0, rb=None, short_sell=True, risk_aversion=None):
        """
        Deals with all aspects of mean-variance optimization
        """
        # TODO Documentation
        # TODO Investor's portfolio depends on the risk aversion and borrowing
        # TODO Composition chart for the complete portfolio and for risky

        # Assertions
        self._assert_indexes(mu, cov)
        assert rb > rf, "'rb' must be larger than 'rf'"


        # Save attributes
        self.mu = mu
        self.cov = cov
        self.sigma = pd.Series(data=np.sqrt(np.diag(cov)), index=mu.index)
        self.rf = rf
        self.rb = rb
        self.short_sell = short_sell
        self.n_assets = self._n_assets()
        self.risk_aversion = risk_aversion

        # Optimal risky porfolio (max sharpe / tangency)
        self.mu_p, self.sigma_p, self.risky_weights, self.sharpe_p = self._get_optimal_risky_portfolio(self.rf)

        # Optimal risky porfolio for borrowers (max sharpe / tangency)
        if self.rb is not None:
            self.mu_b, self.sigma_b, self.risky_weights_b, self.sharpe_b = self._get_optimal_risky_portfolio(self.rb)

        # Global minimum variance portfolio
        self.mu_mv, self.sigma_mv, self.mv_weights, self.sharpe_mv = self._get_minimal_variance_portfolio()

        # Investor's Portfolio
        if risk_aversion is not None:
            self.weight_p, self.complete_weights, self.mu_c, self.sigma_c, self.certain_equivalent = self._investor_allocation()

    def plot(self,
             figsize=(10, 7),
             save_path=None,
             title=None,
             assets=True,  # plot individual assets
             gmvp=True,  # plot global min var
             max_sharpe=True,  # Max Sharpe port
             risk_free=True,  # plot rf
             mvf=True,  # MinVar Frontier
             mvfnoss=True,  # MinVar Frontier no short selling
             cal=True,  # Capital Allocation Line
             investor=True,  # Investor's indifference, portfolio, and CE
             ):

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")
        ax = plt.subplot2grid((1, 1), (0, 0))

        # Elements
        if assets:
            ax.scatter(self.sigma, self.mu, label='Assets')

        if gmvp:
            ax.scatter(self.sigma_mv, self.mu_mv, label='Global Minimum Variance')

        if max_sharpe:
            ax.scatter(self.sigma_p, self.mu_p, label='Maximum Sharpe')

        if risk_free:
            ax.scatter(0, self.rf, label='Risk-Free')

        if mvf and self.n_assets != 1:
            mu_mv, sigma_mv = self.min_var_frontier(short_sell=True)
            ax.plot(sigma_mv, mu_mv, marker=None, zorder=-1, label='Minimum Variance Frontier')

        if mvfnoss and self.n_assets != 1:
            mu_mv, sigma_mv = self.min_var_frontier(short_sell=False)
            ax.plot(sigma_mv, mu_mv, marker=None, zorder=-1, label='Minimum Variance Frontier (No Short Selling)')

        if cal:
            if self.rb is None:
                max_sigma = self.sigma.max() + 0.05
                x_values = [0, max_sigma]
                y_values = [self.rf, self.rf + self.sharpe_p * max_sigma]
                plt.plot(x_values, y_values, marker=None, zorder=-1, label='Capital Allocation Line')
            else:
                # If borrowing costs more
                x_cal = [0, self.sigma_p]
                y_cal = [self.rf, self.rf + self.sharpe_p * self.sigma_p]
                plt.plot(x_cal, y_cal, marker=None, zorder=-1, label='Capital Allocation Line (No Borrowing)')

                ax.scatter(self.sigma_b, self.mu_b, label='Maximum Sharpe (Borrowing)')
                ax.scatter(0, self.rb, label='Borrowing Cost')

                max_sigma = self.sigma.max() + 0.05
                x_bor1 = [0, self.sigma_b]
                x_bor2 = [self.sigma_b, max_sigma]
                y_bor1 = [self.rb, self.rb + self.sharpe_b * self.sigma_b]
                y_bor2 = [self.rb + self.sharpe_b * self.sigma_b, self.rb + self.sharpe_b * max_sigma]
                plt.plot(x_bor1, y_bor1, marker=None, zorder=-1, color='grey', ls='--', lw=1, label=None)
                plt.plot(x_bor2, y_bor2, marker=None, zorder=-1, color='grey', label='Capital Allocation Line (Borrowing)')

        if investor and (self.risk_aversion is not None):
            max_sigma = self.sigma_c + 0.02
            x_values = np.arange(0, max_sigma, max_sigma / 100)
            y_values = self.certain_equivalent + 0.5 * self.risk_aversion * (x_values ** 2)
            ax.plot(x_values, y_values, marker=None, zorder=-1, label='Indiference Curve')
            ax.scatter(self.sigma_c, self.mu_c, label="Investor's Portfolio")

        # Adjustments
        ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.legend(loc='best')
        ax.set_xlim((0, self.sigma.max() + 0.05))
        ax.set_xlabel('Risk')
        ax.set_ylabel('Return')
        plt.tight_layout()

        # Save as picture
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def min_var_frontier(self, n_steps=100, short_sell=True):

        if short_sell:
            # Analytical solution when short-selling is allowed
            E = self.mu.values
            inv_cov = np.linalg.inv(self.cov)

            A = E @ inv_cov @ E
            B = np.ones(self.n_assets) @ inv_cov @ E
            C = np.ones(self.n_assets) @ inv_cov @ np.ones(self.n_assets)

            def min_risk(mu):
                return np.sqrt((C * (mu ** 2) - 2 * B * mu + A) / (A * C - B ** 2))

            min_mu = min(self.mu.min(), self.rf) - 0.05
            max_mu = max(self.mu.max(), self.rf) + 0.05

            mu_range = np.arange(min_mu, max_mu, (max_mu - min_mu) / n_steps)
            sigma_range = np.array(list(map(min_risk, mu_range)))

        else:

            sigma_range = []

            # Objective function
            def risk(x):
                return np.sqrt(x @ self.cov @ x)

            # initial guess
            w0 = np.zeros(self.n_assets)
            w0[0] = 1

            # Values for mu to perform the minimization
            mu_range = np.linspace(self.mu.min(), self.mu.max(), n_steps)

            for mu_step in tqdm(mu_range, 'Finding Mininmal variance frontier'):

                # budget and return constraints
                constraints = ({'type': 'eq',
                                'fun': lambda w: w.sum() - 1},
                               {'type': 'eq',
                                'fun': lambda w: sum(w * self.mu) - mu_step})

                bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

                # Run optimization
                res = minimize(risk, w0,
                               method='SLSQP',
                               constraints=constraints,
                               bounds=bounds,
                               options={'ftol': 1e-9, 'disp': False})

                if not res.success:
                    raise RuntimeError("Convergence Failed")

                # Compute optimal portfolio parameters
                sigma_step = np.sqrt(res.x @ self.cov @ res.x)

                sigma_range.append(sigma_step)

            sigma_range = np.array(sigma_range)

        return mu_range, sigma_range

    def _investor_allocation(self):
        weight_p = (self.mu_p - self.rf) / (self.risk_aversion * (self.sigma_p**2))
        complete_weights = self.risky_weights * weight_p
        complete_weights.loc['Risk Free'] = 1 - weight_p

        mu_c = weight_p * self.mu_p + (1 - weight_p) * self.rf
        sigma_c = weight_p * self.sigma_p

        ce = self._utility(mu_c, sigma_c, self.risk_aversion)

        return weight_p, complete_weights, mu_c, sigma_c, ce

    def _get_optimal_risky_portfolio(self, rf):

        if self.n_assets == 1:  # one risky asset (analytical)
            mu_p = self.mu.iloc[0]
            sigma_p = self.cov.iloc[0, 0]
            sharpe_p = (mu_p - rf) / sigma_p
            weights = pd.Series(data={self.mu.index[0]: 1},
                                name='Risky Weights')

        else:  # multiple risky assets (optimization)

            # objective function (notice the sign change on the return value)
            def sharpe(x):
                return - self._sharpe(x, self.mu.values, self.cov.values, rf, self.n_assets)

            # budget constraint
            constraints = ({'type': 'eq',
                            'fun': lambda w: w.sum() - 1})

            # Create bounds for the weights if short-selling is restricted
            if self.short_sell:
                bounds = None
            else:
                bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

            # initial guess
            w0 = np.zeros(self.n_assets)
            w0[0] = 1

            # Run optimization
            res = minimize(sharpe, w0,
                           method='SLSQP',
                           constraints=constraints,
                           bounds=bounds,
                           options={'ftol': 1e-9, 'disp': False})

            if not res.success:
                raise RuntimeError("Convergence Failed")

            # Compute optimal portfolio parameters
            mu_p = np.sum(res.x * self.mu.values.flatten())
            sigma_p = np.sqrt(res.x @ self.cov @ res.x)
            sharpe_p = - sharpe(res.x)
            weights = pd.Series(index=self.mu.index,
                                data=res.x,
                                name='Risky Weights')

        return mu_p, sigma_p, weights, sharpe_p

    def _get_minimal_variance_portfolio(self):

        def risk(x):
            return np.sqrt(x @ self.cov @ x)

        # budget constraint
        constraints = ({'type': 'eq',
                        'fun': lambda w: w.sum() - 1})

        # Create bounds for the weights if short-selling is restricted
        if self.short_sell:
            bounds = None
        else:
            bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

        # initial guess
        w0 = np.zeros(self.n_assets)
        w0[0] = 1

        # Run optimization
        res = minimize(risk, w0,
                       method='SLSQP',
                       constraints=constraints,
                       bounds=bounds,
                       options={'ftol': 1e-9, 'disp': False})

        if not res.success:
            raise RuntimeError("Convergence Failed")

        # Compute optimal portfolio parameters
        mu_mv = np.sum(res.x * self.mu.values.flatten())
        sigma_mv = np.sqrt(res.x @ self.cov @ res.x)
        sharpe_mv = (mu_mv - self.rf) / sigma_mv
        weights = pd.Series(index=self.mu.index,
                            data=res.x,
                            name='Minimal Variance Weights')

        return mu_mv, sigma_mv, weights, sharpe_mv

    @staticmethod
    def _assert_indexes(mu, cov):
        cond = sorted(mu.index) == sorted(cov.index)
        assert cond, "elements in the input indexes do not match"

    def _n_assets(self):
        """
        Makes sure that all inputs have the correct shape and returns the number of assets
        """
        shape_mu = self.mu.shape
        shape_sigma = self.cov.shape

        max_shape = max(shape_mu[0], shape_sigma[0], shape_sigma[1])
        min_shape = min(shape_mu[0], shape_sigma[0], shape_sigma[1])

        if max_shape == min_shape:
            return max_shape
        else:
            raise AssertionError('Mismatching dimensions of inputs')

    @staticmethod
    def _sharpe(w, mu, cov, rf, n):
        er = np.sum(w * mu)

        w = np.reshape(w, (n, 1))
        risk = np.sqrt(w.T @ cov @ w)[0][0]

        sharpe = (er - rf) / risk
        return sharpe

    @staticmethod
    def _utility(mu, sigma, risk_aversion):
        return mu - 0.5 * risk_aversion * (sigma ** 2)
