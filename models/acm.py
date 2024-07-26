from sklearn.decomposition import PCA
from numpy.linalg import inv
import pandas as pd
import numpy as np
from utils.math import vec


class NominalACM:
    """
    This class implements the model from the article:

        Adrian, Tobias, Richard K. Crump, and Emanuel Moench. “Pricing the
        Term Structure with Linear Regressions.” SSRN Electronic Journal,
        2012. https://doi.org/10.2139/ssrn.1362586.

    It handles data transformation, estimates parameters and generates the
    relevant outputs. The version of the article that was published by the NY
    FED is not 100% explicit on how the data is being manipulated, but I found
    an earlier version of the paper on SSRN where the authors go deeper into the
    details on how everything is being estimated:
        - Data for zero yields uses monthly maturities starting from month 1
        - All principal components and model parameters are estiamted with data
          resampled to a monthly frequency by averaging observations in each
          month.
        - To get daily / real-time estimates, the factor loading estimated from
          the monthly frquency are used to transform the daily data.

    This class was updated usign code from GitHub user miabrahams. His jupyter
    notebook had much more succint formulas, that made code easier to understand
    and faster to run.
    """

    def __init__(self, curve, n_factors=5):
        """
        Runs the baseline varsion of the ACM term premium model. Works for data
        with monthly frequency or higher.

        Parameters
        ----------
        curve : pandas.DataFrame
            Annualized log-yields. Maturities (columns) must start at month 1
            and be equally spaced in monthly frequency. The labels of the
            columns do not matter, they be kept the same. Observations (index)
            must be of monthly frequency or higher. The index must be a
            pandas.DateTimeIndex.

        n_factors : int
            number of principal components to used as state variables.
        """

        self.n_factors = n_factors
        self.curve = curve
        self.curve_monthly = curve.resample('M').mean()
        self.t = self.curve_monthly.shape[0] - 1
        self.n = self.curve_monthly.shape[1]
        self.rx_m, self.rf_m = self._get_excess_returns()
        self.rf_d = self.curve.iloc[:, 0] * (1 / 12)
        self.pc_factors_m, self.pc_loadings_m, self.pc_explained_m = self._get_pcs(self.curve_monthly)
        self.pc_factors_d, self.pc_loadings_d, self.pc_explained_d = self._get_pcs(self.curve)
        self.mu, self.phi, self.Sigma, self.v = self._estimate_var()
        self.a, self.beta, self.c, self.sigma2 = self._excess_return_regression()
        self.lambda0, self.lambda1 = self._retrieve_lambda()

        if self.curve.index.freqstr == 'M':
            X = self.pc_factors_m
            r1 = self.rf_m
        else:
            X = self.pc_factors_d
            r1 = self.rf_d

        self.miy = self._affine_recursions(self.lambda0, self.lambda1, X, r1)
        self.rny = self._affine_recursions(0, 0, X, r1)
        self.tp = self.miy - self.rny
        self.er_loadings, self.er_hist_m, self.er_hist_d = self._expected_return()
        self._inference()

    def _get_excess_returns(self):
        ttm = np.arange(1, self.n + 1) / 12
        log_prices = - self.curve_monthly * ttm
        rf = - log_prices.iloc[:, 0].shift(1)
        rx = (log_prices - log_prices.shift(1, axis=0).shift(-1, axis=1)).subtract(rf, axis=0)
        rx = rx.dropna(how='all', axis=0).dropna(how='all', axis=1)
        return rx, rf.dropna()

    def _get_pcs(self, curve):
        pca = PCA(n_components=self.n_factors)
        pca.fit(curve)
        col_names = [f'PC {i + 1}' for i in range(self.n_factors)]
        df_loadings = pd.DataFrame(data=pca.components_.T,
                                   columns=col_names,
                                   index=curve.columns)

        # Normalize the direction of the eigenvectors
        signal = np.sign(df_loadings.iloc[-1])
        df_loadings = df_loadings * signal
        df_pc = (curve - curve.mean()) @ df_loadings

        # Percent Explained
        df_explained = pd.Series(data=pca.explained_variance_ratio_,
                                 name='Explained Variance',
                                 index=col_names)

        return df_pc, df_loadings, df_explained

    def _estimate_var(self):
        X = self.pc_factors_m.copy().T
        X_lhs = X.values[:, 1:]  # X_t+1. Left hand side of VAR
        X_rhs = np.vstack((np.ones((1, self.t)), X.values[:, 0:-1]))  # X_t and a constant.

        var_coeffs = (X_lhs @ np.linalg.pinv(X_rhs))
        mu = var_coeffs[:, [0]]
        phi = var_coeffs[:, 1:]

        v = X_lhs - var_coeffs @ X_rhs
        Sigma = v @ v.T / self.t

        return mu, phi, Sigma, v

    def _excess_return_regression(self):
        X = self.pc_factors_m.copy().T.values[:, :-1]
        Z = np.vstack((np.ones((1, self.t)), self.v, X))  # Innovations and lagged X
        abc = self.rx_m.values.T @ np.linalg.pinv(Z)
        E = self.rx_m.values.T - abc @ Z
        sigma2 = np.trace(E @ E.T) / (self.n * self.t)

        a = abc[:, [0]]
        beta = abc[:, 1:self.n_factors + 1].T
        c = abc[:, self.n_factors + 1:]

        return a, beta, c, sigma2

    def _retrieve_lambda(self):
        BStar = np.squeeze(np.apply_along_axis(self.vec_quad_form, 1, self.beta.T))
        lambda1 = np.linalg.pinv(self.beta.T) @ self.c
        lambda0 = np.linalg.pinv(self.beta.T) @ (self.a + 0.5 * (BStar @ vec(self.Sigma) + self.sigma2))
        return lambda0, lambda1

    def _affine_recursions(self, lambda0, lambda1, X_in, r1):
        X = X_in.T.values[:, 1:]
        r1 = vec(r1.values)[-X.shape[1]:, :]

        A = np.zeros((1, self.n))
        B = np.zeros((self.n_factors, self.n))

        delta = r1.T @ np.linalg.pinv(np.vstack((np.ones((1, X.shape[1])), X)))
        delta0 = delta[[0], [0]]
        delta1 = delta[[0], 1:]

        A[0, 0] = - delta0
        B[:, 0] = - delta1

        for i in range(self.n - 1):
            A[0, i + 1] = A[0, i] + B[:, i].T @ (self.mu - lambda0) + 1 / 2 * (B[:, i].T @ self.Sigma @ B[:, i] + 0 * self.sigma2) - delta0
            B[:, i + 1] = B[:, i] @ (self.phi - lambda1) - delta1

        # Construct fitted yields
        ttm = np.arange(1, self.n + 1) / 12
        fitted_log_prices = (A.T + B.T @ X).T
        fitted_yields = - fitted_log_prices / ttm
        fitted_yields = pd.DataFrame(
            data=fitted_yields,
            index=self.curve.index[1:],
            columns=self.curve.columns,
        )
        return fitted_yields

    def _expected_return(self):
        """
        Compute the "expected return" and "convexity adjustment" terms, to get
        the expected return loadings and historical estimate

        Loadings are interpreted as the effect of 1sd of the PCs on the
        expected returns
        """
        stds = self.pc_factors_m.std().values[:, None].T
        er_loadings = (self.beta.T @ self.lambda1) * stds
        er_loadings = pd.DataFrame(
            data=er_loadings,
            columns=self.pc_factors_m.columns,
            index=self.curve.columns[:-1],
        )

        # Monthly
        exp_ret = (self.beta.T @ (self.lambda1 @ self.pc_factors_m.T + self.lambda0)).values
        conv_adj = np.diag(self.beta.T @ self.Sigma @ self.beta) + self.sigma2
        er_hist = (exp_ret + conv_adj[:, None]).T
        er_hist_m = pd.DataFrame(
            data=er_hist,
            index=self.pc_factors_m.index,
            columns=self.curve.columns[:er_hist.shape[1]]
        )

        # Higher frequency
        exp_ret = (self.beta.T @ (self.lambda1 @ self.pc_factors_d.T + self.lambda0)).values
        conv_adj = np.diag(self.beta.T @ self.Sigma @ self.beta) + self.sigma2
        er_hist = (exp_ret + conv_adj[:, None]).T
        er_hist_d = pd.DataFrame(
            data=er_hist,
            index=self.pc_factors_d.index,
            columns=self.curve.columns[:er_hist.shape[1]]
        )

        return er_loadings, er_hist_m, er_hist_d

    def _inference(self):

        Z = self.pc_factors_m.copy().T
        Z = Z.values[:, 1:]
        Z = np.vstack((np.ones((1, self.t)), Z))
        upsilon_zz = (1 / self.t) * Z @ Z.T

        v1 = np.kron(inv(upsilon_zz), self.Sigma)

    @staticmethod
    def vec_quad_form(x):
        return vec(np.outer(x, x))
