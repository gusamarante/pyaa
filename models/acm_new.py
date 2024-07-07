# TODO cite https://github.com/miabrahams
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


class NominalACM:
    # TODO add inference

    def __init__(self, curve, n_factors=5):
        # TODO documentation
        #  curve must be of log yields
        #  must be monthly and start at month 1
        #  equally spaced maturities

        self.n_factors = n_factors
        self.curve = curve
        self.curve_monthly = curve.resample('M').mean()
        self.t = self.curve_monthly.shape[0] - 1
        self.n = self.curve_monthly.shape[1]
        self.n_maturities = curve.shape[1]
        self.rx_m, self.rf_m = self._get_excess_returns()
        self.rf_d = self.curve.iloc[:, 0] * (1 / 12)
        self.pc_factors, self.pc_loadings = self._get_pcs(self.curve_monthly)
        self.pc_factors_d, self.pc_loadings_d = self._get_pcs(self.curve)
        self.mu, self.phi, self.Sigma, self.v = self._estimate_var()
        self.a, self.beta, self.c, self.sigma2 = self._excess_return_regression()
        self.lambda0, self.lambda1 = self._retrieve_lambda()

        self.miy = self._affine_recursions(self.lambda0, self.lambda1, self.pc_factors_d, self.rf_d)
        self.rny = self._affine_recursions(0, 0, self.pc_factors_d, self.rf_d)
        # self.tp = 1
        # TODO Compute Yields on daily frequency, NOT WORKING

    def _get_excess_returns(self):
        ttm = np.arange(1, self.n_maturities + 1) / 12
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

        return df_pc, df_loadings

    def _estimate_var(self):
        X = self.pc_factors.copy().T
        X_lhs = X.values[:, 1:]  # X_t+1. Left hand side of VAR
        X_rhs = np.vstack((np.ones((1, self.t)), X.values[:, 0:-1]))  # X_t and a constant.

        var_coeffs = (X_lhs @ np.linalg.pinv(X_rhs))
        mu = var_coeffs[:, [0]]
        phi = var_coeffs[:, 1:]

        v = X_lhs - var_coeffs @ X_rhs
        Sigma = v @ v.T / self.t

        return mu, phi, Sigma, v

    def _excess_return_regression(self):
        X = self.pc_factors.copy().T.values[:, :-1]
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
        lambda0 = np.linalg.pinv(self.beta.T) @ (self.a + 0.5 * (BStar @ self.vec(self.Sigma) + self.sigma2))
        return lambda0, lambda1

    def _affine_recursions(self, lambda0, lambda1, X_in, r1):
        X = X_in.T.values[:, 1:]
        r1 = self.vec(r1.values)[1:, :]

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
        ttm = np.arange(1, self.n_maturities + 1) / 12
        fitted_log_prices = (A.T + B.T @ X).T
        fitted_yields = - fitted_log_prices / ttm
        fitted_yields = pd.DataFrame(
            data=fitted_yields,
            index=self.curve.index[1:],
            columns=self.curve.columns,
        )
        return fitted_yields

    @staticmethod
    def vec(x):
        return np.reshape(x, (-1, 1))

    def vec_quad_form(self, x):
        return self.vec(np.outer(x, x))
