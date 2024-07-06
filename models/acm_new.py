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
        self.n_factors = n_factors
        self.curve = curve
        self.t =
        self.curve_monthly = curve.resample('M').mean()
        self.n_maturities = curve.shape[1]
        self.rx = self._get_excess_returns()
        self.pc_factors, self.pc_loadings = self._get_pcs()
        self._estimate_var()


        # TODO Compute PCs on daily frequency

    def _get_excess_returns(self):
        ttm = np.arange(1, self.n_maturities + 1) / 12
        log_prices = - self.curve_monthly * ttm
        rf = - log_prices.iloc[:, 0].shift(1)
        rx = (log_prices - log_prices.shift(1, axis=0).shift(-1, axis=1)).subtract(rf, axis=0)
        return rx

    def _get_pcs(self):
        pca = PCA(n_components=self.n_factors)
        pca.fit(self.curve_monthly)
        col_names = [f'PC {i + 1}' for i in range(self.n_factors)]
        df_loadings = pd.DataFrame(data=pca.components_.T,
                                   columns=col_names,
                                   index=self.curve.columns)

        # Normalize the direction of the eigenvectors
        signal = np.sign(df_loadings.iloc[-1])
        df_loadings = df_loadings * signal
        df_pc = (self.curve_monthly - self.curve_monthly.mean()) @ df_loadings

        return df_pc, df_loadings

    def _estimate_var(self):
        X = self.pc_factors.copy().T
        X_lhs = X[:, 1:]  # X_t+1. Left hand side of VAR
        X_rhs = np.vstack((np.ones((1, t)), X[:, 0:-1]))  # X_t and a constant.
        var_coeffs = (X_lhs @ np.linalg.pinv(X_rhs))
        mu = var_coeffs[:, [0]]
        phi = var_coeffs[:, 1:]

        v = X_lhs - var_coeffs @ X_rhs
        Sigma = v @ v.T / t

        # TODO parei aqui, arrumar t e check var estimation
