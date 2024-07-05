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
        self.curve_monthly = curve.resample('M').mean()
        self.n_maturities = curve.shape[1]
        self.rx = self._get_excess_returns()
        self.pc_factors, self.pc_loadings = self._get_pcs()

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

        # TODO PAREI AQUI

        a = 1

