import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np


class TSMOM:
    """
    This class replicates the methodology from

    Moskowitz, Tobias J., Yao Hua Ooi, and Lasse Heje Pedersen. “Time Series
    Momentum.” Journal of Financial Economics 104, no. 2 (May 2012): 228–50.
    https://doi.org/10.1016/j.jfineco.2011.11.003.
    """
    # TODO Adapt code to allow for multiple series in `tracker`

    def __init__(self, tracker):
        self.tracker = tracker
        self.ret_m = self.tracker.pct_change(21)
        self.vol_m = self._get_vol()
        self.scaled_rets = self.ret_m / self.vol_m.shift(1)

    def predictability(self, n_lags=24, show_chart=False):
        # TODO Documentation
        """
        sr_t = a + b * sret_t-1 + eps
        """
        srets = self.scaled_rets.resample('M').last().dropna()

        tstats = pd.Series(name='t-stats')
        for h in range(1, n_lags + 1):
            sretsl = srets.shift(h)  # H

            aux = pd.DataFrame({
                'Y': srets,
                'X': sretsl
            })
            aux = aux.dropna()

            model = sm.OLS(aux['Y'], sm.add_constant(aux['X']))
            results = model.fit()

            tstats.loc[h] = results.tvalues['X']

        if show_chart:
            ax = tstats.plot(
                kind='bar',
                grid=True,
                title='Predictability Regression',
            )
            ax.set_xlabel('Months of Lag')
            ax.set_ylabel('t-Statistics')
            plt.tight_layout()
            plt.show()

        return tstats

    def strategy(self):
        # TODO to implement
        pass

    def _get_vol(self):
        rets = self.tracker.pct_change(1)
        vol = rets.ewm(com=60).std() * np.sqrt(21)
        vol = vol.dropna()
        return vol
