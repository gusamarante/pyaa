import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)


def fihp(series, lamb=100, forecast_steps=None, min_obs=10, arima_order=(1, 1, 1)):
    """
    Forecast Incremental HP Filter detrending method. At each point in time,
    makes an ARIMA forecast, uses the series with the forecast as input for the
     HP filter, and finally only grabs the trend value of the HP filter for
     this date.

    Parameters
    ----------
    series: pandas.Series
        must have a defined frequency for its index.

    lamb: float
        penalization parameter for the HP filter

    forecast_steps: int
        Number of steps ahead in the forecast step

    min_obs: int
        Minimum number of observations to start doing the forecast.

    arima_order: tuple
        3-uple with the (p, d, q) specification parameters of an ARIMA model

    Returns
    -------
    trend: pandas.Series
        Estiamted trend component

    cycle: pandas.Series
        Estiamted cycle component
    """

    assert series.index.freq is not None, "`series` must have a set frequency"

    trend = pd.Series(index=series.index, name=f"{series.name} FIHP")
    for d in series.index[min_obs:]:

        # Get the partial series
        aux_series = series.loc[:d].copy()

        # Estimate the ARIMA
        mod = ARIMA(endog=aux_series, order=arima_order, trend='t')
        res = mod.fit()

        # Make the forecast
        aux_series = pd.concat(
            [
                aux_series,
                res.forecast(steps=forecast_steps),
            ],
            axis=0,
        )

        # Run the HP Filter
        _, aux_trend = hpfilter(aux_series, lamb)

        # Save the one for date d
        trend.loc[d] = aux_trend.loc[d]

    # Fill missing with full sample
    _, fill_trend = hpfilter(series, lamb)
    trend = trend.fillna(fill_trend)

    # Compute Cycle
    cycle = series - trend

    return trend, cycle
