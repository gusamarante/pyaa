"""
Recession identification
"""
import getpass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter

from models import fihp

# User Parameters
size = 5

username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Problem Set 01/figures')


# ===== READ DATA =====
# Quarterly GDP
data_wb = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Problem Set 01/PS1 Data Clean.xlsx',
    index_col=0,
    sheet_name="Quarterly GDP",
)
data_wb.index = pd.to_datetime(data_wb.index)
data_wb = data_wb.resample("Q").last()
gdp_us = data_wb['US GDP'].dropna()
gdp_kr = data_wb['KR GDP'].dropna()

# Recessions
data_rec = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Problem Set 01/PS1 Data Clean.xlsx',
    index_col=0,
    sheet_name="Recession",
)
data_rec.index = pd.to_datetime(data_rec.index)
data_rec = data_rec.fillna(0).resample("Q").mean().fillna(0)
data_rec = data_rec.where(data_rec == 0, 1)


# ===== CYCLE ESTIMATORS =====
def log_linear(series):
    """
    ln(series) = a + b * trend + e
    """
    y = np.log(series)
    x = pd.DataFrame(
        data={
            'const': np.ones(len(series)),
            'trend': np.arange(len(series)),
        },
        index=series.index,
    )
    mod = sm.OLS(endog=y, exog=x)
    res = mod.fit()

    trend = pd.Series(
        data=res.params['const'] + res.params['trend'] * np.arange(len(series)),
        index=series.index,
    )
    cycle = res.resid.copy()

    return {'trend': trend, 'cycle': cycle}


def log_quadratic(series):
    """
    ln(series) = a + b1 * trend + b2 * trend^2 + e
    """
    y = np.log(series)
    x = pd.DataFrame(
        data={
            'const': np.ones(len(series)),
            'trend1': np.arange(len(series)),
            'trend2': np.arange(len(series))**2,
        },
        index=series.index,
    )
    mod = sm.OLS(endog=y, exog=x)
    res = mod.fit()

    trend = pd.Series(
        data=res.params['const'] + res.params['trend1'] * np.arange(len(series)) + res.params['trend2'] * np.arange(len(series))**2,
        index=series.index,
    )
    cycle = res.resid.copy()

    return {'trend': trend, 'cycle': cycle}


def hp(series, lamb):
    series = np.log(series)
    cycle, trend = hpfilter(series, lamb=lamb)
    return {'trend': trend, 'cycle': cycle}


def fcast_incr_hp(series, lamb):
    series = np.log(series)
    trend, cycle = fihp(
        series=series,
        lamb=lamb,
        forecast_steps=40,  # 10 years in quarters
        min_obs=16,  # 4 years in quarters
        arima_order=(1, 1, 1),
    )
    return {'trend': trend, 'cycle': cycle}


results = {
    "US": {
        "linear": log_linear(gdp_us),
        "quadratic": log_quadratic(gdp_us),
        "hp 1600": hp(gdp_us, 1600),
        "fihp 1600": fcast_incr_hp(gdp_us, 1600),
    },
    "KR": {
        "linear": log_linear(gdp_kr),
        "quadratic": log_quadratic(gdp_kr),
        "hp 1600": hp(gdp_kr, 1600),
        "fihp 1600": fcast_incr_hp(gdp_kr, 1600),
    },
}

# TODO results["US"]["linear"]['cycle'].shift(2).corr(data_rec["US"])

