"""
Exemplifies different detrending methodologies
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


# ===== ALL CHARTS =====
for method in results['US'].keys():
    fig = plt.figure(figsize=(size * (16 / 7.3), size))

    # US Trend
    ax = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    ax.set_title(f"United States")
    ax.plot(np.log(gdp_us), label='log GDP', lw=2)
    ax.plot(results["US"][method]['trend'], label='Estimated Trend', lw=1)
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=True, loc="upper left")
    ax.set_ylabel("Trend")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(rotation=90, axis="x")
    ax.set_xticklabels([])

    # US Cycle
    ax = plt.subplot2grid((3, 2), (2, 0))
    ax.plot(results["US"][method]['cycle'])
    ax.axhline(0, color='black', lw=0.5)
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Cycle")
    ax.tick_params(rotation=90, axis="x")

    # KR Trend
    ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
    ax.set_title(f"South Korea")
    ax.plot(np.log(gdp_kr), label='log GDP', lw=2)
    ax.plot(results["KR"][method]['trend'], label='Estimated Trend', lw=1)
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=True, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(rotation=90, axis="x")
    ax.set_xticklabels([])

    # KR Cycle
    ax = plt.subplot2grid((3, 2), (2, 1))
    ax.plot(results["KR"][method]['cycle'])
    ax.axhline(0, color='black', lw=0.5)
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(rotation=90, axis="x")

    plt.tight_layout()
    plt.savefig(save_path.joinpath(f'Q01 a Quarterly GDP {method}.pdf'))
    plt.show()
    plt.close()
