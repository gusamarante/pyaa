"""
Recession identification
"""
import getpass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter

from models import fihp

# User Parameters
size = 5
y_lim = 0.5

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
data_wb = data_wb.resample("QE").last()
gdp_us = data_wb['US GDP'].dropna()
gdp_kr = data_wb['KR GDP'].dropna()

# Recessions
data_rec = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Problem Set 01/PS1 Data Clean.xlsx',
    index_col=0,
    sheet_name="Recession",
)
data_rec.index = pd.to_datetime(data_rec.index)
data_rec = data_rec.fillna(0).resample("QE").mean().fillna(0)
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
        "Linear": log_linear(gdp_us),
        "Quadratic": log_quadratic(gdp_us),
        "HP 1600": hp(gdp_us, 1600),
        "FIHP 1600": fcast_incr_hp(gdp_us, 1600),
    },
    "KR": {
        "Linear": log_linear(gdp_kr),
        "Quadratic": log_quadratic(gdp_kr),
        "HP 1600": hp(gdp_kr, 1600),
        "FIHP 1600": fcast_incr_hp(gdp_kr, 1600),
    },
}


# In Sample Exercise
mindex = pd.MultiIndex.from_tuples([], names=["Country", "Method"])
df_corr = pd.DataFrame(index=mindex)
for country in results.keys():
    for method in results[country].keys():
        for lag in range(8):
            df_corr.loc[(country, method), lag] = results[country][method]['cycle'].shift(lag).corr(data_rec[country])



# ===== CHART =====
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((2, 2), (0, 0))
method = "Linear"
data2plot = df_corr.xs(method, level=1)
ax.plot(data2plot.loc["US"], label="US")
ax.plot(data2plot.loc["KR"], label="KR")
ax.axhline(0, color="black", lw=0.5)
ax.set_title(method)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="upper left")
ax.set_ylabel("Correlation")
ax.set_ylim(-y_lim, y_lim)

ax = plt.subplot2grid((2, 2), (0, 1))
method = "Quadratic"
data2plot = df_corr.xs(method, level=1)
ax.plot(data2plot.loc["US"], label="US")
ax.plot(data2plot.loc["KR"], label="KR")
ax.axhline(0, color="black", lw=0.5)
ax.set_title(method)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="upper left")
ax.set_ylim(-y_lim, y_lim)

ax = plt.subplot2grid((2, 2), (1, 0))
method = "HP 1600"
data2plot = df_corr.xs(method, level=1)
ax.plot(data2plot.loc["US"], label="US")
ax.plot(data2plot.loc["KR"], label="KR")
ax.axhline(0, color="black", lw=0.5)
ax.set_title(method)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="upper left")
ax.set_ylabel("Correlation")
ax.set_ylim(-y_lim, y_lim)
ax.set_xlabel("Lag in Cycle Estimate (Quarters)")

ax = plt.subplot2grid((2, 2), (1, 1))
method = "FIHP 1600"
data2plot = df_corr.xs(method, level=1)
ax.plot(data2plot.loc["US"], label="US")
ax.plot(data2plot.loc["KR"], label="KR")
ax.axhline(0, color="black", lw=0.5)
ax.set_title(method)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="upper left")
ax.set_ylim(-y_lim, y_lim)
ax.set_xlabel("Lag in Cycle Estimate (Quarters)")

plt.tight_layout()
plt.savefig(save_path.joinpath(f'Q01 c Correlation with recessions.pdf'))
plt.show()
plt.close()

