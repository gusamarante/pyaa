import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# User Parameters
size = 5


# ===== READ DATA =====
# World Bank
data_wb = pd.read_excel(
    '/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - International Finance/Problem Set 01/PS1 Data Clean.xlsx',
    index_col=0,
)
data_wb.index = pd.to_datetime(data_wb.index)
data_wb = data_wb.resample("Y").last()


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

    return trend, cycle


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

    return trend, cycle


def hp(series, lamb):
    series = np.log(series)
    cycle, trend = hpfilter(series, lamb=lamb)
    return trend, cycle


# TODO Forecast HP Filter


# ===== RUN ALL METHODOLOGIES =====
for country in ['US', 'KR']:

    all_trends = pd.DataFrame()
    all_cycles = pd.DataFrame()

    t, c = log_linear(data_wb[f"{country} GDP"])
    all_trends = pd.concat([all_trends, t.rename('log-linear')], axis=1)
    all_cycles = pd.concat([all_cycles, c.rename('log-linear')], axis=1)

    t, c = log_quadratic(data_wb[f"{country} GDP"])
    all_trends = pd.concat([all_trends, t.rename('log-quadratic')], axis=1)
    all_cycles = pd.concat([all_cycles, c.rename('log-quadratic')], axis=1)

    t, c = hp(data_wb[f"{country} GDP"], 6.25)
    all_trends = pd.concat([all_trends, t.rename('HP 6.25')], axis=1)
    all_cycles = pd.concat([all_cycles, c.rename('HP 6.25')], axis=1)

    t, c = hp(data_wb[f"{country} GDP"], 100)
    all_trends = pd.concat([all_trends, t.rename('HP 100')], axis=1)
    all_cycles = pd.concat([all_cycles, c.rename('HP 100')], axis=1)

    growth = pd.concat([np.log(data_wb[f"{country} GDP"]), all_trends], axis=1).diff(1)


    # --- Chart ---
    fig = plt.figure(figsize=(size * (16 / 7.3), size))

    # Level of Trends
    ax = plt.subplot2grid((1, 2), (0, 0))
    ax.set_title(f"Trend Estimates for {country}")
    ax.plot(np.log(data_wb[f'{country} GDP']), label='log GDP', lw=3)
    ax.plot(all_trends, label=all_trends.columns)
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=True, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(rotation=90, axis="x")

    # Growth of Trends
    ax = plt.subplot2grid((1, 2), (0, 1))
    ax.set_title(f"Trend Growth for {country}")
    ax.plot(growth, label=growth.columns)
    ax.axhline(0, color="black", lw=0.5)
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=True, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(rotation=90, axis="x")

    plt.tight_layout()
    plt.show()
