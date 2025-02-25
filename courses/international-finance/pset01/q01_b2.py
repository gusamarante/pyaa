"""
Computes business cycle moments
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from models import fihp
import getpass
from pathlib import Path

# User Parameters
size = 5

username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Problem Set 01/figures')

# ===== READ DATA =====
# World Bank
data_wb = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Problem Set 01/PS1 Data Clean.xlsx',
    index_col=0,
    sheet_name="World Bank",
)
data_wb.index = pd.to_datetime(data_wb.index)
data_wb = data_wb.resample("Y").last()


def hp(series, lamb):
    cycle, _ = hpfilter(series, lamb=lamb)
    return cycle

def hp_neg(series, lamb):
    """
    This was necessary as statsmodels' HP filter does not work with negative
    values due to the sparse matrices it uses
    """
    nobs = len(series)
    I = np.eye(nobs)  # Identity matrix

    K = np.zeros((nobs - 2, nobs))
    for i in range(nobs - 2):
        K[i, i] = 1.
        K[i, i + 1] = -2.
        K[i, i + 2] = 1.

    trend = np.linalg.inv(np.eye(series.shape[0]) + lamb * K.T @ K) @ series
    cycle = series - trend
    return cycle.rename('cycle')


lamb_all = 6.25
cycle_series = {
    "US": {
        "GDP": hp(np.log(data_wb['US GDP'].dropna()), lamb_all),
        "Consumption": hp(np.log(data_wb['US Cons'].dropna()), lamb_all),
        "Investment": hp(np.log(data_wb['US Inv'].dropna()), lamb_all),
        "Government Spending": hp(np.log(data_wb['US Gov'].dropna()), lamb_all),
        "Exports": hp(np.log(data_wb['US Exp'].dropna()), lamb_all),
        "Imports": hp(np.log(data_wb['US Imp'].dropna()), lamb_all),
        "Trade Balance (% of GDP)": hp_neg(((data_wb['US Exp'] - data_wb['US Imp']) / data_wb['US Imp']).dropna(), lamb_all),
    },
    "KR": {
        "GDP": hp(np.log(data_wb['KR GDP'].dropna()), lamb_all),
        "Consumption": hp(np.log(data_wb['KR Cons'].dropna()), lamb_all),
        "Investment": hp(np.log(data_wb['KR Inv'].dropna()), lamb_all),
        "Government Spending": hp(np.log(data_wb['KR Gov'].dropna()), lamb_all),
        "Exports": hp(np.log(data_wb['KR Exp'].dropna()), lamb_all),
        "Imports": hp(np.log(data_wb['KR Imp'].dropna()), lamb_all),
        "Trade Balance (% of GDP)": hp_neg(((data_wb['KR Exp'] - data_wb['KR Imp']) / data_wb['KR Imp']).dropna(), lamb_all),
    },
}

df_sd = pd.DataFrame()
df_corr = pd.DataFrame()
df_acf = pd.DataFrame()
for country in cycle_series.keys():
    for series in cycle_series[country].keys():

        aux_cycle = cycle_series[country][series]
        df_sd.loc[series, country] = aux_cycle.std()
        df_corr.loc[series, country] = aux_cycle.corr(cycle_series[country]["GDP"])
        df_acf.loc[series, country] = aux_cycle.autocorr(1)



print(df_sd)
print(df_corr)
print(df_acf)

df_acf.to_clipboard()