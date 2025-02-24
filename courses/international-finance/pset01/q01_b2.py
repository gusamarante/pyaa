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


cycle_series = {
    "US": {
        "GDP": hp(np.log(data_wb['US GDP'].dropna()), 100),
        "Consumption": hp(np.log(data_wb['US Cons'].dropna()), 100),
        "Investment": hp(np.log(data_wb['US Inv'].dropna()), 100),
        "Government Spending": hp(np.log(data_wb['US Gov'].dropna()), 100),
        "Exports": hp(np.log(data_wb['US Exp'].dropna()), 100),
        "Imports": hp(np.log(data_wb['US Imp'].dropna()), 100),
        "Trade Balance (% of GDP)": ((data_wb['US Exp'] - data_wb['US Imp']) / data_wb['US Imp']).dropna(),  # TODO why is this not working?
    },
    "KR": {
        "GDP": "",
        "Consumption": "",
        "Investment": "",
        "Government Spending": "",
        "Exports": "",
        "Imports": "",
        "Trade Balance (% of GDP)": "",
    },
}



a = 1