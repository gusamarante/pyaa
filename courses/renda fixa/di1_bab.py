"""
- Montar o BaB em cima do notional
- Montar o 5y em cima do Notional
_ Performance dos componentes e do conjunto
"""

from utils import Performance
from data import trackers_di
import numpy as np
import matplotlib.pyplot as plt
from utils import BLUE, RF_LECTURE
from plottable import ColDef, Table
import pandas as pd
from pandas.tseries.offsets import BDay
from tqdm import tqdm

size = 5
vol_window = 252
rebal_window = 21
index_start = 100
target_vol = 0.1
short_end = "DI 1y"
long_end = "DI 10y"

# Grab data and compute performance
trackers = trackers_di()
df_pnl = trackers.diff()
ret = trackers.pct_change(1)
vol = trackers.pct_change(1).rolling(vol_window).std().dropna() * np.sqrt(252)


df_bt = pd.DataFrame()

# First Date
t0 = vol.index[0]
df_bt.loc[t0, "Quantity Short"] = (target_vol * index_start) / (trackers.loc[t0, short_end] * vol.loc[t0, short_end])
df_bt.loc[t0, "Quantity Long"] = - (target_vol * index_start) / (trackers.loc[t0, long_end] * vol.loc[t0, long_end])
df_bt.loc[t0, "Index"] = index_start

next_rebal = t0 + BDay(rebal_window)


for d, dm1 in tqdm(zip(vol.index[1:], vol.index[:-1])):
    pnl = df_bt.loc[dm1, "Quantity Short"] * df_pnl.loc[d, short_end] + df_bt.loc[dm1, "Quantity Long"] * df_pnl.loc[d, long_end]
    df_bt.loc[d, "Index"] = df_bt.loc[dm1, "Index"] + pnl

    if d >= next_rebal:
        df_bt.loc[d, "Quantity Short"] = (target_vol * df_bt.loc[d, "Index"]) / (trackers.loc[d, short_end] * vol.loc[d, short_end])
        df_bt.loc[d, "Quantity Long"] = - (target_vol * df_bt.loc[d, "Index"]) / (trackers.loc[d, long_end] * vol.loc[d, long_end])
    else:
        df_bt.loc[d, "Quantity Short"] = df_bt.loc[dm1, "Quantity Short"]
        df_bt.loc[d, "Quantity Long"] = df_bt.loc[dm1, "Quantity Long"]


# Performance
perf = Performance(df_bt[["Index"]])
print(perf.table)

# Chart
df_bt["Index"].plot()
plt.show()
