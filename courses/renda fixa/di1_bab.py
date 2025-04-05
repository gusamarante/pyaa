from utils import Performance
from data import trackers_di
import numpy as np
import matplotlib.pyplot as plt
from utils import BLUE, RF_LECTURE
from plottable import ColDef, Table
import pandas as pd
from pandas.tseries.offsets import BDay
from tqdm import tqdm
from scipy.optimize import minimize

size = 5
vol_window = 252
rebal_window = 5
index_start = 100
target_vol = 0.1
short_end = "DI 1y"
long_end = "DI 10y"
benchmark = "DI 5y"
tc_param = 0

# Grab data and compute performance
trackers = trackers_di()
df_pnl = trackers.diff()
ret = trackers.pct_change(1)
vol = trackers.pct_change(1).rolling(vol_window).std().dropna() * np.sqrt(252)


# ===============================
# ===== BUILD THE BAB INDEX =====
# ===============================
df_bt = pd.DataFrame()

# First Date
t0 = vol.index[0]
df_bt.loc[t0, "Quantity Short"] = (target_vol * index_start) / (trackers.loc[t0, short_end] * vol.loc[t0, short_end])
df_bt.loc[t0, "Quantity Long"] = - (target_vol * index_start) / (trackers.loc[t0, long_end] * vol.loc[t0, long_end])
df_bt.loc[t0, "Index"] = index_start

next_rebal = t0 + BDay(rebal_window)

tc = 0
for d, dm1 in tqdm(zip(vol.index[1:], vol.index[:-1])):
    pnl = df_bt.loc[dm1, "Quantity Short"] * df_pnl.loc[d, short_end] + df_bt.loc[dm1, "Quantity Long"] * df_pnl.loc[d, long_end] - tc
    df_bt.loc[d, "Index"] = df_bt.loc[dm1, "Index"] + pnl

    if d >= next_rebal:
        df_bt.loc[d, "Quantity Short"] = (target_vol * df_bt.loc[d, "Index"]) / (trackers.loc[d, short_end] * vol.loc[dm1, short_end])
        df_bt.loc[d, "Quantity Long"] = - (target_vol * df_bt.loc[d, "Index"]) / (trackers.loc[d, long_end] * vol.loc[dm1, long_end])
    else:
        df_bt.loc[d, "Quantity Short"] = df_bt.loc[dm1, "Quantity Short"]
        df_bt.loc[d, "Quantity Long"] = df_bt.loc[dm1, "Quantity Long"]

    tc = tc_param * (abs(df_bt.loc[d, "Quantity Short"] - df_bt.loc[dm1, "Quantity Short"]) * vol.loc[d, short_end] * np.sqrt(1/252) + abs(df_bt.loc[d, "Quantity Long"] - df_bt.loc[dm1, "Quantity Long"]) * vol.loc[d, long_end] * np.sqrt(1/252))

# Performance
df_perf = pd.concat([trackers[benchmark], df_bt["Index"].rename("BAB Index")], axis=1).dropna()
perf = Performance(df_perf, skip_dd=True)
print(perf.table)

# Chart
# df_perf.plot()
# plt.show()

# =====================================
# ===== COMBINE BAB AND BENCHMARK =====
# =====================================
# def combine(cov, tvol):
#     w0 = np.array([1, 0])
#     cons = [{'type': 'eq', 'fun': lambda x: x.sum() - 1}]
#     objf = lambda x: np.sqrt(x @ cov @ x) - target_vol
#     res = minimize(objf, w0, bounds=[(0, 1), (0, 1)], constraints=cons)
#     w = pd.Series(index=cov.index, data=res.x)
#     return w
#
# covar = df_perf.pct_change(1).rolling(vol_window).cov().dropna() * 252
# dates2loop = covar.index.get_level_values(0).unique()
#
# # First date
# t0 = dates2loop[0]
# weights = combine(covar.loc[t0], target_vol)
#
# for d, dm1 in tqdm(zip(dates2loop[1:], dates2loop[:-1])):
#     pass