from utils import AA_LECTURE
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from scipy.optimize import minimize, Bounds
from utils import Performance


# =====================
# ===== READ DATA =====
# =====================
trackers = pd.read_excel(AA_LECTURE.joinpath('Commodity Futures.xlsx'),
                         sheet_name='Trackers',
                         index_col=0)
trackers.index = pd.to_datetime(trackers.index)

carry = pd.read_excel(AA_LECTURE.joinpath('Commodity Futures.xlsx'),
                      sheet_name='Carry',
                      index_col=0)
carry.index = pd.to_datetime(carry.index)

vols = trackers.pct_change(1).ewm(com=252, min_periods=252).std() * np.sqrt(252)
covar = trackers.pct_change(1).ewm(com=252, min_periods=252).cov() * 252


# =======================
# ===== FAMA-FRENCH =====
# =======================
# TODO turn this into a function/class
def fama_french(signal):
    signal = signal.sort_values()
    size = int(signal.shape[0] * 0.2)
    short = signal.iloc[:size].index
    long = signal.iloc[-size:].index
    w = pd.Series(data=0, index=signal.index)
    w.loc[short] = - 1 / len(short)
    w.loc[long] = 1 / len(long)
    return w


start_date = carry.index[0]
notional_start = 100
weights = fama_french(carry.loc[start_date])
holdings = notional_start * weights / trackers.loc[start_date]
backtest_ff = pd.Series(name='Backtest')
backtest_ff.loc[start_date] = notional_start
next_roll_date = start_date + pd.offsets.DateOffset(months=3)

dates2loop = zip(carry.index[1:], carry.index[:-1])
for d, dm1 in tqdm(dates2loop, "Fama-French Style"):
    pnl = (trackers.diff(1).loc[d] * holdings).sum()
    backtest_ff.loc[d] = backtest_ff.loc[dm1] + pnl

    if d >= next_roll_date:
        weights = fama_french(carry.loc[dm1])
        holdings = backtest_ff.loc[d] * weights / trackers.loc[d]


# =========================
# ===== RANK WEIGHTED =====
# =========================
# TODO turn this into a function/class
def rank_weighted(signal):
    signal = signal.sort_values()
    c = (signal.rank() - signal.rank().mean()).abs().sum() / 2
    w = (signal.rank() - signal.rank().mean()) * (1 / c)
    return w


start_date = carry.index[0]
notional_start = 100
weights = rank_weighted(carry.loc[start_date])
holdings = notional_start * weights / trackers.loc[start_date]
backtest_rank = pd.Series(name='Backtest')
backtest_rank.loc[start_date] = notional_start
next_roll_date = start_date + pd.offsets.DateOffset(months=3)

dates2loop = zip(carry.index[1:], carry.index[:-1])
for d, dm1 in tqdm(dates2loop, "Rank Weighted"):
    pnl = (trackers.diff(1).loc[d] * holdings).sum()
    backtest_rank.loc[d] = backtest_rank.loc[dm1] + pnl

    if d >= next_roll_date:
        weights = rank_weighted(carry.loc[dm1])
        holdings = backtest_rank.loc[d] * weights / trackers.loc[d]


# ===========================
# ===== Z-SCORE WEIGHTS =====
# ===========================
# TODO turn this into a function/class
def zscore_weights(signal, winsor=False):
    w = (signal - signal.mean()) / signal.std()
    w[w >= 0] = w[w >= 0] / w[w >= 0].sum()
    w[w < 0] = - w[w < 0] / w[w < 0].sum()

    if winsor:
        w = pd.Series(data=winsorize(w), index=w.index)
    return w


start_date = carry.index[0]
notional_start = 100
weights_nowin = zscore_weights(carry.loc[start_date])
weights_win = zscore_weights(carry.loc[start_date], winsor=True)
holdings_nowin = notional_start * weights_nowin / trackers.loc[start_date]
holdings_win = notional_start * weights_win / trackers.loc[start_date]
backtest_zscore_nowin = pd.Series(name='Backtest')
backtest_zscore_win = pd.Series(name='Backtest')
backtest_zscore_nowin.loc[start_date] = notional_start
backtest_zscore_win.loc[start_date] = notional_start
next_roll_date = start_date + pd.offsets.DateOffset(months=3)

dates2loop = zip(carry.index[1:], carry.index[:-1])
for d, dm1 in tqdm(dates2loop, "Z-Score"):
    pnl_nowin = (trackers.diff(1).loc[d] * holdings_nowin).sum()
    pnl_win = (trackers.diff(1).loc[d] * holdings_win).sum()
    backtest_zscore_nowin.loc[d] = backtest_zscore_nowin.loc[dm1] + pnl_nowin
    backtest_zscore_win.loc[d] = backtest_zscore_win.loc[dm1] + pnl_win

    if d >= next_roll_date:
        weights_nowin = zscore_weights(carry.loc[dm1])
        weights_win = zscore_weights(carry.loc[dm1], winsor=True)
        holdings_nowin = backtest_zscore_nowin.loc[d] * weights_nowin / trackers.loc[d]
        holdings_win = backtest_zscore_win.loc[d] * weights_win / trackers.loc[d]


# ===================================
# ===== Direction + Inverse Vol =====
# ===================================
# TODO turn this into a function/class
def direction_invvol(signal, vol):
    invvol = 1 / vol
    w = np.sign(signal) * (invvol / invvol.sum())
    w[w >= 0] = w[w >= 0] / w[w >= 0].sum()
    w[w < 0] = - w[w < 0] / w[w < 0].sum()
    return w


start_date = carry.index[0]
notional_start = 100
weights = direction_invvol(carry.loc[start_date], vols.loc[start_date])
holdings = notional_start * weights / trackers.loc[start_date]
backtest_div = pd.Series(name='Backtest')
backtest_div.loc[start_date] = notional_start
next_roll_date = start_date + pd.offsets.DateOffset(months=3)

dates2loop = zip(carry.index[1:], carry.index[:-1])
for d, dm1 in tqdm(dates2loop, "Directional + InvVol"):
    pnl = (trackers.diff(1).loc[d] * holdings).sum()
    backtest_div.loc[d] = backtest_div.loc[dm1] + pnl

    if d >= next_roll_date:
        weights = direction_invvol(carry.loc[dm1], vols.loc[dm1])
        holdings = backtest_div.loc[d] * weights / trackers.loc[d]


# ============================
# ===== Maximum Exposure =====
# ============================
# TODO turn this into a function/class
def max_exposure(signal, covariance, c=1, target_vol=0.1):

    constraints = ({'type': 'eq',
                    'fun': lambda x: np.sqrt(x.T @ covariance @ x) - target_vol})

    bounds = Bounds(- c * np.ones(signal.shape[0]), c * np.ones(signal.shape[0]))


    # Run optimization
    res = minimize(lambda x: x.T @ signal,
                   np.zeros(signal.shape[0]),
                   method='SLSQP',
                   constraints=constraints,
                   bounds=bounds,
                   options={'ftol': 1e-9, 'disp': False})
    w = pd.Series(data=res.x, index=signal.index)
    return w


start_date = carry.index[0]
notional_start = 100
weights = max_exposure(carry.loc[start_date], covar.loc[start_date])
holdings = notional_start * weights / trackers.loc[start_date]
backtest_maxexp = pd.Series(name='Backtest')
backtest_maxexp.loc[start_date] = notional_start
next_roll_date = start_date + pd.offsets.DateOffset(months=3)

dates2loop = zip(carry.index[1:], carry.index[:-1])
for d, dm1 in tqdm(dates2loop, "Maximum Exposure"):
    pnl = (trackers.diff(1).loc[d] * holdings).sum()
    backtest_maxexp.loc[d] = backtest_maxexp.loc[dm1] + pnl

    if d >= next_roll_date:
        weights = max_exposure(carry.loc[dm1], covar.loc[dm1])
        holdings = backtest_maxexp.loc[d] * weights / trackers.loc[d]


# ====================
# ===== PLOT ALL =====
# ====================
backtest = pd.concat(
    [
        backtest_ff.rename('Fama-French'),
        backtest_rank.rename('Rank Weighted'),
        backtest_zscore_nowin.rename('Z-Score'),
        backtest_zscore_win.rename('Z-Score (Winsorized)'),
        backtest_div.rename('Direction + InvVol'),
        backtest_maxexp.rename('Maximum Exposure'),
    ],
    axis=1,
)

perf = Performance(backtest)
print(perf.table)

backtest.plot(legend=True, grid=True)
plt.tight_layout()
plt.show()
