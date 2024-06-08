import numpy as np
import pandas as pd
from data.data_api import SGS
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from allocation import HRP
from utils import Performance
from pathlib import Path
from getpass import getuser
from tqdm import tqdm


# ================
# ===== DATA =====
# ================
# Read the Names
names = pd.read_excel(f"/Users/{getuser()}/Dropbox/Personal Portfolio/data/Gestores copy.xlsx",
                      sheet_name='Tickers', index_col=0)
names = names['Fund Name']
for term in [' FIC ', ' FIM ', ' FIM', ' MULT', ' MUL', ' FI ', ' MULT ', ' LP ']:
    names = names.str.replace(term, ' ')


# Read Managers
df = pd.read_excel(f"/Users/{getuser()}/Dropbox/Personal Portfolio/data/Gestores copy.xlsx",
                   sheet_name='BBG', skiprows=3, index_col=0)
df.index = pd.to_datetime(df.index)
df = df.rename(names, axis=1)
df = df.dropna(how='all').ffill()

# CDI
cdi = pd.read_excel(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation/Dados BBG AA Course.xlsx",
                    sheet_name='PX_LAST', skiprows=3, index_col=0)
cdi.index = pd.to_datetime(cdi.index)
cdi = cdi["BZDIOVER Index"] / 100

# Excess Returns of the Hedge Funds
df_xr = []
for fund in df.columns:
    ret = df[fund].pct_change(1).dropna()
    ret = ret - cdi
    df_xr.append(ret.rename(fund))

df_xr = pd.concat(df_xr, axis=1)
df_funds = (1 + df_xr).cumprod()
df_funds = df_funds / df_funds.bfill().iloc[0]
df_funds = df_funds.dropna(how='all')

df_funds.to_clipboard()
df_funds = df_funds.drop(['KP WEALTH CP 35', 'CAPSTONE MACRO '], axis=1)

# All cumulative ERI
df_cum = []
for fund in df_funds.columns:
    df_cum.append(df_funds[fund].dropna().reset_index(drop=True))

df_cum = pd.concat(df_cum, axis=1)
df_cum.index = df_cum.index / 252

# --- NTNB ---
filepath = f'/Users/{getuser()}/PycharmProjects/pyaa/trackers/output data/trackers_ntnb.xlsx'  # mac
ntnb = pd.read_excel(filepath, index_col=0)
ntnb = ntnb['NTNB 20y'].dropna()

# --- NTNF ---
filepath = f'/Users/{getuser()}/PycharmProjects/pyaa/trackers/output data/trackers_ntnf.xlsx'  # mac
ntnf = pd.read_excel(filepath, index_col=0)
ntnf = ntnf['NTNF 7y'].dropna()

# --- IVVB ---
filepath = f'/Users/{getuser()}/Dropbox/Personal Portfolio/data/ETFs.xlsx'  # mac
ivvb = pd.read_excel(filepath, index_col=0, sheet_name='values')
ivvb = ivvb['IVVB11 BZ Equity'].rename('IVVB').dropna()

# --- IDA ---
filepath = f'/Users/{getuser()}/Dropbox/Personal Portfolio/data/IDA Anbima.xlsx'  # mac
ida = pd.read_excel(filepath, index_col=0, sheet_name='Sheet2')
ida = ida['IDADIPCA Index'].rename('IDA').dropna()


df_assets = pd.concat([ntnf, ntnb, ivvb, ida], axis=1).dropna(how='all')


# ====================
# ===== Backtest =====
# ====================
backtest = pd.Series(name='Backtest')
# df_vols = df_assets.pct_change(21).rolling(252).std() * np.sqrt(12)  # Traditional
df_vols = df_assets.pct_change(21).ewm(com=252).std() * np.sqrt(12)  # EWM
notional_start = 1

# We compute weights every day beacause it is easy, but we are only going to use them on rebalance dates.
inv_vol_weights = (1/df_vols).div((1/df_vols).sum(axis=1), axis=0)  # Traditional
inv_vol_weights = inv_vol_weights.dropna(how='all')

# Initial position
start_date = inv_vol_weights.index[0]

holdings = pd.DataFrame(columns=df_assets.columns)
holdings.loc[start_date] = (inv_vol_weights.loc[start_date] * notional_start) / df_assets.loc[start_date]

backtest.loc[start_date] = notional_start
next_rebal = start_date + pd.offsets.DateOffset(months=3)

dates2loop = zip(inv_vol_weights.index[1:], inv_vol_weights.index[:-1])
for d, dm1 in tqdm(dates2loop, "Backtesting"):
    pnl = (df_assets.diff(1).loc[d] * holdings.loc[dm1]).sum()
    backtest.loc[d] = backtest.loc[dm1] + pnl

    if d >= next_rebal:
        holdings.loc[d] = (inv_vol_weights.loc[dm1] * backtest.loc[d]) / df_assets.loc[d]
        next_rebal = d + pd.offsets.DateOffset(months=3)
    else:
        holdings.loc[d] = holdings.loc[dm1]


backtest = (1 + backtest.pct_change(1) - cdi)
backtest.loc[start_date] = notional_start
backtest = backtest.dropna().cumprod()

backtest.plot()
plt.show()

holdings.plot()
plt.show()

perf_port = Performance(backtest.to_frame('Simple'), skip_dd=True)
print(perf_port.table)


# ==============================
# ===== Performance of HFs =====
# ==============================
perf = Performance(df_funds, skip_dd=True)
correls = df_funds.pct_change(21).corrwith(backtest.pct_change(21)).sort_values(ascending=False)

hr = df_funds.resample('M').last().pct_change(1).sub(backtest.resample('M').last().pct_change(1), axis=0)
hr = (hr > 0).where(~hr.isna())
# (hr.sum(axis=1).rolling(12).sum() / hr.count(axis=1).rolling(12).sum()).plot()


# =================================================
# ===== Chart - Cumulative ERI of Hedge Funds =====
# =================================================
fig = plt.figure(figsize=(7 * (16 / 7.3), 7))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Brazilian Hedge Funds - Excess Return Index")
ax.plot(df_cum)
ax.axhline(1, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_yscale('log')
ax.set_ylabel("Index (log-scale)")
ax.set_xlabel("Years since start of series")
ax.get_yaxis().set_major_formatter(ScalarFormatter())

plt.tight_layout()

plt.show()
plt.close()


# =================================================
# ===== Chart - Cumulative ERI of Hedge Funds =====
# =================================================
fig = plt.figure(figsize=(7 * (16 / 7.3), 7))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Brazilian Hedge Funds - Excess Return Index")
ax.plot(df_funds, alpha=0.5)
ax.plot(backtest, color='black', lw=3)
ax.axhline(1, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_yscale('log')
ax.set_ylabel("Index (log-scale)")
ax.set_xlabel("Years since start of series")
ax.get_yaxis().set_major_formatter(ScalarFormatter())

plt.tight_layout()

plt.show()
plt.close()


# =====================================================
# ===== Chart - Cumulative ERI of Simple Strategy =====
# =====================================================
fig = plt.figure(figsize=(7 * (16 / 7.3), 7))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Simple Strategy VS Brazilian Hedge Funds - Excess Return Index")
ax.plot(df_cum, color='lightgrey', alpha=0.7, label=None)

bt2plot = backtest.dropna().reset_index(drop=True)
bt2plot.index = bt2plot.index / 252
ax.plot(bt2plot, color="tab:blue", lw=2, label="Simple Strategy Backtest")
ax.axhline(1, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_yscale('log')
ax.set_ylabel("Index (log-scale)")
ax.set_xlabel("Years since start of series")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.legend(frameon=True, loc='best')

plt.tight_layout()

plt.show()
plt.close()


# ============================================
# ===== Chart - Correlation + Dendrogram =====
# ============================================
cov = df_funds.dropna().pct_change(21).cov()
hrp = HRP(cov)
hrp.plot_dendrogram()
hrp.plot_corr_matrix()

# TODO percentual de HFs de quem o portfolio ganha
# TODO Percentual de anos que um HF ganha do Fundo
