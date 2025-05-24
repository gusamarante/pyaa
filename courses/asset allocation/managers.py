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


dir_path = Path(f'/Users/{getuser()}/Dropbox/Lectures/MPE - Asset Allocation/2025')

# ================
# ===== DATA =====
# ================
# Read the Names
names = pd.read_excel(
    dir_path.joinpath("Gestores.xlsx"),
    sheet_name='Tickers',
    index_col=0,
)
names = names['Fund Name']
for term in [' FIC ', ' FIM ', ' FIM', ' MULT', ' MUL', ' FI ', ' MULT ', ' LP ']:
    names = names.str.replace(term, ' ')


# Read Managers
df = pd.read_excel(
    dir_path.joinpath(f"Gestores.xlsx"),
    sheet_name='BBG',
    skiprows=3,
    index_col=0,
)
df.index = pd.to_datetime(df.index)
df = df.rename(names, axis=1)
df = df.dropna(how='all').ffill()

# CDI
cdi = pd.read_excel(
    dir_path.joinpath("Dados BBG AA Course.xlsx"),
    sheet_name='PX_LAST',
    skiprows=3,
    index_col=0,
)
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


# ==============================
# ===== Performance of HFs =====
# ==============================
perf = Performance(df_funds, skip_dd=True)
# TODO make table



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


# ============================================
# ===== Chart - Correlation + Dendrogram =====
# ============================================
cov = df_funds.dropna().pct_change(21).cov()
hrp = HRP(cov)
hrp.plot_dendrogram()
hrp.plot_corr_matrix()
