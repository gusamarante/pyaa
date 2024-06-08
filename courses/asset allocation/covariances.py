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
from sklearn.covariance import LedoitWolf
from utils import cov2corr


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


df_assets = pd.concat([ntnf, ntnb, ivvb, ida], axis=1).dropna()

ret = df_assets.pct_change().dropna()

cov_full = ret.corr()
cov_expanding = ret.expanding().corr()
cov_rolling_1y = ret.rolling(window=252).corr()
cov_rolling_5y = ret.rolling(window=252 * 5).corr()
cov_emw_1y = ret.ewm(halflife=252).corr()
cov_emw_5y = ret.ewm(halflife=252 * 5).corr()

cov_lw = pd.Series()
for d in tqdm(ret.index[252:]):
    lw_aux = LedoitWolf().fit(ret.loc[:d].tail(252))
    cov_lw.loc[d] = cov2corr(lw_aux.covariance_)[0][1, 2]


fig = plt.figure(figsize=(7 * (16 / 7.3), 7))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Different Correlation Methdologies")
ax.axhline(cov_full.loc["IVVB", "NTNB 20y"], label="Full Sample")
ax.plot(cov_expanding.xs("IVVB", level=1)["NTNB 20y"], label="Expanding")
ax.plot(cov_rolling_1y.xs("IVVB", level=1)["NTNB 20y"], label="Rolling 1y")
ax.plot(cov_rolling_5y.xs("IVVB", level=1)["NTNB 20y"], label="Rolling 5y")
ax.plot(cov_emw_1y.xs("IVVB", level=1)["NTNB 20y"], label="EMW 1y")
ax.plot(cov_emw_5y.xs("IVVB", level=1)["NTNB 20y"], label="EMW 5y")
ax.plot(cov_lw, label="Ledoit-Wolf")
ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_ylabel("Correlations")
ax.legend(frameon=True, loc='best')
ax.set_ylim((-1, 1))

plt.tight_layout()

plt.show()
plt.close()
