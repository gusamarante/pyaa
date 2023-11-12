"""
Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
FFBA08 - Yellow
F25F5C - Red
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from statsmodels.tsa.filters.hp_filter import hpfilter
import numpy as np

from bwbbgdl import GoGet
from bwmktdata import Macrobond
from bwsecrets.api import get_secret

from utils import Performance

size = 5

# ===== Read the Data =====
# Excel
file_path = r"C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Dados BBG AA Course.xlsx"  # Work
df_dy = pd.read_excel(file_path, sheet_name='IDX_EST_DVD_YLD', skiprows=4, index_col=0)
df_dy = df_dy.dropna(how='all')
df_dy = df_dy.sort_index()
rename_tickers = {
    "SPX Index": "S&P500",
    "SXXP Index": "EuroStoxx 600",
    "TPX Index": "Topix",
    "IBOV Index": "Ibovespa",
}
df_dy = df_dy.rename(rename_tickers, axis=1)

# BBG
bbg_tickers = {
    "USGG10YR Index": "Nominal 10y US",
    "GJGB10 Index": "Nominal 10y Japan",
    "GDBR10 Index": "Nominal 10y Germany",
}

gg = GoGet()
df_bbg = gg.fetch(tickers=bbg_tickers, fields="PX_LAST", fmt="pivot", pivot_by="id")
df_bbg = df_bbg.loc["PX_LAST"]
df_bbg = df_bbg.rename(bbg_tickers, axis=1)

# Macrobond
passwords = get_secret("macrobond")
mb = Macrobond(client_id=passwords["client_id"], client_secret=passwords["client_secret"])
q_tickers = {
    "usnaac0169": "US GDP",
    "eueunaac0149": "Europe GDP",
    "jpnaac0004": "Japan GDP",
    "brnaac0016": "Brazil GDP",
}

df_q = mb.fetch_series(q_tickers)

df_hp = pd.DataFrame()
for col in df_q.columns:
    _, trend = hpfilter(df_q[col].dropna())
    df_hp = pd.concat([df_hp, trend], axis=1)

df_growth = df_hp.rolling(4).mean().pct_change(4)
df_growth.index = pd.to_datetime(df_growth.index)
df_growth = df_growth.resample('Q').last()
df_growth = df_growth.resample('D').last().fillna(method='ffill')
df_growth = df_growth.reindex(df_dy.index)
df_growth = df_growth.fillna(method='ffill')

# ===== Computations =====
# US
erp_us = df_dy['S&P500'] / 100 + df_growth['US GDP_trend'] - df_bbg['Nominal 10y US'] / 100

# EU
erp_eu = df_dy['EuroStoxx 600'] / 100 + df_growth['Europe GDP_trend'] - df_bbg['Nominal 10y Germany'] / 100

# JP
erp_jp = df_dy['Topix'] / 100 + df_growth['Japan GDP_trend'] - df_bbg['Nominal 10y Japan'] / 100

df_erp = pd.concat([erp_us.rename('US'),
                    erp_eu.rename('EU'),
                    erp_jp.rename('JP')], axis=1)
df_erp = 100 * df_erp


# ===========================================
# ===== Chart - Cumulative ERI Together =====
# ===========================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))
# fig.suptitle(
#     "Realized Equity Premium",
#     fontsize=16,
#     fontweight="bold",
# )
ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Ex ante ERP Estimate (div yield + potential growth - 10y rate)")
ax.plot(df_erp['US'], label="US", color="#3333B2")
ax.plot(df_erp['EU'], label="EU", color="#0B6E4F")
ax.plot(df_erp['JP'], label="JP", color="#F25F5C")
ax.axhline(0, color="black", lw=1)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_ylabel("Percent")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.savefig(r"C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Figures/Equities- Ex ante ERP Estimates.pdf")
plt.show()
plt.close()


