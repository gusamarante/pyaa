"""
Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
F8E16C - Yellow
F25F5C - Red
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from bwbbgdl import GoGet
from bwmktdata import Macrobond
from bwsecrets.api import get_secret

size = 5

# ===== Read the Data =====
# BBG
gg = GoGet()
bbg_tickers = {
    "SPX Index": "S&P500",
    "SXXP Index": "EuroStoxx 600",
    "TPX Index": "Topix",
    "IBOV Index": "Ibovespa",
}
df = gg.fetch(bbg_tickers, "PX_LAST", fmt="pivot", pivot_by="id")
df = df.loc["PX_LAST"]
df = df.rename(bbg_tickers, axis=1)

# Macrobond
passwords = get_secret("macrobond")
mb = Macrobond(client_id=passwords["client_id"], client_secret=passwords["client_secret"])
mb_tickers = {
    "usrate0190": "FFER",
    "eurate0003": "ECB Deposit Effective Rate",
}

df_mb = mb.fetch_series(mb_tickers)

# mbrec_tickers = {
#     "brlead1000": "Brazil",
#     "eulead1000": "Euro Area",
#     "jplead1000": "Japan",
#     "uslead1000": "United States",
# }
# df_rec = mb.fetch_unified_series(mbrec_tickers, frequency="quarterly")
# df_rec = df_rec.resample("Q").last()

# ===== US =====
ffr = df_mb['FFER'].dropna().copy()
ffr = (1 + ffr / 100) ** (1 / 252) - 1
ffrtr = (1+ffr).cumprod()

spx = df['S&P500'].dropna().copy()
ret_spx = spx.pct_change(1)
xret_spx = (ret_spx - ffr).dropna()
eri_spx = (1 + xret_spx).cumprod()

# TODO add cpi
us_all = pd.concat([spx, ffrtr], axis=1)
us_all = us_all.fillna(method='ffill').dropna()
us_all = us_all / us_all.iloc[0]

# ===== Europe =====
ecb_rate = df_mb['ECB Deposit Effective Rate'].dropna().copy()
ecb_rate = (1 + ecb_rate / 100) ** (1 / 252) - 1
ecbtr = (1+ecb_rate).cumprod()

sxxp = df['EuroStoxx 600'].dropna().copy()
ret_sxxp = sxxp.pct_change(1)
xret_sxxp = (ret_sxxp - ecb_rate).dropna()
eri_sxxp = (1 + xret_sxxp).cumprod()

# TODO add cpi
eu_all = pd.concat([sxxp, ecbtr], axis=1)
eu_all = eu_all.fillna(method='ffill').dropna()
eu_all = eu_all / eu_all.iloc[0]

# ==============================================
# ===== Chart - Separate Countries US / EU =====
# ==============================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))
# fig.suptitle(
#     "Realized Equity Premium",
#     fontsize=16,
#     fontweight="bold",
# )
# US
ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("United States")
ax.plot(us_all['S&P500'], label="S&P 500 Total Return", color="#3333B2")
ax.plot(us_all['FFER'], label="Cumulative Fed Funds Effective Rate", color="#F25F5C")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_yscale('log')
ax.set_ylabel("Index (log-scale)")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
# locators = mdates.YearLocator()
# ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

# EU
ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("Europe")
ax.plot(eu_all['EuroStoxx 600'], label="EuroStoxx 600 Total Return", color="#3333B2")
ax.plot(eu_all['ECB Deposit Effective Rate'], label="Cumulative ECB Deposit Effective Rate", color="#F25F5C")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_yscale('log')
ax.set_ylabel("Index (log-scale)")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.savefig(r"C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Figures/Equities - Historical ERP - US EU.pdf")
plt.show()
plt.close()
