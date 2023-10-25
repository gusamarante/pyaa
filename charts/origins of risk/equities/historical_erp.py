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

from bwbbgdl import GoGet
from bwmktdata import Macrobond
from bwsecrets.api import get_secret

from utils import Performance

size = 5

# ===== Read the Data =====
file_path = r"C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Dados BBG AA Course.xlsx"
df = pd.read_excel(file_path, sheet_name='TOT_RETURN_INDEX_GROSS_DVDS', skiprows=4, index_col=0)
df = df.sort_index()
df = df.dropna(how='all')

rename_tickers = {
    "SPX Index": "S&P500",
    "SXXP Index": "EuroStoxx 600",
    "TPX Index": "Topix",
    "IBOV Index": "Ibovespa",
}
df = df.rename(rename_tickers, axis=1)

# Macrobond
passwords = get_secret("macrobond")
mb = Macrobond(client_id=passwords["client_id"], client_secret=passwords["client_secret"])
mb_tickers = {
    "usrate0190": "FFER",
    "eurate0003": "ECB Deposit Effective Rate",
    "jprate0008": "Japan Discount Rate",
    "brrate0003": "CDI",
}

df_mb = mb.fetch_series(mb_tickers)

# ===== US =====
ffr = df_mb['FFER'].dropna().copy()
ffr = (1 + ffr / 100) ** (1 / 252) - 1
ffrtr = (1+ffr).cumprod()

spx = df['S&P500'].dropna().copy()
ret_spx = spx.pct_change(1)
xret_spx = ((1 + ret_spx) / (1 + ffr) - 1).dropna()
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
xret_sxxp = ((1 + ret_sxxp) / (1 + ecb_rate) - 1).dropna()
eri_sxxp = (1 + xret_sxxp).cumprod()

# TODO add cpi
eu_all = pd.concat([sxxp, ecbtr], axis=1)
eu_all = eu_all.fillna(method='ffill').dropna()
eu_all = eu_all / eu_all.iloc[0]

# ===== Japan =====
jpn_rate = df_mb['Japan Discount Rate'].dropna().copy()
jpn_rate = (1 + jpn_rate / 100) ** (1 / 252) - 1
jpntr = (1 + jpn_rate).cumprod()

topix = df['Topix'].dropna().copy()
ret_topix = topix.pct_change(1)
xret_topix = ((1 + ret_topix) / (1 + jpn_rate) - 1).dropna()
eri_topix = (1 + xret_topix).cumprod()

# TODO add cpi
jp_all = pd.concat([topix, jpntr], axis=1)
jp_all = jp_all.fillna(method='ffill').dropna()
jp_all = jp_all / jp_all.iloc[0]

# ===== Brazil =====
cdi_rate = df_mb['CDI'].dropna().copy()
cdi_rate = cdi_rate / 100
cditr = (1 + cdi_rate).cumprod()

ibov = df['Ibovespa'].dropna().copy()
ret_ibov = ibov.pct_change(1)
xret_ibov = ((1 + ret_ibov) / (1 + cdi_rate) - 1).dropna()
xret_ibov = xret_ibov[xret_ibov.index >= "1995-01-01"]
eri_ibov = (1 + xret_ibov).cumprod()

# TODO add cpi
br_all = pd.concat([ibov, cditr], axis=1)
br_all = br_all.fillna(method='ffill').dropna()
br_all = br_all[br_all.index >= "1995-01-01"]
br_all = br_all / br_all.iloc[0]


# All ERI
all_eri = pd.concat([
    eri_spx.rename("S&P 500"),
    eri_sxxp.rename("EuroStoxx 600"),
    eri_topix.rename("Topix"),
    eri_ibov.rename("Ibovespa"),
        ], axis=1)

# Performance
# perf = Performance(all_eri)
# perf.table.to_clipboard()
# print("COPIED")

# All cumulative ERI
cumulative_eri = pd.concat([
    eri_spx.reset_index(drop=True).rename("S&P 500"),
    eri_sxxp.reset_index(drop=True).rename("EuroStoxx 600"),
    eri_topix.reset_index(drop=True).rename("Topix"),
    eri_ibov.reset_index(drop=True).rename("Ibovespa"),
        ], axis=1)
cumulative_eri.index = cumulative_eri.index / 252
cumulative_eri = cumulative_eri / cumulative_eri.iloc[0]

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


# ==============================================
# ===== Chart - Separate Countries US / EU =====
# ==============================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))
# fig.suptitle(
#     "Realized Equity Premium",
#     fontsize=16,
#     fontweight="bold",
# )
# Japan
ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Japan")
ax.plot(jp_all['Topix'], label="Topix Total Return", color="#3333B2")
ax.plot(jp_all['Japan Discount Rate'], label="Cumulative Japan Discount Rate", color="#F25F5C")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_yscale('log')
ax.set_ylabel("Index (log-scale)")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

# Brazil
ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("Brazil")
ax.plot(br_all['Ibovespa'], label="Ibovespa Total Return", color="#3333B2")
ax.plot(br_all['CDI'], label="Cumulative CDI Rate", color="#F25F5C")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_yscale('log')
ax.set_ylabel("Index (log-scale)")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.savefig(r"C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Figures/Equities - Historical ERP - JP BR.pdf")
plt.show()
plt.close()


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
ax.set_title("Excess Return Index")
ax.plot(cumulative_eri['S&P 500'], label="S&P 500", color="#3333B2")
ax.plot(cumulative_eri['EuroStoxx 600'], label="EuroStoxx 600", color="#0B6E4F")
ax.plot(cumulative_eri['Topix'], label="Topix", color="#FFBA08")
ax.plot(cumulative_eri['Ibovespa'], label="Ibovespa", color="#F25F5C")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_yscale('log')
ax.set_ylabel("Index (log-scale)")
ax.set_xlabel("Years since start of series")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.savefig(r"C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Figures/Equities - Historical ERP - All ERI.pdf")
plt.show()
plt.close()
