"""
Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
FFBA08 - Yellow
F25F5C - Red
"""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from bwmktdata import Macrobond

from utils import AA_LECTURE

size = 5

# ===== Read the Data =====
file_path = AA_LECTURE.joinpath("Dados BBG AA Course.xlsx")
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
mb = Macrobond()
mb_tickers = {
    "usrate0190": "FFER",
    "eurate0003": "ECB Deposit Effective Rate",
    "jprate0008": "Japan Discount Rate",
    "brrate0003": "CDI",
    "usnaac0169": "US Consumption",
    "eueunaac0149": "Europe Consumption",
    "jpnaac0004": "Japan Consumption",
    "brnaac0016": "Brazil Consumption",
}
df_mb = mb.get_series(mb_tickers)

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


# =========================================
# ===== Consumption VS Excess Returns =====
# =========================================
# US
cxr_us = pd.concat([
    df_mb['US Consumption'].dropna().resample("Y").last(),
    all_eri['S&P 500'].dropna().resample("Y").last(),
    ],
    axis=1,
)
cxr_us = cxr_us.pct_change(1) * 100
cxr_us = cxr_us.dropna(how='any')

# Europe
cxr_eu = pd.concat([
    df_mb['Europe Consumption'].dropna().resample("Y").last(),
    all_eri['EuroStoxx 600'].dropna().resample("Y").last(),
    ],
    axis=1,
)
cxr_eu = cxr_eu.pct_change(1) * 100
cxr_eu = cxr_eu.dropna(how='any')

# Japan
cxr_jp = pd.concat([
    df_mb['Japan Consumption'].dropna().resample("Y").last(),
    all_eri['Topix'].dropna().resample("Y").last(),
    ],
    axis=1,
)
cxr_jp = cxr_jp.pct_change(1) * 100
cxr_jp = cxr_jp.dropna(how='any')

# Japan
cxr_br = pd.concat([
    df_mb['Brazil Consumption'].dropna().resample("Y").last(),
    all_eri['Ibovespa'].dropna().resample("Y").last(),
    ],
    axis=1,
)
cxr_br = cxr_br.pct_change(1) * 100
cxr_br = cxr_br.dropna(how='any')


# ==============================================
# ===== Chart - Separate Countries US / EU =====
# ==============================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))
# fig.suptitle(
#     "Realized Equity Premium",
#     fontsize=16,
#     fontweight="bold",
# )
# --- US ---
x = cxr_us['US Consumption'].values
y = cxr_us['S&P 500'].values
ax = plt.subplot2grid((2, 2), (0, 0))
ax.set_title("United States")
ax.scatter(x, y, label=None, color="#3333B2", edgecolor=None)
ax.set_xlabel("US Consumption Growth")
ax.set_ylabel("S&P 500 Excess Returns")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axhline(0, color="black", lw=1)
ax.axvline(0, color="black", lw=1)

mod = sm.OLS(endog=y, exog=sm.add_constant(x))
res = mod.fit()
cov = np.cov((x/100).flatten(), (y/100).flatten())[0, 1]
eq_str = f"y = {round(res.params[0], 2)} + {round(res.params[1], 2)}x"
eq_str = f"{eq_str}\n$R^{{2}}=${round(res.rsquared, 2)}\nCov={round(cov, 5)}"
x1, x2 = min(x), max(x)
y1, y2 = res.params[0] + res.params[1] * x1, res.params[0] + res.params[1] * x2
ax.plot([x1, x2], [y1, y2], color="#F25F5C", label=eq_str, lw=2)

ax.legend(frameon=True, loc="best")


# --- Europe ---
x = cxr_eu['Europe Consumption'].values
y = cxr_eu['EuroStoxx 600'].values
ax = plt.subplot2grid((2, 2), (0, 1))
ax.set_title("Europe")
ax.scatter(x, y, label=None, color="#3333B2", edgecolor=None)
ax.set_xlabel("Europe Consumption Growth")
ax.set_ylabel("EuroStoxx 600 Excess Returns")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axhline(0, color="black", lw=1)
ax.axvline(0, color="black", lw=1)

mod = sm.OLS(endog=y, exog=sm.add_constant(x))
res = mod.fit()
cov = np.cov((x/100).flatten(), (y/100).flatten())[0, 1]
eq_str = f"y = {round(res.params[0], 2)} + {round(res.params[1], 2)}x"
eq_str = f"{eq_str}\n$R^{{2}}=${round(res.rsquared, 2)}\nCov={round(cov, 5)}"
x1, x2 = min(x), max(x)
y1, y2 = res.params[0] + res.params[1] * x1, res.params[0] + res.params[1] * x2
ax.plot([x1, x2], [y1, y2], color="#F25F5C", label=eq_str, lw=2)

ax.legend(frameon=True, loc="best")


# --- Japan ---
x = cxr_jp['Japan Consumption'].values
y = cxr_jp['Topix'].values
ax = plt.subplot2grid((2, 2), (1, 0))
ax.set_title("Japan")
ax.scatter(x, y, label=None, color="#3333B2", edgecolor=None)
ax.set_xlabel("Japan Consumption Growth")
ax.set_ylabel("Topix Excess Returns")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axhline(0, color="black", lw=1)
ax.axvline(0, color="black", lw=1)

mod = sm.OLS(endog=y, exog=sm.add_constant(x))
res = mod.fit()
cov = np.cov((x/100).flatten(), (y/100).flatten())[0, 1]
eq_str = f"y = {round(res.params[0], 2)} + {round(res.params[1], 2)}x"
eq_str = f"{eq_str}\n$R^{{2}}=${round(res.rsquared, 2)}\nCov={round(cov, 5)}"
x1, x2 = min(x), max(x)
y1, y2 = res.params[0] + res.params[1] * x1, res.params[0] + res.params[1] * x2
ax.plot([x1, x2], [y1, y2], color="#F25F5C", label=eq_str, lw=2)

ax.legend(frameon=True, loc="best")

# --- Brazil ---
x = cxr_br['Brazil Consumption'].values
y = cxr_br['Ibovespa'].values
ax = plt.subplot2grid((2, 2), (1, 1))
ax.set_title("Brazil")
ax.scatter(x, y, label=None, color="#3333B2", edgecolor=None)
ax.set_xlabel("Brazil Consumption Growth")
ax.set_ylabel("Ibovespa Excess Returns")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axhline(0, color="black", lw=1)
ax.axvline(0, color="black", lw=1)

mod = sm.OLS(endog=y, exog=sm.add_constant(x))
res = mod.fit()
cov = np.cov((x/100).flatten(), (y/100).flatten())[0, 1]
eq_str = f"y = {round(res.params[0], 2)} + {round(res.params[1], 2)}x"
eq_str = f"{eq_str}\n$R^{{2}}=${round(res.rsquared, 2)}\nCov={round(cov, 5)}"
x1, x2 = min(x), max(x)
y1, y2 = res.params[0] + res.params[1] * x1, res.params[0] + res.params[1] * x2
ax.plot([x1, x2], [y1, y2], color="#F25F5C", label=eq_str, lw=2)

ax.legend(frameon=True, loc="best")


plt.tight_layout()

plt.savefig(AA_LECTURE.joinpath("Figures/Equities - ERI vs Consumption.pdf"))
plt.show()
plt.close()
