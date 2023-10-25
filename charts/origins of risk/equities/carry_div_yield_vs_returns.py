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
import numpy as np
from matplotlib.ticker import ScalarFormatter
import statsmodels.api as sm

from bwmktdata import Macrobond
from bwsecrets.api import get_secret


size = 5

# ===== Read the Data =====
file_path = r"C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Dados BBG AA Course.xlsx"
rename_tickers = {
    "SPX Index": "S&P500",
    "SXXP Index": "EuroStoxx 600",
    "TPX Index": "Topix",
    "IBOV Index": "Ibovespa",
}

# Total Return
tri = pd.read_excel(file_path, sheet_name='TOT_RETURN_INDEX_GROSS_DVDS', skiprows=4, index_col=0)
tri = tri.sort_index()
tri = tri.dropna(how='all')
tri = tri.rename(rename_tickers, axis=1)

# Dividend Yield
dy = pd.read_excel(file_path, sheet_name='NET_AGGTE_DVD_YLD', skiprows=4, index_col=0)
dy = dy.sort_index()
dy = dy.dropna(how='all')
dy = dy.rename(rename_tickers, axis=1)

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

# ===== Compute ERI =====
# US
ffr = df_mb['FFER'].dropna().copy()
ffr = (1 + ffr / 100) ** (1 / 252) - 1
ffrtr = (1+ffr).cumprod()

spx = tri['S&P500'].dropna().copy()
ret_spx = spx.pct_change(1)
xret_spx = ((1 + ret_spx) / (1 + ffr) - 1).dropna()
eri_spx = (1 + xret_spx).cumprod()

# ===== Europe =====
ecb_rate = df_mb['ECB Deposit Effective Rate'].dropna().copy()
ecb_rate = (1 + ecb_rate / 100) ** (1 / 252) - 1
ecbtr = (1+ecb_rate).cumprod()

sxxp = tri['EuroStoxx 600'].dropna().copy()
ret_sxxp = sxxp.pct_change(1)
xret_sxxp = ((1 + ret_sxxp) / (1 + ecb_rate) - 1).dropna()
eri_sxxp = (1 + xret_sxxp).cumprod()

# ===== Japan =====
jpn_rate = df_mb['Japan Discount Rate'].dropna().copy()
jpn_rate = (1 + jpn_rate / 100) ** (1 / 252) - 1
jpntr = (1 + jpn_rate).cumprod()

topix = tri['Topix'].dropna().copy()
ret_topix = topix.pct_change(1)
xret_topix = ((1 + ret_topix) / (1 + jpn_rate) - 1).dropna()
eri_topix = (1 + xret_topix).cumprod()

# ===== Brazil =====
cdi_rate = df_mb['CDI'].dropna().copy()
cdi_rate = cdi_rate / 100
cditr = (1 + cdi_rate).cumprod()

ibov = tri['Ibovespa'].dropna().copy()
ret_ibov = ibov.pct_change(1)
xret_ibov = ((1 + ret_ibov) / (1 + cdi_rate) - 1).dropna()
xret_ibov = xret_ibov[xret_ibov.index >= "1995-01-01"]
eri_ibov = (1 + xret_ibov).cumprod()


# All ERI
all_eri = pd.concat([
    eri_spx.rename("S&P500"),
    eri_sxxp.rename("EuroStoxx 600"),
    eri_topix.rename("Topix"),
    eri_ibov.rename("Ibovespa"),
        ], axis=1)


# Yearly variables
ret = all_eri.resample('Y').last()
ret = ret.pct_change(1) * 100
dyt = dy.resample('Y').last().shift(1)  # lagged

# ====================
# ===== Analysis =====
# ====================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

# --- US ---
index_name = 'S&P500'
aux = pd.concat([ret[index_name].rename('Excess Return'), dyt[index_name].rename('Div Yield')],
                axis=1)
aux = aux.dropna()
x = aux['Div Yield'].values
y = aux['Excess Return'].values
ax = plt.subplot2grid((2, 2), (0, 0))
ax.set_title("United States")
ax.scatter(x, y, label=None, color="#3333B2", edgecolor=None)
ax.set_xlabel(r"Dividend Yield $\frac{D_t}{P_t}$")
ax.set_ylabel(r"Excess Returns $R_{t+1}-R_{t}^{f}$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axhline(0, color="black", lw=1)

mod = sm.OLS(endog=y, exog=sm.add_constant(x))
res = mod.fit()
tstat = res.tvalues[1]

eq_str = f"y = {round(res.params[0], 2)} + {round(res.params[1], 2)}x"
eq_str = f"{eq_str}\nt-stat =${round(tstat, 2)}$\n$R^{{2}}=${round(res.rsquared, 2)}"
x1, x2 = min(x), max(x)
y1, y2 = res.params[0] + res.params[1] * x1, res.params[0] + res.params[1] * x2
ax.plot([x1, x2], [y1, y2], color="#F25F5C", label=eq_str, lw=2)

ax.legend(frameon=True, loc="best")


# --- Europe ---
index_name = 'EuroStoxx 600'
aux = pd.concat([ret[index_name].rename('Excess Return'), dyt[index_name].rename('Div Yield')],
                axis=1)
aux = aux.dropna()
x = aux['Div Yield'].values
y = aux['Excess Return'].values
ax = plt.subplot2grid((2, 2), (0, 1))
ax.set_title("Europe")
ax.scatter(x, y, label=None, color="#3333B2", edgecolor=None)
ax.set_xlabel(r"Dividend Yield $\frac{D_t}{P_t}$")
ax.set_ylabel(r"Excess Returns $R_{t+1}-R_{t}^{f}$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axhline(0, color="black", lw=1)

mod = sm.OLS(endog=y, exog=sm.add_constant(x))
res = mod.fit()
tstat = res.tvalues[1]

eq_str = f"y = {round(res.params[0], 2)} + {round(res.params[1], 2)}x"
eq_str = f"{eq_str}\nt-stat =${round(tstat, 2)}$\n$R^{{2}}=${round(res.rsquared, 2)}"
x1, x2 = min(x), max(x)
y1, y2 = res.params[0] + res.params[1] * x1, res.params[0] + res.params[1] * x2
ax.plot([x1, x2], [y1, y2], color="#F25F5C", label=eq_str, lw=2)

ax.legend(frameon=True, loc="best")


# --- Japan ---
index_name = 'Topix'
aux = pd.concat([ret[index_name].rename('Excess Return'), dyt[index_name].rename('Div Yield')],
                axis=1)
aux = aux.dropna()
x = aux['Div Yield'].values
y = aux['Excess Return'].values
ax = plt.subplot2grid((2, 2), (1, 0))
ax.set_title("Japan")
ax.scatter(x, y, label=None, color="#3333B2", edgecolor=None)
ax.set_xlabel(r"Dividend Yield $\frac{D_t}{P_t}$")
ax.set_ylabel(r"Excess Returns $R_{t+1}-R_{t}^{f}$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axhline(0, color="black", lw=1)

mod = sm.OLS(endog=y, exog=sm.add_constant(x))
res = mod.fit()
tstat = res.tvalues[1]

eq_str = f"y = {round(res.params[0], 2)} + {round(res.params[1], 2)}x"
eq_str = f"{eq_str}\nt-stat =${round(tstat, 2)}$\n$R^{{2}}=${round(res.rsquared, 2)}"
x1, x2 = min(x), max(x)
y1, y2 = res.params[0] + res.params[1] * x1, res.params[0] + res.params[1] * x2
ax.plot([x1, x2], [y1, y2], color="#F25F5C", label=eq_str, lw=2)

ax.legend(frameon=True, loc="best")


# --- Brazil ---
index_name = 'Ibovespa'
aux = pd.concat([ret[index_name].rename('Excess Return'), dyt[index_name].rename('Div Yield')],
                axis=1)
aux = aux.dropna()
x = aux['Div Yield'].values
y = aux['Excess Return'].values
ax = plt.subplot2grid((2, 2), (1, 1))
ax.set_title("Brazil")
ax.scatter(x, y, label=None, color="#3333B2", edgecolor=None)
ax.set_xlabel(r"Dividend Yield $\frac{D_t}{P_t}$")
ax.set_ylabel(r"Excess Returns $R_{t+1}-R_{t}^{f}$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axhline(0, color="black", lw=1)

mod = sm.OLS(endog=y, exog=sm.add_constant(x))
res = mod.fit()
tstat = res.tvalues[1]

eq_str = f"y = {round(res.params[0], 2)} + {round(res.params[1], 2)}x"
eq_str = f"{eq_str}\nt-stat =${round(tstat, 2)}$\n$R^{{2}}=${round(res.rsquared, 2)}"
x1, x2 = min(x), max(x)
y1, y2 = res.params[0] + res.params[1] * x1, res.params[0] + res.params[1] * x2
ax.plot([x1, x2], [y1, y2], color="#F25F5C", label=eq_str, lw=2)

ax.legend(frameon=True, loc="best")


plt.tight_layout()

plt.savefig(r"C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Figures/Equities - Carry - Div Yield VS Returns.pdf")
plt.show()
plt.close()
