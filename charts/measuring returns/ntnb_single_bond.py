import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import pandas as pd

last_year = 2023
notional_start = 100
start_date = '2006-01-01'
size = 5

# Read the Data
ntnb = pd.DataFrame()

for year in tqdm(range(2003, last_year + 1), 'Reading files'):
    aux = pd.read_csv(f'input data/dados_ntnb {year}.csv', sep=';')
    ntnb = pd.concat([ntnb, aux])

ntnb['reference date'] = pd.to_datetime(ntnb['reference date'])
ntnb['maturity'] = pd.to_datetime(ntnb['maturity'])
ntnb = ntnb.drop(['Unnamed: 0', 'index'], axis=1)

ntnb = ntnb[ntnb['maturity'] == "2023-05-15"]  # Choose the bond here
ntnb = ntnb[ntnb['du'] >= 0]

ntnb = ntnb.set_index('reference date')
ntnb = ntnb.sort_index()

# Backtest
ntnb['bought'] = ntnb['coupon'] / ntnb['price']
ntnb['quantity'] = ntnb['bought'].cumsum() + 1
ntnb['notional'] = ntnb['quantity'] * ntnb['price']

# Volatility
ret = ntnb['notional'].pct_change(1).dropna()
vol = ret.rolling(252).std() * np.sqrt(252)


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Total Return Index - NTNB 2023")
ax.plot(ntnb['notional'], label="TRI", color="#3333B2", lw=2)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("Annualized 1y Volatility - NTNB 2023")
ax.plot(vol, label="Annualized Volatility", color="#3333B2", lw=2)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

plt.tight_layout()

plt.savefig('/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Asset Allocation/Figures/Measuring Returns - Single NTNB Total Return.pdf')
plt.show()
plt.close()


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Daily Returns - NTNB 2023")
ax.plot(ret, label="Returns", color="#3333B2", lw=2)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

plt.tight_layout()

plt.savefig('/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Asset Allocation/Figures/Measuring Returns - Single NTNB Returns.pdf')
plt.show()
plt.close()
