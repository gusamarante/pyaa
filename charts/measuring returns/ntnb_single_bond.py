import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from data import raw_ntnb
from utils import AA_LECTURE, ASSET_ALLOCATION, BLUE

last_year = 2024
notional_start = 100
start_date = '2006-01-01'
size = 5

# Read the Data
ntnb = raw_ntnb()

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
vol = ret.ewm(com=252, min_periods=126).std() * np.sqrt(252)


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Total Return Index - NTNB Maturing in May/2023")
ax.plot(ntnb['notional'], label="TRI", color=BLUE, lw=2)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

plt.tight_layout()

plt.savefig(AA_LECTURE.joinpath('Figures/Measuring Returns - Single NTNB Total Return.pdf'))
plt.savefig(ASSET_ALLOCATION.joinpath('Measuring Returns - Single NTNB Total Return.pdf'))
plt.show()
plt.close()


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((2, 1), (0, 0))
ax.set_title("Daily Returns - NTNB Maturing in May/2023")
ax.plot(ret, label="Returns", color=BLUE, lw=2)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

ax = plt.subplot2grid((2, 1), (1, 0), sharex=ax)
ax.set_title("Annualized 1y Volatility - NTNB Maturing in May/2023")
ax.plot(vol, label="Annualized Volatility", color=BLUE, lw=2)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

plt.tight_layout()

plt.savefig(AA_LECTURE.joinpath('Figures/Measuring Returns - Single NTNB Returns and Vol.pdf'))
plt.savefig(ASSET_ALLOCATION.joinpath('Measuring Returns - Single NTNB Returns and Vol.pdf'))
plt.show()
plt.close()
