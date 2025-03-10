import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from data import raw_di
from utils import RF_LECTURE, BLUE

notional_start = 100
size = 5
contract = "F25"

# Read the Data
di = raw_di()

di = di[di["contract"] == contract]
di = di.sort_values("reference_date")
pnl = di.pivot(index="reference_date", values="pnl", columns="contract")[contract]
tracker = pnl.cumsum()
tracker = 100 * tracker / tracker.iloc[0]

# Volatility
ret = tracker.pct_change(1).dropna()
vol = ret.ewm(com=252, min_periods=126).std() * np.sqrt(252)


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title(f"Total Return Index - DI Future {contract}")
ax.plot(tracker, label="ERI", color=BLUE)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

plt.tight_layout()

plt.savefig(RF_LECTURE.joinpath('figures/Single DI Future Total Return.pdf'))

plt.show()
plt.close()


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((2, 1), (0, 0))
ax.set_title(f"Daily Returns - DI Future {contract}")
ax.plot(ret, label="Returns", color=BLUE)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

ax = plt.subplot2grid((2, 1), (1, 0), sharex=ax)
ax.set_title(f"Annualized 1y Volatility - DI Future {contract}")
ax.plot(vol, label="Annualized Volatility", color=BLUE)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

plt.tight_layout()

plt.savefig(RF_LECTURE.joinpath("figures/Single DI Future Return and Vol.pdf"))

plt.show()
plt.close()
