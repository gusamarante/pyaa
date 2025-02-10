from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from data import SGS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import BLUE, RED

df = SGS().fetch(series_id={24364: "IBC-Br"})
series = np.log(df["IBC-Br"])

cycle, trend = hpfilter(series, lamb=129600)


# =================
# ===== Chart =====
# =================
size = 7
fig = plt.figure(figsize=(size * (16 / 7.3), size))

# Level and Trend
ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax.plot(df["IBC-Br"], label="GDP", color=BLUE, lw=2)
ax.plot(np.exp(trend), label="Trend", color=RED, lw=2)
ax.set_title("Level and Trend")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

# Cycle
ax = plt.subplot2grid((3, 1), (2, 0))
ax.plot(cycle, label="GDP", color=BLUE, lw=2)
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Cycle - % deviation from trend")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.show()
plt.close()