import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from matplotlib.pylab import Slider
from statsmodels.tsa.filters.hp_filter import hpfilter
from data import SGS
import matplotlib.dates as mdates
from utils import BLUE, RED

start_val_lambda = 129600

# Get data
df = SGS().fetch(series_id={24364: "IBC-Br"})
series = np.log(df["IBC-Br"])

def hp(s, lamb):
    c, t = hpfilter(s, lamb=lamb)
    return c, t


# =================
# ===== Chart =====
# =================
size = 7
fig = plt.figure(figsize=(size * (16 / 7.3), size))

# Level and Trend
ax = plt.subplot2grid((7, 1), (0, 0), rowspan=4)
ax.plot(series.index, series.values, label="log GDP", color=BLUE, lw=2)
c_trend, = ax.plot(  # TODO this necessary?
    series.index,
    hp(series, lamb=start_val_lambda)[1],
    label="Trend",
    color=RED,
    lw=2,
)
ax.set_title("Level and Trend")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

# Cycle
ax = plt.subplot2grid((7, 1), (4, 0), rowspan=2)
c_cycle, = ax.plot(
    series.index,
    hp(series, lamb=start_val_lambda)[0],
    color=BLUE,
    lw=2,
)
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Cycle - % deviation from trend")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")


# Sliders
# ax_lamb = fig.add_axes([0.15, 0.02, 0.65, 0.04])  # left, bottom, width, height
ax_lamb = plt.subplot2grid((7, 1), (6, 0))
sy = Slider(
    ax=ax_lamb,
    label="$\lambda$",
    valmin=0,
    valmax=start_val_lambda * 2,
    valinit=start_val_lambda,
    valstep=10,
)


def update(val):
    lamb = sy.val
    c_trend.set_ydata(hp(series, lamb)[1])
    c_cycle.set_ydata(hp(series, lamb)[0])
    fig.canvas.draw_idle()

sy.on_changed(update)
plt.tight_layout()

plt.show()
