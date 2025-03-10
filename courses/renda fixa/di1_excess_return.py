"""
Total Return for portfolios of NTN-Bs of different constant durations

Chart is used in the `measuring returns` section, to ilustrate the construction
of trackers
"""
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from data import trackers_di
from utils import AA_LECTURE, RF_LECTURE, ASSET_ALLOCATION, BLUE, GREEN, RED, YELLOW

size = 5

df = trackers_di()

# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("DI Excess Return Index")
ax.plot(df['DI 1y'], label="1y", color=BLUE)
ax.plot(df['DI 3y'], label="3y", color=RED)
ax.plot(df['DI 5y'], label="5y", color=GREEN)
ax.plot(df['DI 10y'], label="10y", color=YELLOW)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_ylabel("Index")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.savefig(RF_LECTURE.joinpath("figures/DI1 - Total Return.pdf"))
plt.show()
plt.close()
