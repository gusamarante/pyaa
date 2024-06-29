"""
Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
FFBA08 - Yellow
F25F5C - Red
"""
from data.tracker_data import trackers_di
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import Performance, AA_LECTURE


size = 5
df = trackers_di()

perf = Performance(df, skip_dd=True)


# =================
# ===== Chart =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax.plot(df, label=df.columns, lw=2)
ax.set_title("DI Futures - Constant Duration Excess Return Indexes")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")


ax = plt.subplot2grid((2, 2), (0, 1))
ax.set_title("DI Futures - Excess Return and Vol")
ax.plot(perf.std.values * 100, perf.returns_ann.values * 100, color="#3333B2", lw=2)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Annualized Return")

labels = perf.std.index.str[3:]
for x, y, lb in zip(perf.std.values * 100, perf.returns_ann.values * 100, labels):
    ax.annotate(lb, (x, y - 0.2))


ax = plt.subplot2grid((2, 2), (1, 1))
ax.set_title("DI Futures - Sharpe Ratios")
ax = perf.sharpe.plot(kind='bar', color="#3333B2", ax=ax)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Annualized Return")

plt.tight_layout()

plt.savefig(AA_LECTURE.joinpath("Figures/Bonds - DI Historical Excess Returns.pdf"))
plt.show()
plt.close()
