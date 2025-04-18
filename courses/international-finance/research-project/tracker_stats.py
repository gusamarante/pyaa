import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.performance import Performance

df = pd.read_excel('/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                   index_col=0)
df.index = pd.to_datetime(df.index)
# TODO resize the timeseries
print(df.columns)

# =================
# ===== Chart =====
# =================
size = 5
fig = plt.figure(figsize=(size * (16 / 7.3), size))
plt.suptitle("Excess Return Index", fontweight="bold")

ax = plt.subplot2grid((1, 2), (0, 0))
ax.plot(df['BRAZIL'], label='Brazil')
ax.plot(df['JAPAN'], label='Japan')
ax.plot(df['ITALY'], label='Italy')
ax.plot(df['POLAND'], label='Poland')
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

ax = plt.subplot2grid((1, 2), (0, 1))
ax.plot(df['RUSSIA'], label='Russia')
ax.plot(df['ARGENT'], label='Argentina')
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()
# plt.savefig(f'/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Problem Set 03/figures/item1 objevtive and value functions.pdf')
plt.show()


# =======================
# ===== Performance =====
# =======================
perf = Performance(df)
pt = perf.table.T
pt = pt.sort_values("Sharpe")
print(pt)
