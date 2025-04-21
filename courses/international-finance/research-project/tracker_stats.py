import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.performance import Performance


df = pd.read_excel('/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                   sheet_name='CDS Trackers',
                   index_col=0)
df.index = pd.to_datetime(df.index)
df.loc["2017-11-15":, 'VENZ'] = np.nan
df.loc["2022-09-12":, 'RUSSIA'] = np.nan
df.loc["2022-09-28":, 'UKRAIN'] = np.nan
print(df.columns)

# =================
# ===== Chart =====
# =================
size = 5
fig = plt.figure(figsize=(size * (16 / 7.3), size))

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
ax.plot(df['UKRAIN'], label='Ukraine')
ax.plot(df['ARGENT'], label='Argentina')
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()
plt.savefig(f'/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Research Project/figures/trackers.pdf')
plt.show()


# =======================
# ===== Performance =====
# =======================
perf = Performance(df, skip_dd=True)
pt = perf.table.T
pt = pt.sort_values("Sharpe")

with pd.ExcelWriter('/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Research Project/Individual Performance.xlsx') as writer:
    pt.to_excel(writer, 'performance')

print(pt)
