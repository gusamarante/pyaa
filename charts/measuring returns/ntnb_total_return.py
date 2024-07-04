"""
Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
FFBA08 - Yellow
F25F5C - Red
"""
# TODO organize the color pallete rotation
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from data import trackers_ntnb

size = 5

df = trackers_ntnb()

# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("NTN-B Total Return Index")
ax.plot(df['NTNB 1y'], label="1y", color="#3333B2", lw=2)
ax.plot(df['NTNB 5y'], label="5y", color="#0B6E4F", lw=2)
ax.plot(df['NTNB 10y'], label="10y", color="#FFBA08", lw=2)
ax.plot(df['NTNB 25y'], label="25y", color="#F25F5C", lw=2)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_ylabel("Index")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.savefig('/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Asset Allocation/Figures/Measuring Returns - NTNB Total Return.pdf')
plt.show()
plt.close()
