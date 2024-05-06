"""
Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
FFBA08 - Yellow
F25F5C - Red
"""
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from utils import SGS, Performance
import pandas as pd

size = 5

# ================
# ===== Data =====
# ================
sgs = SGS()
cdi = sgs.fetch(series_id={12: 'CDI'})
cdi = cdi['CDI'] / 100

filepath = '/trackers/output data/trackers_ntnb.xlsx'  # mac
# filepath = "C:/Users/gamarante/PycharmProjects/pyaa/trackers/output data/trackers_ntnb.xlsx"  # work
df = pd.read_excel(filepath, index_col=0)

df_rx = df.pct_change(1).sub(cdi, axis=0).dropna()
df_eri = (1 + df_rx).cumprod()
df_eri = 100 * df_eri / df_eri.iloc[0]

perf = Performance(df_eri)
perf.table.to_clipboard()
print(perf.table)

# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("NTN-B Excess Return Index")
ax.plot(df_eri['NTNB 1y'], label="1y", color="#3333B2", lw=2)
ax.plot(df_eri['NTNB 5y'], label="5y", color="#0B6E4F", lw=2)
ax.plot(df_eri['NTNB 10y'], label="10y", color="#FFBA08", lw=2)
ax.plot(df_eri['NTNB 25y'], label="25y", color="#F25F5C", lw=2)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_ylabel("Index")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("NTN-B Excess Returns and Vol")
ax.plot(perf.std.values * 100, perf.returns_ann.values * 100, color="#3333B2", lw=2)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Annualized Return")

labels = perf.std.index.str[5:]
for x, y, lb in zip(perf.std.values * 100, perf.returns_ann.values * 100, labels):
    ax.annotate(lb, (x + 0.1, y))


plt.tight_layout()

save_path = '/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Insper - Asset Allocation/Figures/Bonds - NTNB Historical Excess Returns.pdf'  # home
# save_path = "C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Figures/Bonds - NTNB Historical Excess Returns.pdf"  # work
plt.savefig(save_path)
plt.show()
plt.close()