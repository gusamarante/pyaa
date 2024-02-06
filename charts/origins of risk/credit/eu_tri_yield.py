"""
Compara IG e HY para Europa
Total return e Yield

Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
FFBA08 - Yellow
F25F5C - Red
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates

size = 5
start_date = "2001-01-01"


# ================
# ===== Data =====
# ================
filepath = "C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Dados BBG AA Course.xlsx"  # work
df = pd.read_excel(filepath, index_col=0, skiprows=4, sheet_name="Credit")
df = df.drop('Dates', axis=0)
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df = df.dropna(how='all')

df = df[df.index >= start_date]


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

# Total Return
df2plot = df[['EU IG TR', 'EU HY TR']].copy()
df2plot = df2plot.dropna()
df2plot = 100 * df2plot / df2plot.iloc[0]

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Europe Credit Total Return (Bloomberg Aggregates)")
ax.plot(df2plot['EU IG TR'], label="Investment Grade", color="#3333B2", lw=1)
ax.plot(df2plot['EU HY TR'], label="High Yield", color="#0B6E4F", lw=1)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_ylabel("Index (Log Scale)")
ax.set_yscale("log")
ax.get_yaxis().set_minor_formatter(ScalarFormatter())
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")


# Yield
ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("Europe Credit Spread (Bloomberg Aggregates)")
ax.plot(df['EU IG Spread'], label="Investment Grade", color="#3333B2", lw=1)
ax.plot(df['EU HY Spread'], label="High Yield", color="#0B6E4F", lw=1)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_ylabel("Spread")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")


plt.tight_layout()

# save_path = '/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Asset Allocation/Figures/Credit - EU HY vs IG.pdf'  # mac
save_path = "C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Figures/Credit - EU HY vs IG.pdf"  # work
plt.savefig(save_path)
plt.show()
plt.close()

