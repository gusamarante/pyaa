"""
Índice IDA da ANBIMA

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
from bwmktdata import Macrobond
from bwsecrets.api import get_secret

size = 5


# ================
# ===== Data =====
# ================
filepath = "C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Dados BBG AA Course.xlsx"  # work
df = pd.read_excel(filepath, index_col=0, skiprows=4, sheet_name="Credit")
df = df.drop('Dates', axis=0)
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df = df.dropna(how='all')

# Macrobond
passwords = get_secret("macrobond")
mb = Macrobond(client_id=passwords["client_id"], client_secret=passwords["client_secret"])
mb_tickers = {
    "brrate0003": "CDI",
}

df_mb = mb.fetch_series(mb_tickers)
cdi_rate = df_mb['CDI'].dropna().copy()
cdi_rate = cdi_rate / 100
cditr = (1 + cdi_rate).cumprod()


# =================
# ===== CHART =====
# =================
# Total Return
df2plot = pd.concat([df['BR TR'], cditr], axis=1)
df2plot = df2plot.dropna()
df2plot = 100 * df2plot / df2plot.iloc[0]

fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Brazil - ANBIMA Credit Index (IDA-Geral)")
ax.plot(df2plot['BR TR'], label="IDA", color="#3333B2", lw=1)
ax.plot(df2plot['CDI'], label="CDI", color="#F25F5C", lw=1)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_ylabel("Index (Log Scale)")
ax.set_yscale("log")
ax.get_yaxis().set_minor_formatter(ScalarFormatter())
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")


plt.tight_layout()

# save_path = '/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Asset Allocation/Figures/Credit - EU HY vs IG.pdf'  # mac
save_path = "C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Figures/Credit - Brasil IDA.pdf"  # work
plt.savefig(save_path)
plt.show()
plt.close()
