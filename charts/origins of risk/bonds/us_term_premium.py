"""
Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
FFBA08 - Yellow
F25F5C - Red
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from bwmktdata import Macrobond
from bwsecrets.api import get_secret

size = 5

# ===== Read the Data =====
# Macrobond
passwords = get_secret("macrobond")
mb = Macrobond(client_id=passwords["client_id"], client_secret=passwords["client_secret"])
mb_tickers = {
    "acmtp10": "ACM",
    "kwtpy10": "KW",
}

df_mb = mb.fetch_series(mb_tickers)


# =================
# ===== Chart =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("10y Term Premium Models for United States")
ax.plot(df_mb['KW'].dropna(), label="KW (2005)", color="#F25F5C")
ax.plot(df_mb['ACM'].dropna(), label="ACM (2013)", color="#3333B2")
ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_ylabel("Term-Premium")
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.savefig(r"C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Figures/Bonds - US Term Premium.pdf")
plt.show()
plt.close()
