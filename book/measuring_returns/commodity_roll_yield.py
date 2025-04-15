from bwaa.utilities.aa_index_getter import AAIndexGetter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pathlib import Path
from getpass import getuser
from utils import AA_LECTURE, ASSET_ALLOCATION

ig = AAIndexGetter()

df_raw = ig.get_index(index_type="Tracker Commodity Futures",
                      index_name=["CL", "GC"])
df_raw = df_raw.pivot('date', 'index_name', 'index_level_usd')

df_enh = ig.get_index(index_type="Tracker Commodity Futures Enhanced",
                      index_name=["CL", "GC"])
df_enh = df_enh.pivot('date', 'index_name', 'index_level_usd')


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(6 * (16 / 7.3), 6))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Crude Oil - Excess Return Index")

ax.plot(df_raw['CL'], label="Front Month")
ax.plot(df_enh['CL'], label="Semi-Annual Roll")

ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="upper left")


ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("Gold - Excess Return Index")

ax.plot(df_raw['GC'], label="Front Month")
ax.plot(df_enh['GC'], label="Semi-Annual Roll")

ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="upper left")

plt.tight_layout()

plt.savefig(AA_LECTURE.joinpath("Figures/Commodities - Trackers Roll Yield Example.pdf"))
plt.savefig(ASSET_ALLOCATION.joinpath("Measuring Returns - Commodities - Trackers Roll Yield Example.pdf"))
plt.show()
plt.close()
