"""
Total Return for Commdoties

Chart is used in the `measuring returns` section, to ilustrate the construction
of trackers


BBG Code    Subgroup	        Description
CL	        Energy	            Crude Oil (WTI)
CO	        Energy	            Crude Oil (Brent)
HO	        Energy	            Heating Oil
MO	        Energy	            Carbon Emissions
NG	        Energy	            Natural Gas
QS	        Energy	            Gas Oil
XB	        Energy	            Unleaded Gasoline
BO	        Grains	            Soybean Oil
C 	        Grains	            Corn
CA	        Grains	            Milling Wheat
IJ	        Grains	            Rapeseed
KW	        Grains	            Wheat (Kansas)
S 	        Grains	            Soybeans
SM	        Grains	            Soybean Meal
W 	        Grains	            Wheat (Chicago)
HG	        Industrial Metals	High Grade Copper
LA	        Industrial Metals	Aluminum
LL	        Industrial Metals	Lead
LN	        Industrial Metals	Nickel
LP	        Industrial Metals	Copper
LX	        Industrial Metals	Zinc
SCO	        Industrial Metals	Iron Ore
FC	        Livestock	        Feeder Cattle
LC	        Livestock	        Live Cattle
LH	        Livestock	        Lean Hogs
GC	        Precious Metals	    Gold
PA	        Precious Metals	    Palladium
PL	        Precious Metals	    Platinum
SI	        Precious Metals	    Silver
CC	        Softs	            Cocoa
CT	        Softs	            Cotton
DF	        Softs	            Coffee Robusta
KC	        Softs	            Coffee
QC	        Softs	            Cocoa (London)
QW	        Softs	            White (Refined) Sugar
SB	        Softs	            Sugar
"""
from bwaa.indexes import BackfilledInternalTrackers  # TODO add to worksheet
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pathlib import Path
from getpass import getuser
from utils import AA_LECTURE, ASSET_ALLOCATION

names = {
    # "BO": "Soybean Oil",
    # "C ": "Corn",
    # "CC": "Cocoa",
    # "CT": "Cotton",
    # "KC": "Coffee",
    # "KW": "Wheat (Kansas)",
    "S ": "Soybean",
    "SB": "Sugar",
    # "SM": "Soybean Meal",
    # "W ": "Wheat (Chicago)",
    # "FC": "Feeder Cattle",
    "LC": "Live Cattle",
    # "LH": "Lean Hogs",
    "CL": "Crude Oil (WTI)",
    # "HO": "Heating Oil",
    # "CO": "Crude Oil (Brent)",
    # "QS": "Gas Oil",
    # "NG": "Natural Gas",
    # "XB": "Unleaded Gasoline",
    # "HG": "High Grade Copper",
    # "LA": "Aluminum",
    "LP": "Copper",
    # "LN": "Nickel",
    # "LL": "Lead",
    # "LX": "Zinc",
    "GC": "Gold",
    # "SI": "Silver",
    # "PL": "Platinum",
    # "PA": "Palladium",
    # "MO": "Carbon Emissions",
    # "SCO": "Iron Ore",
    # "IJ": "Rapeseed",
    # "QW": "Refined Sugar",
    # "DF": "Coffee Robusta",
    # "QC": "Cocoa (London)",
    # "CA": "Milling Wheat",
}
df = BackfilledInternalTrackers().commodities(markets=list(names.keys()))


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(6 * (16 / 7.3), 6))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Commodities - Excess Return Index")
df2plot = df[list(names.keys())].copy().dropna()
df2plot = 100 * df2plot / df2plot.iloc[0]
df2plot = df2plot.rename(names, axis=1)
ax.plot(df2plot, label=df2plot.columns)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="upper left")

plt.tight_layout()

plt.savefig(AA_LECTURE.joinpath("Figures/Commodities - Trackers.pdf"))
plt.savefig(ASSET_ALLOCATION.joinpath("Measuring Returns - Commodities - Trackers.pdf"))
plt.show()
plt.close()
