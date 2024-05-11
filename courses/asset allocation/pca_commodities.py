"""
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
import pandas as pd
from bwaa.indexes import BackfilledInternalTrackers
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pathlib import Path
from getpass import getuser
from sklearn.decomposition import PCA
import seaborn as sns

names = {
    "BO": "Soybean Oil",
    "C ": "Corn",
    "CC": "Cocoa",
    "CT": "Cotton",
    "KC": "Coffee",
    "KW": "Wheat (Kansas)",
    "S ": "Soybean",
    "SB": "Sugar",
    "SM": "Soybean Meal",
    "W ": "Wheat (Chicago)",
    "FC": "Feeder Cattle",
    "LC": "Live Cattle",
    "LH": "Lean Hogs",
    "CL": "Crude Oil (WTI)",
    "HO": "Heating Oil",
    "CO": "Crude Oil (Brent)",
    "QS": "Gas Oil",
    "NG": "Natural Gas",
    "XB": "Unleaded Gasoline",
    "HG": "High Grade Copper",
    "LA": "Aluminum",
    "LP": "Copper",
    "LN": "Nickel",
    "LL": "Lead",
    "LX": "Zinc",
    "GC": "Gold",
    "SI": "Silver",
    "PL": "Platinum",
    "PA": "Palladium",
    "MO": "Carbon Emissions",
    "SCO": "Iron Ore",
    "IJ": "Rapeseed",
    "QW": "Refined Sugar",
    "DF": "Coffee Robusta",
    "QC": "Cocoa (London)",
    "CA": "Milling Wheat",
}
sector = {
    'CL': 'Energy',
    'CO': 'Energy',
    'HO': 'Energy',
    'MO': 'Energy',
    'NG': 'Energy',
    'QS': 'Energy',
    'XB': 'Energy',
    'BO': 'Grains',
    'C ': 'Grains',
    'CA': 'Grains',
    'IJ': 'Grains',
    'KW': 'Grains',
    'S ': 'Grains',
    'SM': 'Grains',
    'W ': 'Grains',
    'HG': 'Industrial Metals',
    'LA': 'Industrial Metals',
    'LL': 'Industrial Metals',
    'LN': 'Industrial Metals',
    'LP': 'Industrial Metals',
    'LX': 'Industrial Metals',
    'SCO': 'Industrial Metals',
    'FC': 'Livestock',
    'LC': 'Livestock',
    'LH': 'Livestock',
    'GC': 'Precious Metals',
    'PA': 'Precious Metals',
    'PL': 'Precious Metals',
    'SI': 'Precious Metals',
    'CC': 'Softs',
    'CT': 'Softs',
    'DF': 'Softs',
    'KC': 'Softs',
    'QC': 'Softs',
    'QW': 'Softs',
    'SB': 'Softs',
}
df = pd.read_excel(r"C:\Users\gamarante\Dropbox\Aulas\Insper - Asset Allocation\Commodities Total Return.xlsx",
                   index_col=0)
df = df.pct_change(1).dropna()


pca = PCA(n_components=df.shape[1])
pca.fit(df)

df_var = pd.Series(data=pca.explained_variance_ratio_, name='share of var')
df_loadings = pd.DataFrame(data=pca.components_.T,
                           columns=[f'PC {c+1}' for c in range(df.shape[1])],
                           index=df.columns)
df_pca = pd.DataFrame(data=pca.transform(df),
                      columns=[f'PC {c+1}' for c in range(df.shape[1])],
                      index=df.index)

# Chart - Explained Variance Ratio
plt.figure(figsize=(7, 7 * (9 / 16)))
plt.bar([f'PC{a}' for a in range(1, df_var.shape[0] + 1)], df_var.values)
plt.plot(df_var.cumsum(), color='orange')

last_date = df.index[-1]
plt.title(f"Explained Variance of Commodity Returns {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
ax.axhline(0, color='black', linewidth=0.5)
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
# plt.savefig(save_path.joinpath('Commodities - Explained Variance.pdf'))
plt.show()
plt.close()


# Chart - Clustering PCA
plot_pca = df_loadings[[f'PC {c+1}' for c in range(3)]].copy()
plot_pca['Sector'] = pd.Series(sector)
g = sns.pairplot(plot_pca, hue="Sector")
plt.show()
plt.close()
