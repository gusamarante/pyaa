import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.performance import Performance
from allocation import HRP
from sklearn.decomposition import PCA
import numpy as np
from utils import BLUE
import getpass


username = getpass.getuser()

debt = pd.read_excel(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                     sheet_name='External Debt Renamed',
                     index_col=0)
debt.index = pd.to_datetime(debt.index)
debt = debt.resample("M").last().ffill()

oi = pd.read_excel(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                   sheet_name='Open Interest',
                   index_col=0)
oi.index = pd.to_datetime(oi.index)
oi = oi.resample("M").mean()

dtc = (oi / debt).dropna(how='all', axis=0).dropna(how='all', axis=1)

selected_countries = [
    "KINGDOM OF SAUDI ARABIA",
    "FEDERATIVE REPUBLIC OF BRAZIL",
    "UNITED MEXICAN STATES",
    "RUSSIAN FEDERATION",
    "REPUBLIC OF ITALY",
    "PEOPLE'S REPUBLIC OF CHINA",
    "UKRAINE",
    "HELLENIC REPUBLIC",
    "UNITED STATES OF AMERICA",
    "REPUBLIC OF KOREA",
]


# =================
# ===== Chart =====
# =================
size = 5
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(dtc[selected_countries] * 100, label=selected_countries)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="upper left", ncols=2)

plt.tight_layout()
plt.savefig(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/figures/cds to debt.pdf')
plt.show()
