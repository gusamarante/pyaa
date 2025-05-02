import getpass
import numpy as np
import pandas as pd
from utils.performance import Performance
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from data import get_ff5f
import statsmodels.api as sm
import seaborn as sns

params = {
    "ARGENTINE REPUBLIC": {
        "Markit Name": "ARGENT",
        "DMEM": "EM",
    },
    "COMMONWEALTH OF AUSTRALIA": {
        "Markit Name": "AUSTLA",
        "DMEM": "DM",
    },
    "FEDERAL REPUBLIC OF GERMANY": {
        "Markit Name": "DBR",
        "DMEM": "DM",
    },
    "FEDERATIVE REPUBLIC OF BRAZIL": {
        "Markit Name": "BRAZIL",
        "DMEM": "EM"
    },
    "FRENCH REPUBLIC": {
        "Markit Name": "FRTR",
        "DMEM": "DM"
    },
    "HELLENIC REPUBLIC": {
        "Markit Name": "GREECE",
        "DMEM": "EM",
    },
    "HUNGARY": {
        "Markit Name": "HUNGAA",
        "DMEM": "EM"
    },
    "IRELAND": {
        "Markit Name": "IRELND",
        "DMEM": "DM",
    },
    "JAPAN": {
        "Markit Name": "JAPAN",
        "DMEM": "DM"
    },
    "KINGDOM OF BELGIUM": {
        "Markit Name": "BELG",
        "DMEM": "DM"
    },
    "KINGDOM OF THE NETHERLANDS": {
        "Markit Name": "NETHRS",
        "DMEM": "DM"
    },
    "KINGDOM OF SAUDI ARABIA": {
        "Markit Name": "SAUDI",
        "DMEM": "EM"
    },
    "KINGDOM OF SPAIN": {
        "Markit Name": "SPAIN",
        "DMEM": "DM"
    },
    "KINGDOM OF SWEDEN": {
        "Markit Name": "SWED",
        "DMEM": "DM",
    },
    "KINGDOM OF THAILAND": {
        "Markit Name": "THAI",
        "DMEM": "EM",
    },
    "MALAYSIA": {
        "Markit Name": "MALAYS",
        "DMEM": "EM",
    },
    "PEOPLE'S REPUBLIC OF CHINA": {
        "Markit Name": "CHINA",
        "DMEM": "EM",
    },
    "PORTUGUESE REPUBLIC": {
        "Markit Name": "PORTUG",
        "DMEM": "DM",
    },
    "REPUBLIC OF AUSTRIA": {
        "Markit Name": "AUST",
        "DMEM": "DM",
    },
    "REPUBLIC OF CHILE": {
        "Markit Name": "CHILE",
        "DMEM": "EM",
    },
    "REPUBLIC OF COLOMBIA": {
        "Markit Name": "COLOM",
        "DMEM": "EM",
    },
    "REPUBLIC OF FINLAND": {
        "Markit Name": "FINL",
        "DMEM": "DM",
    },
    "REPUBLIC OF INDONESIA": {
        "Markit Name": "INDON",
        "DMEM": "EM",
    },
    "REPUBLIC OF ITALY": {
        "Markit Name": "ITALY",
        "DMEM": "DM",
    },
    "REPUBLIC OF KAZAKHSTAN": {
        "Markit Name": "KAZAKS",
        "DMEM": "EM",
    },
    "REPUBLIC OF KOREA": {
        "Markit Name": "KOREA",
        "DMEM": "EM",
    },
    "REPUBLIC OF PANAMA": {
        "Markit Name": "PANAMA",
        "DMEM": "EM",
    },
    "REPUBLIC OF PERU": {
        "Markit Name": "PERU",
        "DMEM": "EM",
    },
    "REPUBLIC OF THE PHILIPPINES": {
        "Markit Name": "PHILIP",
        "DMEM": "EM",
    },
    "REPUBLIC OF POLAND": {
        "Markit Name": "POLAND",
        "DMEM": "EM",
    },
    "REPUBLIC OF SOUTH AFRICA": {
        "Markit Name": "SOAF",
        "DMEM": "EM",
    },
    "REPUBLIC OF TURKEY": {
        "Markit Name": "TURKEY",
        "DMEM": "EM",
    },
    "RUSSIAN FEDERATION": {
        "Markit Name": "RUSSIA",
        "DMEM": "EM",
    },
    "STATE OF QATAR": {
        "Markit Name": "QATAR",
        "DMEM": "EM",
    },
    "UKRAINE": {
        "Markit Name": "UKIN",
        "DMEM": "EM",
    },
    "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND": {
        "Markit Name": "UKIN",
        "DMEM": "DM",
    },
    "UNITED MEXICAN STATES": {
        "Markit Name": "MEX",
        "DMEM": "EM",
    },
    "UNITED STATES OF AMERICA": {
        "Markit Name": "USGB",
        "DMEM": "DM",
    },
    "BOLIVARIAN REPUBLIC OF VENEZUELA": {
        "Markit Name": "VENZ",
        "DMEM": "EM",
    },
}

username = getpass.getuser()


# Spreads
spreads = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
    sheet_name='CDS Spread',
    index_col=0,
)

# CDS-to-Debt
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

ctd = (oi / debt).dropna(how='all', axis=0).dropna(how='all', axis=1)
ctd = ctd[[n for n in params.keys() if n in ctd.columns]]
ctd = ctd.rename({k: params[k]["Markit Name"] for k in params.keys()}, axis=1)



avg_spr = spreads.mean()
avg_ctd = ctd.mean()

df2plot = pd.concat(
    [
        np.log(10_000 * avg_spr.rename("Average Spread (Log)")),
        np.log(100 * avg_ctd.rename("Average OI-to-Debt (Log)")),
        pd.Series({params[k]["Markit Name"]: params[k]["DMEM"] for k in params.keys()}, name="DM/EM"),
    ],
    axis=1,
)

pp = sns.pairplot(df2plot, hue="DM/EM")
plt.savefig(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/figures/scatter signals.pdf')
plt.show()


