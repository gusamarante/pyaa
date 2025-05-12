import getpass
import numpy as np
import pandas as pd
from utils.performance import Performance
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA


n_port_spread = 2
n_port_otd = 2
size = 5
username = getpass.getuser()
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


# ============================
# ===== CUSTOM FUNCTIONS =====
# ============================
def assign_portfolio(x, n_port, labels):
    x = x[is_available.loc[x.name]]
    x = pd.qcut(x, q=n_port, labels=labels)
    return x


# =======================
# ===== DATA INPUTS =====
# =======================
# Trackers
trackers = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
    sheet_name='CDS Trackers',
    index_col=0,
)
trackers = trackers.resample('ME').last()
is_available = ~trackers.isna()
rets = np.log(trackers).diff(1)


# Spread
spreads = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
    sheet_name='CDS Spread',
    index_col=0,
)
spreads = spreads.resample('ME').last()
spreads = spreads.reindex(trackers.index)
port_spread = spreads.apply(
    assign_portfolio,
    n_port=2, labels=[f"SPR{i+1}" for i in range(n_port_spread)],
    axis=1,
)


# OI-to-Debt
debt = pd.read_excel(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                     sheet_name='External Debt Renamed',
                     index_col=0)
debt.index = pd.to_datetime(debt.index)
debt = debt.resample("ME").last().ffill()

oi = pd.read_excel(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                   sheet_name='Open Interest',
                   index_col=0)
oi.index = pd.to_datetime(oi.index)
oi = oi.resample("ME").mean()

otd = (oi / debt).dropna(how='all', axis=0).dropna(how='all', axis=1)
otd = otd[[n for n in params.keys() if n in otd.columns]]
otd = otd.rename({k: params[k]["Markit Name"] for k in params.keys()}, axis=1)
port_otd = otd.apply(
    assign_portfolio,
    n_port=2, labels=[f"OTD{i+1}" for i in range(n_port_otd)],
    axis=1,
)


# consolidate in single df
data = rets.dropna(how='all')
data = data.stack().rename('returns')
data.index.names = ['date', 'country']

ps = port_spread.stack().rename('port spread')
ps.index.names = ['date', 'country']

potd = port_otd.stack().rename('port otd')
potd.index.names = ['date', 'country']

data = pd.concat([data, ps, potd], axis=1)


# ====================================
# ===== DOUBLE SORTED PORTFOLIOS =====
# ====================================
# Independent Bivariate Sort
# Returns
port_ds = data.pivot_table(
    index='date',
    values='returns',
    columns=['port spread', 'port otd'],
    aggfunc='mean',
)

# Trackers
tracker_ds = 1 + port_ds
new_idx = pd.date_range(
    start=tracker_ds.index[0] - pd.offsets.MonthEnd(1),
    end=tracker_ds.index[-1],
    freq="ME",
)
tracker_ds = tracker_ds.reindex(new_idx)
tracker_ds.iloc[0] = 100
tracker_ds = tracker_ds.cumprod()

tracker_ds.plot()
plt.show()




a = 1