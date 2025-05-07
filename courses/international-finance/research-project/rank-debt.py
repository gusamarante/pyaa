import getpass
import numpy as np
import pandas as pd
from utils.performance import Performance
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA


n_portfolios = 5
size = 5
port_labels = [f"Port {i+1}" for i in range(n_portfolios)]

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

trackers = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
    sheet_name='CDS Trackers',
    index_col=0,
)
trackers = trackers.resample('M').last()
is_available = ~trackers.isna()
rets = np.log(trackers).diff(1)

# CDS-to-Debt
debt = pd.read_excel(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                     sheet_name='External Debt Renamed',
                     index_col=0)
debt.index = pd.to_datetime(debt.index)
debt = debt.resample("M").last().ffill()

# Open Interest
oi = pd.read_excel(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                   sheet_name='Open Interest',
                   index_col=0)
oi.index = pd.to_datetime(oi.index)
oi = oi.resample("M").mean()

# OI to Debt ratio
ctd = (oi / debt).dropna(how='all', axis=0).dropna(how='all', axis=1)
ctd = ctd[[n for n in params.keys() if n in ctd.columns]]
ctd = ctd.rename({k: params[k]["Markit Name"] for k in params.keys()}, axis=1)


# Assign countries to portfolios
def assign_portfolio(x):
    x = x[is_available.loc[x.name]]
    x = pd.qcut(x, q=n_portfolios, labels=port_labels)
    return x

portfolios = ctd.apply(assign_portfolio, axis=1)
portfolios = portfolios.shift(1)  # Today's selection is tomorrow's portfolio

portfolios = pd.DataFrame({"returns": rets.stack(), "portfolio": portfolios.stack()})
portfolios = portfolios.groupby(['date', 'portfolio']).mean()
portfolios = portfolios.unstack("portfolio")["returns"]

portfolios[f"Port {n_portfolios - 1}-1"] = portfolios[f"Port {n_portfolios - 1}"] - portfolios[f"Port 1"]
portfolios[f"Port {n_portfolios}-{n_portfolios - 1}"] = portfolios[f"Port {n_portfolios}"] - portfolios[f"Port {n_portfolios - 1}"]

portfolios = portfolios.dropna(how='all')
port_trackers = (1 + portfolios).cumprod()
port_trackers = 100 * port_trackers / port_trackers.iloc[0]

# Performance
perf = Performance(port_trackers, skip_dd=True)
perf.table.to_clipboard()
print(perf.table)

# # PCA
pca = PCA(n_components=n_portfolios)
pca_ports = portfolios.iloc[:, :n_portfolios]
pca_ports = (pca_ports - pca_ports.mean()) / pca_ports.std()

pca.fit(pca_ports)

var_ratio = pd.Series(data=pca.explained_variance_ratio_,
                        index=[f'PC {i+1}' for i in range(n_portfolios)])
loadings = pd.DataFrame(data=pca.components_.T,
                        columns=[f'PC {i+1}' for i in range(n_portfolios)],
                        index=portfolios.columns[:n_portfolios])

print(var_ratio)
print(var_ratio.cumsum())
print(loadings)


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (4 / 3), size))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(loadings.iloc[:, :2], label=loadings.iloc[:, :2].columns)
ax.axhline(0, color="black", lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="upper left")

plt.tight_layout()
plt.savefig(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/figures/PCA ranked debt.pdf')
plt.show()
plt.close()


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(port_trackers.iloc[:, :n_portfolios], label=port_trackers.columns[:n_portfolios])
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="upper left")
ax.set_title("Ranked Portfolios")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

# ax = plt.subplot2grid((1, 2), (0, 1))  # TODO change to perf table
# ax.plot(port_trackers.iloc[:, n_portfolios:], label=port_trackers.columns[n_portfolios:])
# ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.legend(frameon=True, loc="upper left")
# ax.set_title("Long-Short Portfolios")
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
# ax.tick_params(rotation=90, axis="x")

plt.tight_layout()
plt.savefig(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/figures/trackers debt ranked portfolios.pdf')
plt.show()
plt.close()
