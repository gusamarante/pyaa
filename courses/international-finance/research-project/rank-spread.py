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
username = getpass.getuser()

trackers = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
    sheet_name='CDS Trackers',
    index_col=0,
)
trackers = trackers.resample('M').last()
is_available = ~trackers.isna()
rets = np.log(trackers).diff(1)

spreads = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
    sheet_name='CDS Spread',
    index_col=0,
)
spreads = spreads.resample('M').last()
spreads = spreads.reindex(trackers.index)


# Assign countries to portfolios
def assign_portfolio(x):
    x = x[is_available.loc[x.name]]
    x = pd.qcut(x, q=n_portfolios, labels=port_labels)
    return x

portfolios = spreads.apply(assign_portfolio, axis=1)
portfolios = portfolios.shift(1)  # Today's selection is tomorrow's portfolio

portfolios = pd.DataFrame({"returns": rets.stack(), "portfolio": portfolios.stack()})
portfolios = portfolios.groupby(['date', 'portfolio']).mean()
portfolios = portfolios.unstack("portfolio")["returns"]

portfolios[f"Port {n_portfolios - 1}-1"] = portfolios[f"Port {n_portfolios - 1}"] - portfolios[f"Port 1"]
portfolios[f"Port {n_portfolios}-{n_portfolios - 1}"] = portfolios[f"Port {n_portfolios}"] - portfolios[f"Port {n_portfolios - 1}"]

port_trackers = (1 + portfolios).cumprod()
port_trackers = 100 * port_trackers / port_trackers.iloc[0]

# Performance
perf = Performance(port_trackers, skip_dd=True, freq="M")
perf.table.T.to_clipboard()
print(perf.table)

# PCA
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
plt.savefig(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/figures/PCA ranked spread.pdf')
plt.show()
plt.close()


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.plot(port_trackers.iloc[:, :n_portfolios], label=port_trackers.columns[:n_portfolios])
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="upper left")
ax.set_title("Ranked Portfolios")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

ax = plt.subplot2grid((1, 2), (0, 1))  # TODO change to perf table
ax.plot(port_trackers.iloc[:, n_portfolios:], label=port_trackers.columns[n_portfolios:])
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="upper left")
ax.set_title("Long-Short Portfolios")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

plt.tight_layout()
plt.savefig(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/figures/trackers portfolios spread.pdf')
plt.show()
plt.close()
