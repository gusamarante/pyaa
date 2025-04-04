"""
generates the charts for the static PCA analysis for yield curves based on
the DI futures curve.
"""

import getpass
import numpy as np
import pandas as pd
from pathlib import Path
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA


# User defined parameters
start_date = '2007-01-01'
username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2024/figures')

# get data
rate = pd.read_csv(f'/Users/{username}/PycharmProjects/pyaa/data/data_output/di monthly maturities.csv',
                   index_col=0)
rate.index = pd.to_datetime(rate.index)
rate.columns = rate.columns.str[:-1].astype(int)
rate = rate.dropna(how='all', axis=0).dropna(how='any', axis=1)
pu = 100_000 / ((1+rate)**(rate.columns / 12))
dv01 = (((- rate.columns / 252) * pu) / (1 + rate)) / 10_000


# ===========================
# ===== Full sample PCA =====
# ===========================
pca = PCA(n_components=4)
pca.fit(rate)

df_var_full = pd.DataFrame(data=pca.explained_variance_ratio_)
df_loadings_full = pd.DataFrame(data=pca.components_.T,
                                columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'],
                                index=rate.columns)
df_mean_full = pd.DataFrame(data=pca.mean_,
                            index=rate.columns,
                            columns=['Médias'])
df_pca_full = pd.DataFrame(data=pca.transform(rate.values),
                           columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'],
                           index=rate.index)

signal = np.sign(df_loadings_full.iloc[-1])
df_loadings_full = df_loadings_full * signal
df_pca_full = df_pca_full * signal


# =======================
# ===== PCA by Hand =====
# =======================
val, vec = la.eig(rate.cov())
explained_varaince = val / val.sum()

loadings = vec[:, 0:3]  # First 3 components
signal = np.sign(loadings[-1, :])  # Normalize signal
loadings = loadings * signal

X = rate.values - rate.values.mean(axis=0)
pcs = X @ loadings


# ==================
# ===== Charts =====
# ==================
# Curve dynamics
df_plot = rate[[12 * a for a in range(1, 10)]] * 100
df_plot.columns =  [f"{a}y" for a in range(1, 10)]

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.plot(df_plot)

last_date = rate.index[-1]
plt.title(f"Yields of the DI Curve as of {last_date.strftime('%d/%b/%y')}")

ax = plt.gca()
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(rotation=90, axis='x')
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(df_plot.columns, frameon=True, loc='best', ncol=2)

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Yields.pdf'))
plt.show()
plt.close()


# Time Series of the PCs
df_plot = df_pca_full

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.plot(df_plot)


last_date = rate.index[-1]
plt.title(f"Principal Components of the DI Curve as of {last_date.strftime('%d/%b/%y')}")

ax = plt.gca()
ax.axhline(0, color='black', linewidth=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(rotation=90, axis='x')
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(df_plot.columns, frameon=True, loc='best')

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Principal Components.pdf'))
plt.show()
plt.close()

# Factor Loadings
df_plot = df_loadings_full

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.plot(df_plot)
plt.xlabel('Maturity (DU)')

last_date = rate.index[-1]
plt.title(f"Factor Loadings of the DI Curve as of {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
ax.axhline(0, color='black', linewidth=0.5)
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(df_plot.columns, frameon=True, loc='best')

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Factor Loadings.pdf'))
plt.show()
plt.close()

# Explained Variance
df_plot = df_var_full

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.bar([f'PC{a}' for a in range(1, 5)], df_plot.iloc[:, 0].values)
plt.plot(df_plot.iloc[:, 0].cumsum(), color='orange')

last_date = rate.index[-1]
plt.title(f"Explained Variance of the DI Curve as of {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
ax.axhline(0, color='black', linewidth=0.5)
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Explained Variance.pdf'))
plt.show()
plt.close()


# ===== Save Results =====
filename = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2024/DI1 PCA Output.xlsx')
with pd.ExcelWriter(filename) as writer:
    df_pca_full.to_excel(writer, sheet_name="PCs")
    df_loadings_full.to_excel(writer, sheet_name="Loadings")
    df_mean_full.to_excel(writer, sheet_name="Means")
    df_var_full.to_excel(writer, sheet_name="Variance")