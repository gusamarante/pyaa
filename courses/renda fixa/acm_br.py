"""
Run the ACM Term premium model for brazil
"""
from sklearn.decomposition import PCA
from data.xls_data import curve_di, trackers_di
import matplotlib.dates as mdates
from models.acm import NominalACM
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


size = 5
curve = curve_di()
trackers = trackers_di()
xrets = trackers.pct_change(1).dropna()
curve = curve.reindex(xrets.index)

curve.columns = [i + 1 for i in range(curve.shape[1])]
xrets.columns = [i + 1 for i in range(xrets.shape[1])]

acm = NominalACM(
    curve=curve,
    excess_returns=xrets,
    compute_miy=True,
)


# ====================================
# ===== CHART - Fixed Maturities =====
# ====================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(curve, label=curve.columns, lw=2)
ax.set_title("DI Futures - Fixed Maturities")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best", ncols=2)

plt.tight_layout()

# plt.savefig("/Users/gamarante/Desktop/DI Curve.pdf")
plt.show()
plt.close()


# ===========================
# ===== Full sample PCA =====
# ===========================
pca = PCA(n_components=5)
pca.fit(curve)

col_names = [f'PC {i+1}' for i in range(5)]
df_var = pd.DataFrame(data={'Marginal': pca.explained_variance_ratio_,
                            'Cumulative': pca.explained_variance_ratio_.cumsum()},
                      index=col_names)
df_loadings = pd.DataFrame(data=pca.components_.T,
                           columns=col_names,
                           index=curve.columns)
df_mean = pd.DataFrame(data=pca.mean_,
                       index=curve.columns,
                       columns=['Médias'])
df_pca = pd.DataFrame(data=pca.transform(curve.values),
                      columns=col_names,
                      index=curve.index)

signal = np.sign(df_loadings.iloc[-1])
df_loadings = df_loadings * signal
df_pca = df_pca * signal


# ======================================
# ===== CHART - Explained Variance =====
# ======================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax = df_var['Marginal'].plot(kind='bar', color="#3333B2", ax=ax, label='Marginal')
ax.set_title("Variance Explained by the PCs")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.tick_params(rotation=90, axis="x")

for p in ax.patches:
    ax.annotate(str(round(100*p.get_height(), 2)), (p.get_x() + 0.1, p.get_height() + 0.01))


ax = plt.subplot2grid((1, 2), (0, 1))
ax.plot(df_loadings, label=df_loadings.columns, lw=2)
ax.set_title("Factor Loadings")
ax.axhline(0, lw=0.5, color="black")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best", ncols=2)


plt.tight_layout()

# plt.savefig("/Users/gamarante/Desktop/DI Curve PCA.pdf")
plt.show()
plt.close()
