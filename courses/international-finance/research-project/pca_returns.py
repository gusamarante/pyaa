import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import BLUE
import getpass


username = getpass.getuser()

df = pd.read_excel(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                   sheet_name='CDS Trackers',
                   index_col=0)
df.index = pd.to_datetime(df.index)

df = df.drop(
    [
        "ARGENT",
        "RUSSIA",
        "UKRAIN",
        "VENZ",
    ],
    axis=1,
)

df = df.resample("M").last()
df = df.dropna()

pca = PCA(n_components=5)
pca.fit(df)

var_raio = pd.DataFrame(data=pca.explained_variance_ratio_)
loadings = pd.DataFrame(data=pca.components_.T,
                        columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'],
                        index=df.columns)
pcs = pd.DataFrame(data=pca.transform(df.values),
                   columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'],
                   index=df.index)


# =================
# ===== Chart =====
# =================
size = 5
fig = plt.figure(figsize=(size * (16 / 7.3), size))

plotpc2 = loadings['PC 2'].sort_values()
plotpc1 = loadings['PC 1'].loc[plotpc2.index]
ax = plt.subplot2grid((2, 1), (0, 0))
ax.bar(plotpc1.index, plotpc1.values, color=BLUE)
ax.tick_params(rotation=90, axis="x")
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

ax = plt.subplot2grid((2, 1), (1, 0))
ax.bar(plotpc2.index, plotpc2.values, color=BLUE)
ax.tick_params(rotation=90, axis="x")
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.show()