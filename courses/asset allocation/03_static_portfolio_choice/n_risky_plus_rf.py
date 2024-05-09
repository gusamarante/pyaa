import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from getpass import getuser


rf = 0.05
mu = pd.Series(
    data={
        "A": 0.0606,
        "B": 0.1065,
        "C": 0.11,
        "D": 0.085,
    }
)
muf = mu - rf

vols = pd.Series(
    data={
        "A": 0.0888,
        "B": 0.2255,
        "C": 0.1064,
        "D": 0.15,
    }
)

corr = pd.DataFrame(
    columns=vols.index,
    index=vols.index,
    data=[[1, -0.3, -0.2, 0],
          [-0.3, 1, 0.5, 0],
          [-0.2, 0.5, 1, 0],
          [0, 0, 0, 1]],
)

cov = np.diag(vols) @ corr @ np.diag(vols)
cov = pd.DataFrame(
    columns=vols.index,
    index=vols.index,
    data=cov.values,
)

invGamma = np.linalg.inv(cov)

numer = invGamma @ muf
denom = muf.T @ invGamma @ muf

df = pd.DataFrame()
for idx, mu_bar in enumerate(np.linspace(min(mu.min() - 0.05, 0), mu.max() + 0.05, 500)):
    w = ((mu_bar - rf) / denom) * numer
    df.loc[idx, 'mu'] = mu_bar
    df.loc[idx, 'sigma'] = np.sqrt(w.T @ cov @ w)

# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(4 * (16 / 7.3), 4))

# Expected returns
ax = plt.subplot2grid((1, 1), (0, 0))

ax.plot(df['sigma'].values, df['mu'].values, color='#3333B2', zorder=-1, label="Min Variance Frontier")
ax.scatter(vols, mu, label='Assets', color="#F25F5C", s=50)
# ax.scatter(1, mu1, label='Asset 1', color="#0B6E4F", s=50)

ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$\sigma_{p}$")
ax.set_ylabel(r"$\mu_{p}$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")


plt.tight_layout()

file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation/Figures")
plt.savefig(file_path.joinpath("Static Portfolio Choice - Many Risky plus riskless.pdf"))
plt.show()
plt.close()
