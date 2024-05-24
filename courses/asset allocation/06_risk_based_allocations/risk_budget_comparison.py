import numpy as np
import pandas as pd
from pathlib import Path
from getpass import getuser
import matplotlib.pyplot as plt
from allocation import RiskBudgetVol, HRP


file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation")
df = pd.read_excel(file_path.joinpath('Commodities Total Return.xlsx'), index_col=0)
df.index = pd.to_datetime(df.index)
df = df.resample('M').last()
df = df.pct_change(1, fill_method=None)
df = df.dropna()
cov = df.cov()

invvar = 1 / np.diag(cov)
ivp = pd.Series(name='InvVar',
                data=invvar / invvar.sum(),
                index=cov.index)
erc = RiskBudgetVol(cov=cov)
hrp = HRP(cov=cov)

all_weights = pd.concat([ivp, hrp.weights, erc.weights.rename('ERC')], axis=1)
weight_order = all_weights.mean(axis=1).sort_values(ascending=False).index
all_weights = all_weights.loc[weight_order]

# =================
# ===== Chart =====
# =================
fig = plt.figure(figsize=(5 * (16 / 7), 5))
plt.suptitle("Comparing Risk Budget Weights", fontweight="bold")
ax = plt.subplot2grid((1, 1), (0, 0))

ax = all_weights.plot(kind='bar', ax=ax)

ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel(r"Commodities")
ax.set_ylabel(r"Weight (%)")
ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig(file_path.joinpath("Figures/Risk Budget - Weight Comparison.pdf"))
plt.show()
plt.close()
