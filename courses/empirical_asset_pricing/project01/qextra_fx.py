import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import chi2, f
from getpass import getuser
import numpy as np
import numpy.linalg as la
from statsmodels.regression.rolling import RollingOLS

# User parameters
file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Doutorado - Empirical Finance/Project 1")
show_charts = False

# Trackers
trackers = pd.read_csv(file_path.joinpath('fx_trackers.csv'), index_col=0)
trackers.index = pd.to_datetime(trackers.index)
trackers = trackers.resample('M').last().dropna()

carry = pd.read_csv(file_path.joinpath('HML_fx_factor.csv'), index_col=0)
carry.index = pd.to_datetime(carry.index)
carry = carry['fx_carry']
carry = carry.resample('M').last().dropna()

value = pd.read_csv(file_path.joinpath('Value_fx_factor.csv'), index_col=0)
value.index = pd.to_datetime(value.index)
value = value['fx_value']
value = value.resample('M').last().dropna()

# Build "market" factor
# mkt = trackers.pct_change(1).mean(axis=1)
# mkt = 1 + mkt
# mkt.iloc[0] = 1
# mkt = mkt.cumprod()

mkt = pd.read_csv(file_path.joinpath('credit.csv'), index_col=0)
mkt.index = pd.to_datetime(mkt.index)
mkt = mkt['CDX IG']

# concatenate factors
factors = pd.concat([mkt.rename('mkt'), carry.rename('carry'), value.rename('value')], axis=1)

# Summary Statistics
means = trackers.pct_change(1).mean()
fmeans = factors.pct_change(1).mean()


# ==========================
# ===== Estimate betas =====
# ==========================
betas = pd.DataFrame()

for ccy in trackers.columns:
    reg_data = pd.concat([trackers[ccy].rename(ccy), factors[['mkt', 'carry', 'value']]], axis=1)
    reg_data = reg_data.pct_change(1).dropna()

    model = sm.OLS(reg_data[ccy], sm.add_constant(reg_data[['mkt', 'carry', 'value']]))
    res = model.fit()

    betas.loc[ccy, "mkt"] = res.params["mkt"]
    betas.loc[ccy, "carry"] = res.params["carry"]
    betas.loc[ccy, "value"] = res.params["value"]

print(betas)


# =====================
# ===== 2nd Stage =====
# =====================
model = sm.OLS(means * 100, betas)
res = model.fit()
print(res.summary())


# ========================================
# ===== Chart - Predicted VS Realized ====
# ========================================
predicted = (betas * fmeans).sum(axis=1) * 100
realized = means * 100

fig = plt.figure(figsize=(5 * (16 / 9), 5))
ax = plt.subplot2grid((1, 1), (0, 0))

ax.scatter(predicted, realized)
for ccy in trackers.columns:
    ax.annotate(ccy, (predicted.loc[ccy] + 0.00005, realized.loc[ccy]))

ax.axline([0, 0], [0.1, 0.1], color="tab:orange", label='45-degree line')
ax.axhline(color='black', lw=0.5)
ax.axvline(color='black', lw=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Predict Average Monthly Excess Return")
ax.set_ylabel("Realized Average Monthly Excess Return")
ax.legend(frameon=True)

plt.tight_layout()

# plt.savefig(file_path.joinpath("figures/Q_extra Predicted VS Realized.pdf"))
if show_charts:
    plt.show()
plt.close()
