"""
Add vix as a factor
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import chi2, f
from getpass import getuser
import numpy as np
import numpy.linalg as la

# User parameters
file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Doutorado - Empirical Finance/Project 1")
show_charts = True


# ================
# ===== Data =====
# ================
# --- read portfolios ---
ff25 = pd.read_excel(file_path.joinpath("Dados.xlsx"),
                     skiprows=2, header=[0, 1], index_col=0, sheet_name="FF25")
ff25.index = pd.to_datetime(ff25.index)
ff25 = ff25.resample('M').last().astype(float)

# --- read FF factors ---
ff5f = pd.read_excel(file_path.joinpath("Dados.xlsx"),
                     index_col=0, sheet_name="Factors")
ff5f.index = pd.to_datetime(ff5f.index)
ff5f = ff5f.resample('M').last().astype(float)

# --- Execess Returns of the FF25 ---
ff25 = ff25.sub(ff5f['RF'], axis=0)

# --- Read the VIX ---
vix = pd.read_excel(file_path.joinpath("dados vix.xlsx"), sheet_name="Values", index_col=0)
vix.index = pd.to_datetime(vix.index)
vix = vix['VIX Index'].dropna().rename('VIX')
vix = vix.resample('M').mean()

# --- Orgnize Factors ---
factors = pd.concat([ff5f.drop('RF', axis=1), vix], axis=1)
factors = factors.dropna().astype(float)

# --- Summary ---
means = ff25.reindex(factors.index).mean()
fmeans = factors.reindex(factors.index).mean()


# =======================================================
# ===== First Stage - The 25 Timeseries Regressions =====
# =======================================================
betas = pd.DataFrame(index=ff25.columns, columns=["Mkt", "VIX"])
r2s = pd.Series(index=ff25.columns)
df_resids = pd.DataFrame()

for s in range(1, 6):
    for v in range(1, 6):
        reg_data = pd.concat([ff25[s][v].rename('Y'), factors[["Mkt", "VIX"]]], axis=1)
        reg_data = reg_data.dropna()

        Y = reg_data["Y"]
        X = sm.add_constant(reg_data[["Mkt", "VIX"]])
        model = sm.OLS(Y, X)
        res = model.fit()

        betas.loc[s, v] = res.params
        df_resids[f"S{s}V{v}"] = res.resid
        r2s.loc[s, v] = res.rsquared


# =======================================================
# ===== Second Stage - The Cross-Section Regression =====
# =======================================================
Sig = df_resids.cov()
SigF = factors[['Mkt', 'VIX']].cov()
df2nd = []
T = ff25.shape[0]
N = ff25.shape[1]
for add_cons in [True, False]:
    for estimator in ['OLS', 'GLS']:
        for test_assets in [True, False]:

            res = pd.Series({'Add Const': add_cons,
                             'Estimator': estimator,
                             'Include TA': test_assets})

            Y = means.copy().astype(float)
            X = betas.copy()

            # Inclusion of test asset
            S = Sig.copy()
            if test_assets:
                X.loc[('TA', 'Mkt'), 'Mkt'] = 1
                X.loc[('TA', 'Mkt'), 'VIX'] = 0
                Y.loc['TA', 'Mkt'] = fmeans.loc['Mkt']
                S.loc['Mkt'] = 0
                S['Mkt'] = 0
                S.loc['Mkt', 'Mkt'] = 1e-12

            # Add constant or not
            if add_cons:
                X = sm.add_constant(X)

            X = X.astype(float)

            # Estimator
            X = X.values
            Y = Y.values
            S = S.values
            if estimator == 'OLS':
                lambda_hat = la.inv(X.T @ X) @ (X.T @ Y)
                alpha_hat = Y - X @ lambda_hat

                shanken = 1 + lambda_hat[-2:].T @ la.inv(SigF) @ lambda_hat[-2:]
                quad_term = np.eye(N + 1 * test_assets) - X @ la.inv(X.T @ X) @ X.T
                var_alpha = (shanken/T) * quad_term @ S @ quad_term

                test_stat = T * (alpha_hat.T @ la.inv(var_alpha) @ alpha_hat)

            else:
                lambda_hat = la.inv(X.T @ la.inv(S) @ X) @ (X.T @ la.inv(S) @ Y)
                alpha_hat = Y - X @ lambda_hat

                shanken = 1 + lambda_hat[-2:].T @ la.inv(SigF) @ lambda_hat[-2:]
                var_alpha = (shanken / T) * (S - X @ la.inv(X.T @ la.inv(S) @ X) @ X.T)

                test_stat = T * shanken * (alpha_hat.T @ la.inv(S) @ alpha_hat)

            pval = 1 - chi2.cdf(test_stat, N + 1 * test_assets)

            cols = ["const", "mkt", "vix"] if add_cons else ["mkt", "vix"]
            lambda_hat = pd.Series(index=cols, data=lambda_hat)

            # --- Save Results ---
            res.loc['mkt'] = lambda_hat.loc["mkt"]
            res.loc['vix'] = lambda_hat.loc["vix"]

            try:
                res.loc['const'] = lambda_hat.loc["const"]
            except KeyError:
                res.loc['const'] = 0

            res.loc['test stat'] = test_stat
            res.loc['pval'] = pval

            df2nd.append(res)

df2nd = pd.concat(df2nd, axis=1).T
df2nd = df2nd.set_index(['Add Const', 'Estimator', 'Include TA'])

print(df2nd)
df2nd.to_clipboard()


# =======================================
# ===== Chart - Beta_Mkt VS Beta_VIX ====
# =======================================
fig = plt.figure(figsize=(5 * (16 / 9), 5))

ax = plt.subplot2grid((1, 1), (0, 0))

ax.scatter(betas['Mkt'], betas['VIX'])
for s in range(1, 6):
    for v in range(1, 6):
        ax.annotate(f"S{s}V{v}", (betas['Mkt'].loc[s,v] + 0.005, betas['VIX'].loc[s, v]))

ax.axhline(0, color='black', lw=0.5)
ax.axvline(1, color='black', lw=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel(r"Beta to Market Factor $\beta_{MKT}$")
ax.set_ylabel(r"Beta to VIX Index $\beta_{VIX}$")

plt.tight_layout()

plt.savefig(file_path.joinpath("figures/Q05 Betas.pdf"))
if show_charts:
    plt.show()
plt.close()


# ========================================
# ===== Chart - Predicted VS Realized ====
# ========================================
predicted = (fmeans * betas).sum(axis=1)
realized = means

fig = plt.figure(figsize=(5 * (16 / 9), 5))
ax = plt.subplot2grid((1, 1), (0, 0))

ax.scatter(predicted, realized)
for s in range(1, 6):
    for v in range(1, 6):
        ax.annotate(f"S{s}V{v}", (predicted.loc[s,v] + 0.005, realized.loc[s, v]))

xlims = ax.get_xlim()
ax.axline([0, 0], [1, 1], color="tab:orange")
ax.set_xlim(xlims)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Predict Average Monthly Excess Return")
ax.set_ylabel("Realized Average Monthly Excess Return")

plt.tight_layout()

plt.savefig(file_path.joinpath("figures/Q05 Predicted VS Realized.pdf"))
if show_charts:
    plt.show()
plt.close()
