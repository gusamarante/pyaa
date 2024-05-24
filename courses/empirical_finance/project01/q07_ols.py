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

# --- read factors ---
ff5f = pd.read_excel(file_path.joinpath("Dados.xlsx"),
                     index_col=0, sheet_name="Factors")
ff5f.index = pd.to_datetime(ff5f.index)

# --- Execess Returns of the FF25 ---
ff25 = ff25.sub(ff5f['RF'], axis=0)
ff25 = ff25[ff25.index >= "1963-07-01"]
ff5f = ff5f.drop('RF', axis=1)

# --- summary statistics ---
means = ff25.mean()
stds = ff25.std()

fmeans = ff5f.mean()

# --- Estimate betas ---
betas = pd.DataFrame(index=means.index, columns=ff5f.columns)
alphas = pd.Series(index=means.index)
ic_alpha = pd.DataFrame(columns=['LB', 'UB'], index=means.index)
df_resids = pd.DataFrame()

for s in range(1, 6):
    for v in range(1, 6):
        reg_data = pd.concat([ff25[s][v].rename('Y'), ff5f], axis=1)
        reg_data = reg_data.dropna()

        model = sm.OLS(reg_data["Y"], sm.add_constant(reg_data.drop("Y", axis=1)))
        res = model.fit()

        betas.loc[s, v] = res.params
        alphas.loc[s, v] = res.params["const"]
        ic_alpha.loc[(s, v), "LB"] = res.conf_int().loc["const", 0]
        ic_alpha.loc[(s, v), "UB"] = res.conf_int().loc["const", 1]
        df_resids[f"S{s}V{v}"] = res.resid


# --- Statistics ---
T = ff5f.shape[0]
N = ff25.shape[1]
K = ff5f.shape[1]
print("sample size T", T)
print("sample size N", N)

alpha_hat = alphas.values
sigma_hat = df_resids.cov().values
omega_hat = ff5f.cov().values
quad_term = alpha_hat @ la.inv(sigma_hat) @ alpha_hat
denom = 1 + fmeans @ la.inv(omega_hat) @ fmeans

GRS_stat = (((T - N - K) / N) / denom) * quad_term
GRS_crit = f.ppf(0.95, N, T - N - K)
GRS_pval = 1 - f.cdf(GRS_stat, N, T - N - K)
print("GRS Stat", GRS_stat)
print("GRS Critical value 95%", GRS_crit)
print("GRS p-value", GRS_pval)


# ================================
# ===== Chart - Alphas t-test ====
# ================================
fig = plt.figure(figsize=(5 * (16 / 9), 5))

#  --- Bar ---
ax = plt.subplot2grid((1, 2), (0, 0))
ax = alphas.plot(kind='bar', ax=ax, width=0.9)
ax.axhline(0, color="black", lw=0.5)

ax.errorbar(ax.get_xticks(), alphas.values,
            yerr=((ic_alpha["UB"] - ic_alpha["LB"]) / 2).values,
            ls='none', ecolor='tab:orange')

#  --- Heatmap ---
alphas_table = alphas.reset_index().rename({0: 'Alpha'}, axis=1)
alphas_table = alphas_table.pivot(index='Size', columns='Value', values='Alpha')
alphas_table = alphas_table.sort_index(ascending=False)
ax = plt.subplot2grid((1, 2), (0, 1))
ax = sns.heatmap(
    alphas_table,
    ax=ax,
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap="RdBu",
    annot_kws={"fontsize": 12, "weight": "normal"},
    linewidths=1,
    linecolor="lightgrey",
    center=0,  # Force 0 to be white
)
ax.tick_params(rotation=0, axis="y")
plt.tick_params(axis="x", which="both", top=False, bottom=False)
plt.tick_params(axis="y", which="both", left=False)


plt.tight_layout()

plt.savefig(file_path.joinpath("figures/Q07 Alphas to FF5F.pdf"))
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

plt.savefig(file_path.joinpath("figures/Q07 OLS Predicted VS Realized.pdf"))
if show_charts:
    plt.show()
plt.close()
