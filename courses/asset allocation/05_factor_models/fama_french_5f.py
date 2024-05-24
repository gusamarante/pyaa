"""
Example of multifactor model

Data from Ken French's website
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

from data.fama_french.ff_reader import get_ff5f, get_ff25p, get_ffrf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from getpass import getuser
from pathlib import Path
import seaborn as sns
import pandas as pd


file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation/Figures")

# =========================
# ===== Data Handling =====
# =========================
# Fama-French Portfolios - Excess Returns
ports = get_ff25p()
rf = get_ffrf()
ports = ports.sub(rf, axis=0)
ports = ports.dropna()

# Fama-French Factors
factors = get_ff5f()


# =====================================
# ===== Estimate Alphas and Betas =====
# =====================================
betas = pd.DataFrame(index=ports.columns, columns=factors.columns)
alphas = pd.Series(index=ports.columns, dtype=float, name='alpha')
r2s = pd.Series(index=ports.columns, dtype=float, name='r-squared')
ic_alpha = pd.DataFrame(columns=['LB', 'UB'], index=ports.columns)
df_resids = pd.DataFrame()

for s in range(1, 6):
    for v in range(1, 6):
        model = sm.OLS(ports[s][v], sm.add_constant(factors))
        res = model.fit()

        betas.loc[s, v] = res.params
        alphas.loc[s, v] = res.params["const"]
        r2s.loc[s, v] = res.rsquared
        ic_alpha.loc[(s, v), "LB"] = res.conf_int().loc["const", 0]
        ic_alpha.loc[(s, v), "UB"] = res.conf_int().loc["const", 1]
        df_resids[f"S{s}V{v}"] = res.resid


# ================================
# ===== Chart - Alphas t-test ====
# ================================
fig = plt.figure(figsize=(5 * (16 / 7), 5))
plt.suptitle("Alphas of the Fama-French 5-Factor Model", fontweight="bold")

# --- Bar ---
ax = plt.subplot2grid((1, 2), (0, 0))
ax = alphas.plot(kind='bar', ax=ax, width=0.9)
ax.axhline(0, color="black", lw=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

ax.errorbar(ax.get_xticks(), alphas.values,
            yerr=((ic_alpha["UB"] - ic_alpha["LB"]) / 2).values,
            ls='none', ecolor='tab:orange')

#  --- Heatmap ---
alphas_table = alphas.reset_index()
alphas_table = alphas_table.pivot(index='size', columns='value', values='alpha')
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
plt.savefig(file_path.joinpath("Factor Models - FF25 Alphas to FF5F.pdf"))
plt.show()
plt.close()


# ========================================
# ===== Chart - Predicted VS Realized ====
# ========================================
predicted = (betas * factors.mean()).sum(axis=1)
realized = ports.mean()

fig = plt.figure(figsize=(5 * (16 / 7), 5))
ax = plt.subplot2grid((1, 1), (0, 0))

ax.scatter(predicted, realized, label="Fama-French 25 Portfolios")
for s in range(1, 6):
    for v in range(1, 6):
        ax.annotate(f"S{s}V{v}", (predicted.loc[s, v] + 0.005, realized.loc[s, v]))

# xlims = ax.get_xlim()
ax.axline([0, 0], [1, 1], color="tab:orange", label="45-degree line")
ax.set_xlim((0, 1.2))
ax.set_ylim((0, 1.2))
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Predict Average Monthly Excess Return")
ax.set_ylabel("Realized Average Monthly Excess Return")
ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig(file_path.joinpath("Factor Models - FF5F predicted VS realized.pdf"))
plt.show()
plt.close()
