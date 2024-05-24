"""
- Plot monthly average returns for each of the 25 FF portfolios
- Plot Market betas for each portfolio, estimated using the Market Model
- Plot Average Excess Return against Market Beta
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from getpass import getuser

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

# --- Manipulations ---
ff25 = ff25.sub(ff5f['RF'], axis=0)  # generate excess returns
ff25 = ff25[ff25.index >= "1963-07-01"]  # Filter dates

ff5f = ff5f.drop('RF', axis=1)  # drop RF from factors

# --- summary statistics ---
means = ff25.mean()
stds = ff25.std()

# --- Estimate betas ---
betas = pd.Series(index=means.index)
alphas = pd.Series(index=means.index)
r2s = pd.Series(index=means.index)

for s in range(1, 6):
    for v in range(1, 6):
        reg_data = pd.concat([ff25[s][v].rename('Y'), ff5f["Mkt"].rename('X')], axis=1)
        reg_data = reg_data.dropna()

        model = sm.OLS(reg_data["Y"], sm.add_constant(reg_data["X"]))
        res = model.fit()

        betas.loc[s, v] = res.params["X"]
        alphas.loc[s, v] = res.params["const"]
        r2s.loc[s, v] = res.rsquared


# ==========================================================
# ===== Chart - Heatmap of Mean and SD Monthly Returns =====
# ==========================================================
mean_table = means.reset_index().rename({0: 'Mean'}, axis=1)
mean_table = mean_table.pivot(index='Size', columns='Value', values='Mean')
mean_table = mean_table.sort_index(ascending=False)

sd_table = stds.reset_index().rename({0: 'Mean'}, axis=1)
sd_table = sd_table.pivot(index='Size', columns='Value', values='Mean')
sd_table = sd_table.sort_index(ascending=False)

fig = plt.figure(figsize=(5 * (16 / 9), 5))

#  --- Mean ---
ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Average of Monthly Excess Returns")
ax = sns.heatmap(
    mean_table,
    ax=ax,
    cbar=False,
    annot=True,
    fmt=".2f",
    cmap="RdBu",
    annot_kws={"fontsize": 12, "weight": "normal"},
    linewidths=1,
    linecolor="lightgrey",
    center=mean_table.min().min(),  # Force lowest value to be white
)
ax.tick_params(rotation=0, axis="y")
plt.tick_params(axis="x", which="both", top=False, bottom=False)
plt.tick_params(axis="y", which="both", left=False)

#  --- SD ---
ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("Standard Deviation of Monthly Excess Returns")
ax = sns.heatmap(
    sd_table,
    ax=ax,
    cbar=False,
    annot=True,
    fmt=".2f",
    cmap="RdBu",
    annot_kws={"fontsize": 12, "weight": "normal"},
    linewidths=1,
    linecolor="lightgrey",
    center=sd_table.min().min(),  # Force lowest value to be white
)
ax.tick_params(rotation=0, axis="y")
plt.tick_params(axis="x", which="both", top=False, bottom=False)
plt.tick_params(axis="y", which="both", left=False)


plt.tight_layout()

plt.savefig(file_path.joinpath("figures/Q01 Mean and SD of Monthly Returns Heatmap.pdf"))
if show_charts:
    plt.show()
plt.close()


# ===============================================
# ===== Chart - Bar of Mean Monthly Returns =====
# ===============================================
fig = plt.figure(figsize=(5 * (16 / 9), 5))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Average of Monthly Excess Returns")
ax = means.plot(kind='bar', ax=ax)

ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("Standard Deviation of Monthly Excess Returns")
ax = stds.plot(kind='bar', ax=ax)

plt.tight_layout()

plt.savefig(file_path.joinpath("figures/Q01 Mean and SD of Monthly Returns Bar.pdf"))
if show_charts:
    plt.show()
plt.close()


# ======================================
# ===== Chart - Betas for the FF25 =====
# ======================================
fig = plt.figure(figsize=(5 * (16 / 9), 5))

#  --- Bar ---
ax = plt.subplot2grid((1, 2), (0, 0))
# ax.set_title("Betas of the Market Model")
ax = betas.plot(kind='bar', ax=ax)

#  --- Heatmap ---
betas_table = betas.reset_index().rename({0: 'Beta'}, axis=1)
betas_table = betas_table.pivot(index='Size', columns='Value', values='Beta')
betas_table = betas_table.sort_index(ascending=False)
ax = plt.subplot2grid((1, 2), (0, 1))
# ax.set_title("Standard Deviation of Monthly Returns")
ax = sns.heatmap(
    betas_table,
    ax=ax,
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap="RdBu",
    annot_kws={"fontsize": 12, "weight": "normal"},
    linewidths=1,
    linecolor="lightgrey",
    center=1,  # Force 1 to be white
)
ax.tick_params(rotation=0, axis="y")
plt.tick_params(axis="x", which="both", top=False, bottom=False)
plt.tick_params(axis="y", which="both", left=False)


plt.tight_layout()

plt.savefig(file_path.joinpath("figures/Q01 Betas to Mkt.pdf"))
if show_charts:
    plt.show()
plt.close()


# ==================================
# ===== Chart - R2 for the FF25 ====
# ==================================
fig = plt.figure(figsize=(5 * (16 / 9), 5))

#  --- Bar ---
ax = plt.subplot2grid((1, 2), (0, 0))
# ax.set_title("Betas of the Market Model")
ax = r2s.plot(kind='bar', ax=ax)
ax.axhline(0, color="black", lw=0.5)

#  --- Heatmap ---
r2_table = r2s.reset_index().rename({0: 'R2'}, axis=1)
r2_table = r2_table.pivot(index='Size', columns='Value', values='R2')
r2_table = r2_table.sort_index(ascending=False)
ax = plt.subplot2grid((1, 2), (0, 1))
# ax.set_title("Standard Deviation of Monthly Returns")
ax = sns.heatmap(
    r2_table,
    ax=ax,
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap="RdBu",
    annot_kws={"fontsize": 12, "weight": "normal"},
    linewidths=1,
    linecolor="lightgrey",
    center=r2_table.min().min(),  # Force 1 to be white
)
ax.tick_params(rotation=0, axis="y")
plt.tick_params(axis="x", which="both", top=False, bottom=False)
plt.tick_params(axis="y", which="both", left=False)


plt.tight_layout()

plt.savefig(file_path.joinpath("figures/Q01 R2 to Mkt.pdf"))
if show_charts:
    plt.show()
plt.close()


# ===============================================
# ===== Chart - ER against beta for the FF25 ====
# ===============================================
fig = plt.figure(figsize=(5 * (16 / 9), 5))

ax = plt.subplot2grid((1, 1), (0, 0))

ax.scatter(betas, means)
for s in range(1, 6):
    for v in range(1, 6):
        ax.annotate(f"S{s}V{v}", (betas.loc[s,v] + 0.005, means.loc[s, v]))

ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Estimated Market Beta")
ax.set_ylabel("Average Excess Return")

plt.tight_layout()

plt.savefig(file_path.joinpath("figures/Q01 Scatter ER against Beta.pdf"))
if show_charts:
    plt.show()
plt.close()
