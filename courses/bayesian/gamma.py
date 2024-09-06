from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

size = 4
params = [(2, 1), (2, 2), (3, 2), (2, 3)]

inc = 0.01
theta_grid = np.arange(
    start=0,
    stop=5 + inc,
    step=inc,
)

# item 2
# fig = plt.figure(figsize=(size * (16 / 7.3), size))
# ax = plt.subplot2grid((1, 1), (0, 0))
# for a, b in params:
#     gamma_pdf = gamma.pdf(theta_grid, a=a, scale=1/b)
#     ax.plot(theta_grid, gamma_pdf, label=f"G({a},{b})", lw=2)
#
# # ax.axvline(theta_mle, label=f"MLE", lw=2, color=color_palette['yellow'], ls='--')
# ax.set_xlim(0, None)
# ax.set_ylim(0, None)
# ax.set_xlabel(r"$\theta$")
# ax.set_ylabel(r"Probability Density")
# ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.legend(frameon=True, loc="best")
#
#
# plt.tight_layout()
# plt.savefig("/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - Bayesiana/HW02/part 1 gamma plot.pdf")
# plt.show()
# plt.close()


# Itens 3 and 4
m_sizes = [1000, 10000, 100000]
r_times = 100

results = pd.DataFrame()
for a, b in params:
    for m in m_sizes:
        sample = np.random.gamma(shape=a, scale=1/b, size=(m, r_times))
        mean_r = sample.mean(axis=0)
        var_r = sample.var(axis=0)
        aux = pd.DataFrame(
            data={
                "Mean": mean_r,
                "Variance": var_r,
                "Dist": f"Gamma({a}, {b})",
                "MC Size": m,
            }
        )
        results = pd.concat([results, aux])

size = 6
fig = plt.figure(figsize=(size * (16 / 12), size))
dists = results["Dist"].unique()
for n, (a, b) in enumerate(params):
    ax = plt.subplot2grid((2, len(dists)), (0, n))
    sns.boxenplot(
        results[results["Dist"] == f"Gamma({a}, {b})"], x="MC Size", y="Mean", ax=ax
    )
    ax.axhline(a/b, color="tab:orange", ls="--")
    ax.tick_params(rotation=90, axis="x")
    ax.set_title(f"Gamma({a}, {b})")


    ax = plt.subplot2grid((2, len(dists)), (1, n))
    sns.boxenplot(
        results[results["Dist"] == f"Gamma({a}, {b})"], x="MC Size", y="Variance", ax=ax
    )
    ax.axhline(a / (b**2), color="tab:orange", ls="--")
    ax.tick_params(rotation=90, axis="x")
    ax.set_title(f"Gamma({a}, {b})")

plt.tight_layout()
plt.savefig("/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - Bayesiana/HW02/part 1 MC Convergence.pdf")
plt.show()
