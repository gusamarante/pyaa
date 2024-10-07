import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

data = pd.read_excel('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/Dados.xlsx',
                   sheet_name="com_na")
data = data.set_index(['patient', 'visit'])

gibbs = pd.read_csv('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/samples.csv',
                 index_col=0)
gibbs = gibbs.iloc[1000:].reset_index(drop=True)


patients = [1, 2, 4]
pred = pd.DataFrame()
for row in tqdm(gibbs.iterrows()):
    for p in patients:
        pred.loc[row[0], f"patient {p}"] = row[1].loc[f"alpha {p}"] + row[1].loc[f"beta {p}"] * data.loc[(p, 7), "gestational_age"]


# ===== POSTERIORS =====
size = 5
fig = plt.figure(figsize=(size * (16 / 9), size))

for n, p in enumerate(patients):
    # alpha_i posterior
    ax = plt.subplot2grid((3, 3), (n, 0))
    ax.hist(gibbs[f'alpha {p}'], density=True, bins=40)
    ax.set_title(fr"$\alpha_{p}$")
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

    # beta_i posterior
    ax = plt.subplot2grid((3, 3), (n, 1))
    ax.hist(gibbs[f'beta {p}'], density=True, bins=40)
    ax.set_title(fr"$\beta_{p}$")
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

    # Posterior predictive
    ax = plt.subplot2grid((3, 3), (n, 2))
    ax.hist(pred[f'patient {p}'], density=True, bins=40)
    ax.set_title(rf"$y_{p}$")
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)


plt.tight_layout()
plt.savefig("/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/figures/Item E posteriors.pdf")
plt.show()
plt.close()


# ===== POSTERIORS =====
size = 5
fig = plt.figure(figsize=(size * (16 / 9), size))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_xlabel("gestational_age")
ax.set_ylabel("weight_gain")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

colors = {
    1: "tab:blue",
    2: "tab:orange",
    4: "tab:green",
}
for n, p in enumerate(patients):
    ax.plot(data.loc[p, "gestational_age"], data.loc[p, "weight_gain"], color=colors[p], lw=2, label=f"Patient {p}")

    # Median
    ax.plot(
        [data.loc[(p, 6), "gestational_age"], data.loc[(p, 7), "gestational_age"]],
        [data.loc[(p, 6), "weight_gain"], pred[f'patient {p}'].quantile(0.5)],
        color=colors[p],
        ls='dashed',
        lw=2,
    )

    # 95% CI
    ax.fill_between(
        x=[data.loc[(p, 6), "gestational_age"], data.loc[(p, 7), "gestational_age"]],
        y1=[data.loc[(p, 6), "weight_gain"], pred[f'patient {p}'].quantile(0.975)],
        y2=[data.loc[(p, 6), "weight_gain"], pred[f'patient {p}'].quantile(0.025)],
        color=colors[p],
        alpha=0.3,
    )

ax.legend(frameon=True, loc='upper left')


plt.tight_layout()
plt.savefig("/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/figures/Item E proj.pdf")
plt.show()
plt.close()

