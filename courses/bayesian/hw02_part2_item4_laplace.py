from scipy.stats import t, norm, laplace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

m_values = [
    500,
    5000,
    50000,
    500000,
]
n_values = [
    100,
    1000,
    10000,
    100000,
]
r_times = 1000

df = []
for m in m_values:
    for n in n_values:
        for r in tqdm(range(r_times), f"M={m}, N={n}"):
            sample = t.rvs(df=3, size=m)
            w_a = laplace.pdf(sample, loc=0, scale=1) / t.pdf(sample, df=3)
            w_a = w_a / w_a.sum()
            draws_a = np.random.choice(sample, p=w_a, size=n, replace=True)

            aux = pd.Series({
                "M": m,
                "N": n,
                "prob": (draws_a >= 2).mean(),
                "r": r,
            })
            df.append(aux)

df = pd.concat(df, axis=1).T
df["M"] = df["M"].astype(int)
df["N"] = df["N"].astype(int)


# =================
# ===== Chart =====
# =================
size = 9
fig = plt.figure(figsize=(size * (12 / 16), size))
for en, m in enumerate(m_values):
    ax = plt.subplot2grid((len(m_values), 1), (en, 0))
    sns.boxenplot(
        df[df["M"] == m], x="N", y="prob", ax=ax
    )
    ax.axhline(0.06766196, color="tab:orange", ls="--")
    ax.tick_params(rotation=90, axis="x")
    ax.set_title(f"M={m}")
    ax.set_ylim(0, None)


plt.tight_layout()
plt.savefig("/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - Bayesiana/HW02/part2 item4 laplace.pdf")
plt.show()
