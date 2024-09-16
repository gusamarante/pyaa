"""
Y ~ N(mu, sigma2)
mu ~ N(gamma, tau2)
sigma2 ~ IG(a, b)

Use gibbs sampling to find the JOINT posterior for theta = (mu, sigma2)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import invgamma, norm
from tqdm import tqdm
import seaborn as sns
from utils import STATISTICS_BAYESIAN

y = np.array([2.68, 1.18, -0.97, -0.98, -1.03])
n = len(y)

n_samples = 10_000  # Number of sample from posterior

# Initial Values
mu = y.mean()
sig2 = y.var()

# Prior parameters
gamma, tau = 0, 10000
a, b = 0.1, 0.1


# Gibbs sampling
samples = pd.DataFrame()
for s in tqdm(range(n_samples)):
    num = y.sum() / sig2 + gamma / tau
    denom = n / sig2 + 1 / tau
    mu = norm.rvs(
        loc=num / denom,
        scale=1 / np.sqrt(denom),
    )

    A = n / 2 + a
    B = ((y - mu)**2).sum() / 2 + b
    sig2 = invgamma.rvs(a=A, scale=B)  # TODO this might be wrong

    samples.loc[s, "mu"] = mu
    samples.loc[s, "sigma"] = np.sqrt(sig2)


# =================
# ===== Chart =====
# =================
g = sns.jointplot(data=samples, x="mu", y="sigma", alpha=0.3)
g.plot_joint(sns.kdeplot, color="tab:orange", zorder=1, levels=8)
g.ax_joint.set_xlabel(r"$\mu$")
g.ax_joint.set_ylabel(r"$\sigma$")

plt.tight_layout()
plt.savefig(STATISTICS_BAYESIAN.joinpath("Gibbs - Normal Joint Example.pdf"))
plt.show()

# TODO Compare conditional posteriors with marginal posteriors
