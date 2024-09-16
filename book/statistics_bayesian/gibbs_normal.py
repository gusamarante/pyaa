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

n_samples = 50_000  # Number of sample from posterior

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
    sig2 = invgamma.rvs(a=A, scale=B)

    samples.loc[s, "mu"] = mu
    samples.loc[s, "sigma"] = np.sqrt(sig2)


# =================
# ===== Chart =====
# =================
g = sns.JointGrid(data=samples, x="mu", y="sigma")

g.plot_joint(sns.scatterplot, alpha=0.4)

g.plot_marginals(sns.histplot, stat='density', edgecolor=None)

mu_grid = np.linspace(start=samples["mu"].min(), stop=samples["mu"].max(), num=200)
num = y.sum() / y.var() + gamma / tau
denom = n / sig2 + 1 / tau
mu_pdf = norm.pdf(mu_grid, loc=num/denom, scale=1/denom)
g.ax_marg_x.plot(mu_grid, mu_pdf, color="tab:red")

sigma_grid = np.linspace(start=0, stop=samples["sigma"].max(), num=200)
A = n / 2 + a
B = ((y - y.mean()) ** 2).sum() / 2 + b
sigma_pdf = invgamma.pdf(sigma_grid**2, a=A, scale=B)
g.ax_marg_y.plot(sigma_pdf, sigma_grid, color="tab:red")

g.ax_joint.set_xlabel(r"$\mu$")
g.ax_joint.set_ylabel(r"$\sigma$")

plt.tight_layout()
plt.savefig(STATISTICS_BAYESIAN.joinpath("Gibbs - Normal Joint Example.pdf"))
plt.show()
