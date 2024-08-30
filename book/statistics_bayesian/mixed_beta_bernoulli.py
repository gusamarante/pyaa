"""
Plot the prior and posterior of a mixed beta with 2 components and bernouli
likelihood

This is originally from a problem set I did in bayesian econometrics
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from scipy.special import gamma
from scipy.stats import beta

from utils import color_palette, STATISTICS_BAYESIAN

size = 4
n = 10
sn = 4

inc = 1 / 500  # step size for theta grid
theta_grid = np.arange(start=0, stop=1 + inc, step=inc)

# --- Prior ---
a, b = 2, 5
c, d = 5, 2
pi = 0.5

original_prior = beta.pdf(x=theta_grid, a=a, b=b)
mixed_prior = pi * beta.pdf(x=theta_grid, a=a, b=b) + (1 - pi) * beta.pdf(x=theta_grid, a=c, b=d)

# --- Posterior ---
k1 = pi * gamma(a + b) * gamma(sn + a) * gamma(n - sn + b) / (gamma(a) * gamma(b) * gamma(a + n + b))
k2 = (1 - pi) * gamma(c + d) * gamma(sn + c) * gamma(n - sn + d) / (gamma(c) * gamma(d) * gamma(c + n + d))
pi1 = k1 / (k1 + k2)

a1 = a + sn
b1 = n - sn + b
c1 = c + sn
d1 = n - sn + d

original_posterior = beta.pdf(x=theta_grid, a=a1, b=b1)
mixed_posterior = pi1 * beta.pdf(x=theta_grid, a=a1, b=b1) + (1 - pi1) * beta.pdf(x=theta_grid, a=c1, b=d1)


# =====================================================
# ===== Chart - Observed, Fitted and Risk-Neutral =====
# =====================================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(theta_grid, original_prior, label=f"Original Prior", lw=2, color=color_palette['blue'], ls='--')
ax.plot(theta_grid, original_posterior, label=f"Original Posterior", lw=2, color=color_palette['blue'])
ax.plot(theta_grid, mixed_prior, label=f"Mixed Prior", lw=2, color=color_palette['red'], ls='--')
ax.plot(theta_grid, mixed_posterior, label=f"Mixed Posterior", lw=2, color=color_palette['red'])
ax.set_xlim(0, 1)
ax.set_ylim(0, None)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"Probability Density")
loc = plticker.MultipleLocator(base=0.1)
ax.xaxis.set_major_locator(loc)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")

plt.tight_layout()
plt.savefig(STATISTICS_BAYESIAN.joinpath("Bayes - Mixed Beta Bernoulli prior posterior.pdf"))
plt.show()
plt.close()
