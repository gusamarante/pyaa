"""
Combination of 2 risky assets

Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
FFBA08 - Yellow
F25F5C - Red
"""

import matplotlib.pyplot as plt
from pathlib import Path
from getpass import getuser
import numpy as np


mu1 = 0.15
sigma1 = 0.2

mu2 = 0.25
sigma2 = 0.35

corr12 = -0.1
cov12 = sigma1 * sigma2 * corr12

w1 = np.linspace(-0.4, 1.4, 50)

mu_w = mu1 * w1 + mu2 * (1 - w1)
sigma_w = np.sqrt((w1 * sigma1)**2 + ((1 - w1) * sigma2)**2 + w1 * (1 - w1) * sigma1 * sigma2 * corr12)


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(5 * (16 / 7.3), 5))

# Expected returns
ax = plt.subplot2grid((2, 2), (0, 0))

ax.plot(w1, mu_w, color='#3333B2', zorder=-1)
ax.scatter(0, mu2, label='Asset 2', color="#F25F5C", s=50)
ax.scatter(1, mu1, label='Asset 1', color="#0B6E4F", s=50)

ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$w_{1}$")
ax.set_ylabel(r"$\mu_{p}$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")


# Volatility
ax = plt.subplot2grid((2, 2), (1, 0))

ax.plot(w1, sigma_w, color='#3333B2', zorder=-1)
ax.scatter(0, sigma2, label='Asset 2', color="#F25F5C", s=50)
ax.scatter(1, sigma1, label='Asset 1', color="#0B6E4F", s=50)

ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$w_{1}$")
ax.set_ylabel(r"$\sigma_{p}$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")


# Mean-Variance
ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

ax.plot(sigma_w, mu_w, color='#3333B2', zorder=-1)
ax.scatter(sigma2, mu2, label='Asset 2', color="#F25F5C", s=50)
ax.scatter(sigma1, mu1, label='Asset 1', color="#0B6E4F", s=50)

ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$\sigma_{p}$")
ax.set_ylabel(r"$\mu_{p}$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")


plt.tight_layout()

file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation/Figures")
plt.savefig(file_path.joinpath("Static Portfolio Choice - Two risky.pdf"))
plt.show()
plt.close()
