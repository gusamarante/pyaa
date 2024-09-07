from scipy.stats import t, norm, laplace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


inc = 0.1
theta_grid = np.arange(
    start=-5,
    stop=5 + inc,
    step=inc,
)

t_pdf = t.pdf(theta_grid, df=3)
norm_pdf = norm.pdf(theta_grid, loc=0, scale=1)
laplace_pdf = laplace.pdf(theta_grid, loc=0, scale=1)


# Chart comparing candidate and target densities
size = 5
fig = plt.figure(figsize=(size * (16 / 7.3), size))

axn = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
axn.plot(theta_grid, t_pdf, label=f"t(3)", lw=2)
axn.plot(theta_grid, norm_pdf, label=f"N(0,1)", lw=2)
axn.set_ylim(0, None)

axwn = plt.subplot2grid((3, 2), (2, 0))
axwn.plot(theta_grid, norm_pdf / t_pdf, label=f"Weight / Density Ratio", lw=2)

axl = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
axl.plot(theta_grid, t_pdf, label=f"t(3)", lw=2)
axl.plot(theta_grid, laplace_pdf, label=f"Laplace(0,1)", lw=2)
axl.set_ylim(0, None)

axwl = plt.subplot2grid((3, 2), (2, 1))
axwl.plot(theta_grid, laplace_pdf / t_pdf, label=f"Weight / Density Ratio", lw=2)

for ax in [axn, axwn, axl, axwl]:
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"Probability Density")
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=True, loc="best")

plt.tight_layout()
plt.savefig("/Users/gustavoamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - Bayesiana/HW02/part 2 candidate target.pdf")
plt.show()
plt.close()



