"""
Introductory example
Beta prior + bernoulli likelihood = beta posterior
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from utils import color_palette, STATISTICS_BAYESIAN


size = 5
inc = 1 / 500  # Granularity of theta grid
prop = 0.4
n_small = 10
sn_small = n_small * prop

n_big = 100
sn_big = n_big * prop

theta_mle = sn_big / n_big

theta_grid = np.arange(
    start=0,
    stop=1+inc,
    step=inc,
)
# --- Prior distribution ---
a = 2
b = 5

beta_pdf_prior = beta.pdf(
    x=theta_grid,
    a=a,
    b=b,
)

# --- Posterior ---
a_post_small = a + sn_small
b_post_small = b + n_small - sn_small
beta_pdf_post_small = beta.pdf(
    x=theta_grid,
    a=a_post_small,
    b=b_post_small,
)

a_post_big = a + sn_big
b_post_big = b + n_big - sn_big
beta_pdf_post_big = beta.pdf(
    x=theta_grid,
    a=a_post_big,
    b=b_post_big,
)


# =====================================================
# ===== Chart - Observed, Fitted and Risk-Neutral =====
# =====================================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(theta_grid, beta_pdf_prior, label=f"Prior Beta({a},{b})", lw=2, color=color_palette['blue'])
ax.plot(theta_grid, beta_pdf_post_small, label=f"Posterior Beta({a_post_small},{b_post_small}) n={n_small}", lw=2, color=color_palette['red'])
ax.plot(theta_grid, beta_pdf_post_big, label=f"Posterior Beta({a_post_big},{b_post_big}) n={n_big}", lw=2, color=color_palette['green'])
ax.axvline(theta_mle, label=f"MLE", lw=2, color=color_palette['yellow'], ls='--')
ax.set_xlim(0, 1)
ax.set_ylim(0, None)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"Probability Density")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")

plt.tight_layout()
plt.savefig(STATISTICS_BAYESIAN.joinpath("Bayes Intro - Beta-Bernoulli prior posterior.pdf"))
plt.show()
plt.close()
