from scipy.stats import t, norm, laplace, uniform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

m = 5000  # Samples from candidate
n = 1000  # Weighted samples from set

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
size = 4
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
ax.plot(theta_grid, t_pdf, label=f"t(3)", lw=2)
ax.plot(theta_grid, norm_pdf, label=f"N(0,1)", lw=2)
ax.set_ylim(0, None)
ax.set_ylabel(r"Probability Density")
ax.set_xlabel(r"$\theta$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")

ax = plt.subplot2grid((3, 2), (2, 0))
ax.plot(theta_grid, norm_pdf / t_pdf, label=f"Weight / Density Ratio", lw=2)
ax.set_ylabel(r"Density Ratio")
ax.set_xlabel(r"$\theta$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
ax.plot(theta_grid, t_pdf, label=f"t(3)", lw=2)
ax.plot(theta_grid, laplace_pdf, label=f"Laplace(0,1)", lw=2)
ax.set_ylim(0, None)
ax.set_ylabel(r"Probability Density")
ax.set_xlabel(r"$\theta$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")

ax = plt.subplot2grid((3, 2), (2, 1))
ax.plot(theta_grid, laplace_pdf / t_pdf, label=f"Weight / Density Ratio", lw=2)
ax.set_ylabel(r"Density Ratio")
ax.set_xlabel(r"$\theta$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("/Users/gamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - Bayesiana/HW02/part 2 candidate target.pdf")
plt.show()
plt.close()


# =============================
# ===== Item (i) Sampling =====
# =============================

# Get samples from t distribution
sample = t.rvs(df=3, size=m)

# compute weights for target \pi_a
w_a = norm.pdf(sample, loc=0, scale=1) / t.pdf(sample, df=3)
w_a = w_a / w_a.sum()

# compute weights for target \pi_b
w_b = laplace.pdf(sample, loc=0, scale=1) / t.pdf(sample, df=3)
w_b = w_b / w_b.sum()

# draw for weighted set
draws_a = np.random.choice(sample, p=w_a, size=n, replace=True)
draws_b = np.random.choice(sample, p=w_b, size=n, replace=True)

print("t dist")
print("mean A", draws_a.mean())
print("var A", draws_a.var())
print("mean B", draws_b.mean())
print("var B", draws_b.var())
print("Tail A", (draws_a >= 2).mean())
print("Tail B", (draws_b >= 2).mean())
print("\n")


# ==============================
# ===== Item (iii) Uniform =====
# ==============================
# Get samples from uniform distribution
sample_u = uniform.rvs(loc=-10, scale=20, size=m)

# compute weights for target \pi_a
w_a_u = norm.pdf(sample_u, loc=0, scale=1) / uniform.pdf(sample_u, loc=-10, scale=20)
w_a_u = w_a_u / w_a_u.sum()

# compute weights for target \pi_b
w_b_u = laplace.pdf(sample_u, loc=0, scale=1) / uniform.pdf(sample_u, loc=-10, scale=20)
w_b_u = w_b_u / w_b_u.sum()

# draw for weighted set
draws_a_u = np.random.choice(sample_u, p=w_a_u, size=n, replace=True)
draws_b_u = np.random.choice(sample_u, p=w_b_u, size=n, replace=True)

print("Uniform")
print("mean A", draws_a_u.mean())
print("var A", draws_a_u.var())
print("mean B", draws_b_u.mean())
print("var B", draws_b_u.var())

# Compare KDEs
df_a = pd.DataFrame({"t": draws_a, "uniform": draws_a_u})
df_b = pd.DataFrame({"t": draws_b, "uniform": draws_b_u})

size = 4
n_bins = 100
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.plot(theta_grid, norm_pdf, label="original")
ax.hist(draws_a, label="t", density=True, alpha=0.4, bins=n_bins)
ax.hist(draws_a_u, label="uniform", density=True, alpha=0.4, bins=n_bins)
ax.set_title("Target: Normal")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

ax = plt.subplot2grid((1, 2), (0, 1))
ax.plot(theta_grid, laplace_pdf, label="original")
ax.hist(draws_b, label="t", density=True, alpha=0.4, bins=n_bins)
ax.hist(draws_b_u, label="uniform", density=True, alpha=0.4, bins=n_bins)
ax.set_title("Target: Laplace")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()
plt.savefig("/Users/gamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - Bayesiana/HW02/part 2 item iii.pdf")
plt.show()
