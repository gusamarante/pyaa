import matplotlib.pyplot as plt
import numpy as np

# Declarations
mu = np.array([0.12, 0.20, 0.15])
vols = np.array([0.23, 0.30, 0.25])
corr = np.array([[1, 0.6, 0.4],
                 [0.6, 1, 0],
                 [0.4, 0, 1]])
cov = np.diag(vols) @ corr @ np.diag(vols)

# Calculations
invcov = np.linalg.inv(cov)
a = mu.T @ invcov @ mu
b = np.ones(len(mu)).T @ invcov @ mu
c = np.ones(len(mu)).T @ invcov @ np.ones(len(mu))

mu_bar = np.linspace(-0.05, max(mu) + 0.05, 100)
sigma_bar = np.sqrt((c * (mu_bar**2) - 2 * b * mu_bar + a) / (c * a - b**2))

# =================
# ===== Chart =====
# =================
plt.plot(sigma_bar, mu_bar)
plt.scatter(vols, mu, color='tab:orange')
plt.axhline(0, color='black', lw=0.5)
plt.xlim(0, None)
plt.tight_layout()
plt.show()
