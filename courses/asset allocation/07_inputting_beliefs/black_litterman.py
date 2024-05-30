from allocation import BlackLitterman, MeanVar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from getpass import getuser

file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation")
df = pd.read_excel(file_path.joinpath('Commodities Total Return.xlsx'), index_col=0)

asset_list = ['Asset A', 'Asset B', 'Asset C']
view_list = ['View 1', 'View 2', 'Views 3']
risk_free = 0.0075

# Covariance of returns
corr = np.array([[1, 0.5, 0.0],
                 [0.5, 1, 0.0],
                 [0.0, 0.0, 1]])

vol = np.array([0.1, 0.12, 0.15])
sigma = np.diag(vol) @ corr @ np.diag(vol)

corr = pd.DataFrame(data=corr, columns=asset_list, index=asset_list)
vol = pd.Series(data=vol, index=asset_list, name='Vol')
sigma = pd.DataFrame(data=sigma, columns=asset_list, index=asset_list)

# Views
tau = 1/500

views_p = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
views_p = pd.DataFrame(data=views_p, columns=asset_list, index=view_list)

views_v = np.array([0.01, 0.025, 0.02])
views_v = pd.DataFrame(data=views_v, index=view_list, columns=['View Values'])

u = np.array([1, 1, 0.3])
u = pd.DataFrame(data=u, index=view_list, columns=['Relative Uncertainty'])

# best guess for mu
w_equilibrium = np.array([1/3, 1/3, 1/3])
w_equilibrium = pd.DataFrame(data=w_equilibrium, index=asset_list, columns=['Equilibrium Weights'])

mu_historical = np.array([0.013, 0.017, 0.016])
mu_historical = pd.DataFrame(data=mu_historical, index=asset_list, columns=['Historical Returns'])

bl = BlackLitterman(sigma=sigma,
                    estimation_error=tau,
                    w_equilibrium=w_equilibrium,  # TODO make series
                    avg_risk_aversion=1.2,
                    mu_shrink=1,
                    views_p=views_p,
                    views_v=views_v,
                    overall_confidence=100,
                    relative_uncertainty=u,
                    mu_historical=mu_historical)

mkw_original = MeanVar(mu=bl.mu_best_guess,
                       cov=sigma,
                       rf=risk_free,
                       risk_aversion=1.2)
original_frontier_mu, original_frontier_sigma = mkw_original.min_var_frontier(n_steps=300)

mkw_bl = MeanVar(mu=bl.mu_bl,
                 cov=bl.sigma_bl,
                 rf=risk_free,
                 risk_aversion=1.2)
bl_frontier_mu, bl_frontier_sigma = mkw_bl.min_var_frontier(n_steps=300)

# ===== Chart =====
fig = plt.figure(figsize=(5 * (16 / 7.3), 5))
ax = plt.subplot2grid((1, 1), (0, 0))

# Assets
ax.scatter(vol, bl.mu_best_guess, label='Original Assets', color='red', marker='o',
           edgecolor='black', s=65)
ax.scatter(np.diag(bl.sigma_bl)**0.5, bl.mu_bl, label='Black-Litterman Outputs', color='green', marker='p',
           edgecolor='black', s=65)

# risk-free
ax.scatter(0, risk_free, label='Risk-Free', edgecolor='black', s=65)

# Optimal risky portfolio
plt.scatter(mkw_original.sigma_p, mkw_original.mu_p, label='Original Optimal', color='firebrick',
            marker='X', s=65, zorder=-1)
plt.scatter(mkw_bl.sigma_p, mkw_bl.mu_p, label='Black-Litterman Optimal', color='darkgreen',
            marker='X',  s=65, zorder=-1)

# Minimal variance frontier
plt.plot(original_frontier_sigma, original_frontier_mu, marker=None, color='red',
         label='Original Min Variance Frontier')
plt.plot(bl_frontier_sigma, bl_frontier_mu, marker=None, color='green',
         label='Black-Litterman Min Variance Frontier')

# Capital allocation line
max_sigma = vol.max() + 0.05
x_values = [0, max_sigma]
y_values = [risk_free, risk_free + mkw_original.sharpe_p * max_sigma]
plt.plot(x_values, y_values, marker=None, color='red', label='Original Capital Allocation Line', linestyle='--')
y_values = [risk_free, risk_free + mkw_bl.sharpe_p * max_sigma]
plt.plot(x_values, y_values, marker=None, color='green', label='Black-Litterman Capital Allocation Line', linestyle='--')

# legend
ax.legend(loc='upper left', frameon=True)

# adjustments
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlim((0, vol.max() + 0.03))
ax.set_ylim((0.005, bl.mu_bl.max() + 0.01))
ax.set_xlabel('Risk')
ax.set_ylabel('Return')

# Save as picture
plt.savefig(file_path.joinpath('Figures/Beliefs - Black Litterman Example.pdf'), pad_inches=0)

plt.tight_layout()
plt.show()
plt.close()
