import pandas as pd
import numpy as np
from scipy.stats import norm, invgamma
import matplotlib.pyplot as plt
from tqdm import tqdm

n = 100  # Size of series
n_gibbs = 1000  # Number of Gibbs samples
burnin = 100
n_tot = n_gibbs + burnin

alpha = 0  # AR(1) Intercept
beta = 1  # AR(1) Coefficient
tau = np.sqrt(1)  # AR(1) Noise
sigma_y = np.sqrt(0.25)  # Observation Error

tau2 = tau ** 2
sig2 = sigma_y ** 2


def simul_ar1(phi0, phi1, sigma_e, sigma_omg):
    omg = norm.rvs(loc=0, scale=sigma_omg, size=n)
    theta = [0]
    for t in range(n-1):
        theta.append(phi0 + phi1 * theta[t] + omg[t])
    theta = np.array(theta)
    eps = norm.rvs(loc=0, scale=sigma_e, size=n)
    y = theta + eps
    return y


obs = simul_ar1(
    phi0=alpha,
    phi1=beta,
    sigma_e=sigma_y,
    sigma_omg=tau,
)


# ===========================
# ===== Gibbs samplings =====
# ===========================
# gibbs = pd.DataFrame(
#     columns=[k + 1 for k in range(n)],
# )
# gibbs.loc[0] = obs
#
# a1 = 0
# R1 = 9
# R2 = R1 ** 2
#
# for ng in tqdm(range(1, n_tot + 1)):
#     for nt in range(1, n + 1):
#
#         if nt == 1:  # Start of the series
#             T1 = sig2 * R2 * alpha * beta - tau2 * R2 * obs[0] - sig2 * R2 * beta * gibbs.loc[ng-1, 2] - sig2 * tau2 * a1
#             T2 = tau2 * R2 + sig2 * R2 * (beta**2) + sig2 * tau2
#             Tmu = - T1 / T2
#             Tsig = np.sqrt(sig2 * tau2 * R2 / T2)
#             gibbs.loc[ng, nt] = norm.rvs(loc=Tmu, scale=Tsig)
#
#         elif nt == n:  # End of the series
#             Q1 = - tau2 * obs[-1] - sig2 * alpha - sig2 * beta * gibbs.loc[ng, nt - 1]
#             Q2 = sig2 + tau2
#             Qmu = - Q1 / Q2
#             Qsig = np.sqrt(sig2 * tau2 / Q2)
#             gibbs.loc[ng, nt] = norm.rvs(loc=Qmu, scale=Qsig)
#
#         else:  # Somewhere in the middle
#             M1 = sig2 * alpha * beta - tau2 * obs[nt - 1] - sig2 * beta * gibbs.loc[ng - 1, nt + 1] - sig2 * alpha - sig2 * beta * gibbs.loc[ng, nt - 1]
#             M2 = tau2 + sig2 * (beta ** 2) + sig2
#             Mmu = - M1 / M2
#             Msig = np.sqrt(sig2 * tau2 / M2)
#             gibbs.loc[ng, nt] = norm.rvs(loc=Mmu, scale=Msig)
#
# cis = gibbs.quantile(q=[0.025, 0.5, 0.975])
# cis = cis.iloc[-n_gibbs:]


# =================
# ===== CHART =====
# =================
# size = 5
# fig = plt.figure(figsize=(size * (16 / 9), size))
# ax = plt.subplot2grid((1, 1), (0, 0))
# ax.plot(cis.columns, obs, label=rf"Observed $y_t$", lw=2)
# ax.plot(cis.loc[0.5], label=rf"Median", lw=2)
# ax.fill_between(cis.columns, cis.loc[0.025], cis.loc[0.975], label="95% CI", lw=0, color='grey', alpha=0.3)
# ax.set_xlabel(r"$t$")
# ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.legend(frameon=True, loc="best")
#
# plt.tight_layout()
# plt.savefig("/Users/gamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - Bayesiana/HW04/ar1 gibbs.pdf")
# plt.show()
# plt.close()


# ======================================
# ===== Gibbs samplings with Gamma =====
# ======================================
mu_alpha = 0
mu_beta = 0
delta = 2

a_sigma = 3  # TODO comment mena and stdev of 1
b_sigma = 2

a_tau = 3
b_tau = 2


cols = [k + 1 for k in range(n)]
cols.append('alpha')
cols.append('beta')
cols.append('sigma2')
cols.append('tau2')

gibbs = pd.DataFrame(
    columns=cols,
)

gibbs.loc[0, 1:n] = obs
gibbs.loc[0, 'alpha'] = 0
gibbs.loc[0, 'beta'] = 0
gibbs.loc[0, 'sigma2'] = invgamma.mean(a=a_sigma, scale=b_sigma)
gibbs.loc[0, 'tau2'] = invgamma.mean(a=a_tau, scale=b_tau)

# TODO montar o gibbs sampling com todas as variáveis


print(gibbs)
