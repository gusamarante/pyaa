import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy.stats import norm, invgamma, multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

n = 100  # Size of series
n_gibbs = 10000  # Number of Gibbs samples
burnin = 100
n_tot = n_gibbs + burnin

# Real parameters for the DGP
real_alpha = 0  # AR(1) Intercept
real_beta = 1  # AR(1) Coefficient
tau = np.sqrt(1)  # AR(1) Noise
sigma_y = np.sqrt(0.25)  # Observation Error

tau2 = tau**2
sig2 = sigma_y**2


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
    phi0=real_alpha,
    phi1=real_beta,
    sigma_e=sigma_y,
    sigma_omg=tau,
)


# ====================================
# ===== Gibbs Samplings on Theta =====
# ====================================
gibbs_theta = pd.DataFrame(
    columns=[k + 1 for k in range(n)],
)
gibbs_theta.loc[0] = obs

a1 = 0
R1 = 9
R2 = R1 ** 2

for ng in tqdm(range(1, n_tot + 1), "Gibbs on Theta"):
    for nt in range(1, n + 1):

        if nt == 1:  # Start of the series
            T1 = sig2 * R2 * real_alpha * real_beta - tau2 * R2 * obs[0] - sig2 * R2 * real_beta * gibbs_theta.loc[ng-1, 2] - sig2 * tau2 * a1
            T2 = tau2 * R2 + sig2 * R2 * (real_beta**2) + sig2 * tau2
            Tmu = - T1 / T2
            Tsig = np.sqrt(sig2 * tau2 * R2 / T2)
            gibbs_theta.loc[ng, nt] = norm.rvs(loc=Tmu, scale=Tsig)

        elif nt == n:  # End of the series
            Q1 = - tau2 * obs[-1] - sig2 * real_alpha - sig2 * real_beta * gibbs_theta.loc[ng, nt - 1]
            Q2 = sig2 + tau2
            Qmu = - Q1 / Q2
            Qsig = np.sqrt(sig2 * tau2 / Q2)
            gibbs_theta.loc[ng, nt] = norm.rvs(loc=Qmu, scale=Qsig)

        else:  # Somewhere in the middle
            M1 = sig2 * real_alpha * real_beta - tau2 * obs[nt - 1] - sig2 * real_beta * gibbs_theta.loc[ng - 1, nt + 1] - sig2 * real_alpha - sig2 * real_beta * gibbs_theta.loc[ng, nt - 1]
            M2 = tau2 + sig2 * (real_beta ** 2) + sig2
            Mmu = - M1 / M2
            Msig = np.sqrt(sig2 * tau2 / M2)
            gibbs_theta.loc[ng, nt] = norm.rvs(loc=Mmu, scale=Msig)

gibbs_theta = gibbs_theta.iloc[-n_gibbs:]
cis_theta = gibbs_theta.quantile(q=[0.025, 0.5, 0.975])


# =======================================
# ===== CHART - Gibbs only on theta =====
# =======================================
size = 5
fig = plt.figure(figsize=(size * (16 / 9), size))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(cis_theta.columns, obs, label=rf"Observed $y_t$", lw=2)
ax.plot(cis_theta.loc[0.5], label=rf"Median", lw=2)
ax.fill_between(cis_theta.columns, cis_theta.loc[0.025], cis_theta.loc[0.975], label="95% CI", lw=0, color='grey', alpha=0.3)
ax.set_xlabel(r"$t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")

plt.tight_layout()
plt.savefig("/Users/gamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - Bayesiana/HW04/ar1 gibbs.pdf")
plt.show()
plt.close()


# ================================
# ===== Full Gibbs Samplings =====
# ================================
a_sigma = 1
b_sigma = 1

a_tau = 1
b_tau = 1


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
gibbs.loc[0, 'sigma2'] = b_sigma / (a_sigma + 1)
gibbs.loc[0, 'tau2'] = b_tau / (a_tau + 1)

a1 = 0
R1 = 9
R2 = R1 ** 2
delta = 9

for ng in tqdm(range(1, n_tot + 1), "Full Gibbs"):

    # --- Sample thetas ---
    for nt in range(1, n + 1):

        if nt == 1:  # Start of the series
            T1 = gibbs.loc[ng-1, 'sigma2'] * R2 * gibbs.loc[ng-1, 'alpha'] * gibbs.loc[ng-1, 'beta'] - gibbs.loc[ng-1, 'tau2'] * R2 * obs[0] - gibbs.loc[ng-1, 'sigma2'] * R2 * gibbs.loc[ng-1, 'beta'] * gibbs.loc[ng-1, 2] - gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'tau2'] * a1
            T2 = gibbs.loc[ng-1, 'tau2'] * R2 + gibbs.loc[ng-1, 'sigma2'] * R2 * (gibbs.loc[ng-1, 'beta']**2) + gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'tau2']
            Tmu = - T1 / T2
            Tsig = np.sqrt(gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'tau2'] * R2 / T2)
            gibbs.loc[ng, nt] = norm.rvs(loc=Tmu, scale=Tsig)

        elif nt == n:  # End of the series
            Q1 = - gibbs.loc[ng-1, 'tau2'] * obs[-1] - gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'alpha'] - gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'beta'] * gibbs.loc[ng, nt - 1]
            Q2 = gibbs.loc[ng-1, 'sigma2'] + gibbs.loc[ng-1, 'tau2']
            Qmu = - Q1 / Q2
            Qsig = np.sqrt(gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'tau2'] / Q2)
            gibbs.loc[ng, nt] = norm.rvs(loc=Qmu, scale=Qsig)

        else:  # Somewhere in the middle
            M1 = gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'alpha'] * gibbs.loc[ng-1, 'beta'] - gibbs.loc[ng-1, 'tau2'] * obs[nt - 1] - gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'beta'] * gibbs.loc[ng - 1, nt + 1] - gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'alpha'] - gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'beta'] * gibbs.loc[ng, nt - 1]
            M2 = gibbs.loc[ng-1, 'tau2'] + gibbs.loc[ng-1, 'sigma2'] * (gibbs.loc[ng-1, 'beta'] ** 2) + gibbs.loc[ng-1, 'sigma2']
            Mmu = - M1 / M2
            Msig = np.sqrt(gibbs.loc[ng-1, 'sigma2'] * gibbs.loc[ng-1, 'tau2'] / M2)
            gibbs.loc[ng, nt] = norm.rvs(loc=Mmu, scale=Msig)

    # --- Sample alpha, beta ---
    y_theta = gibbs.loc[ng, 2:n].values.astype(float)
    X_theta = (np.vstack([np.ones(n-1), gibbs.loc[ng, 1:n-1].values]).T).astype(float)
    lambda_ab = inv((delta / gibbs.loc[ng - 1, 'tau2']) * X_theta.T @ X_theta + np.eye(2))
    mu_ab = (delta / gibbs.loc[ng - 1, 'tau2']) * lambda_ab @ X_theta.T @ y_theta
    rab = multivariate_normal.rvs(mean=mu_ab, cov=delta * lambda_ab)
    gibbs.loc[ng, 'alpha'] = rab[0]
    gibbs.loc[ng, 'beta'] = rab[1]

    # --- Sample tau2 ---
    S_tau = ((y_theta - X_theta @ rab)**2).sum()
    ra_tau = a_tau + (n - 1) / 2
    rb_tau = b_tau + S_tau / 2
    rtau2 = invgamma.rvs(a=ra_tau, scale=rb_tau)
    gibbs.loc[ng, 'tau2'] = rtau2

    # --- Sample sigma2 ---
    S_sigma = ((obs - gibbs.loc[ng, 1:n].values)**2).sum()
    ra_sigma = a_sigma + n / 2
    rb_sigma = b_sigma + S_sigma / 2
    rsigma2 = invgamma.rvs(a=ra_sigma, scale=rb_sigma)
    gibbs.loc[ng, 'sigma2'] = rsigma2

gibbs = gibbs.astype(float)
gibbs = gibbs.iloc[-n_gibbs:]
cis = gibbs.quantile(q=[0.025, 0.5, 0.975])


# =================
# ===== CHART =====
# =================
size = 5
fig = plt.figure(figsize=(size * (16 / 9), size))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(cis.columns[:-4], obs, label=rf"Observed $y_t$", lw=2)
ax.plot(cis.iloc[:, :-4].loc[0.5], label=rf"Median", lw=2)
ax.fill_between(cis.columns[:-4].values.astype(int), cis.iloc[:, :-4].loc[0.025], cis.iloc[:, :-4].loc[0.975], label="95% CI", lw=0, color='grey', alpha=0.3)
ax.set_xlabel(r"$t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")

plt.tight_layout()
plt.savefig("/Users/gamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - Bayesiana/HW04/ar1 gibbs 2.pdf")
plt.show()
plt.close()


# ====================================
# ===== CHART - Compare CI width =====
# ====================================
size = 5
fig = plt.figure(figsize=(size * (16 / 9), size))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(cis_theta.loc[0.975] - cis_theta.loc[0.025], label='Only Theta')
ax.plot(cis.iloc[:, :-4].loc[0.975] - cis.iloc[:, :-4].loc[0.025], label='Full')
ax.set_xlabel(r"$t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")

plt.tight_layout()
plt.savefig("/Users/gamarante/Library/CloudStorage/Dropbox/Aulas/Doutorado - Bayesiana/HW04/compare cis.pdf")
plt.show()
plt.close()

