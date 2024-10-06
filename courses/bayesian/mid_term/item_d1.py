"""
This routine only generates the samples from the posterior using the gibbs sampler.

Analysis of this sample is in another routine
"""

import pandas as pd
from tqdm import tqdm
from scipy.stats import norm, invgamma, multivariate_normal
import numpy as np
from numpy.linalg import inv
import statsmodels.api as sm
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


n_burn = 1000
n_gibs = 10000
n_total = n_gibs + n_burn

I = 68
J = 7


a, b = 0.001, 0.001  # Inv Gamma Hyperparameters

# ===== Read Data =====
df = pd.read_excel('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/Dados.xlsx',
                   sheet_name="com_na")
df_dna = df.dropna()


# ===== Set initial values =====
IJ = I * J
IJ_na = df_dna.shape[0]

gibbs = pd.DataFrame()

gibbs.loc[0, "sig alpha"] = 1
gibbs.loc[0, "sig beta"] = 1
gibbs.loc[0, "sig y"] = 1

gibbs.loc[0, "alpha"] = 0
gibbs.loc[0, "beta"] = 0

for i in range(I):
    gibbs.loc[0, f"alpha {i+1}"] = 0

for i in range(I):
    gibbs.loc[0, f"beta {i + 1}"] = 0


# ===== Gibbs Sampling =====
for n in tqdm(range(1, n_total)):

    # Sample sig alpha
    ra = a + I / 2
    sum_sq_a = ((gibbs.loc[n - 1, "alpha 1": "alpha 68"] - gibbs.loc[n - 1, "alpha"])**2).sum()
    rb = b + sum_sq_a / 2
    sig_a2 = invgamma.rvs(a=ra, scale=rb)
    gibbs.loc[n, "sig alpha"] = np.sqrt(sig_a2)

    # Sample sig beta
    ra = a + I / 2
    sum_sq_b = ((gibbs.loc[n - 1, "beta 1": "beta 68"] - gibbs.loc[n - 1, "beta"])**2).sum()
    rb = b + sum_sq_b / 2
    sig_b2 = invgamma.rvs(a=ra, scale=rb)
    gibbs.loc[n, "sig beta"] = np.sqrt(sig_b2)

    # Sample sig y
    sqr = 0
    for i in range(1, I + 1):
        y = df_dna[df_dna['patient'] == i]['weight_gain']
        x = df_dna[df_dna['patient'] == i]['gestational_age']
        sqr =+ ((y - gibbs.loc[n - 1, f"alpha {i}"] - gibbs.loc[n - 1, f"beta {i}"]*x)**2).sum()
    ra = a + IJ_na / 2
    rb = b + sqr / 2
    sig_y2 = invgamma.rvs(a=ra, scale=rb)
    gibbs.loc[n, "sig y"] = np.sqrt(sig_y2)

    # Sample alpha
    p_a = 1 / 1000 + I / sig_a2
    mu_a = ((1 / sig_a2) * gibbs.loc[n - 1, "alpha 1": "alpha 68"].sum()) / p_a
    gibbs.loc[n, "alpha"] = norm.rvs(loc=mu_a, scale=np.sqrt(1 / p_a))

    # Sample beta
    p_b = 1 / 1000 + I / sig_b2
    mu_b = ((1 / sig_b2) * gibbs.loc[n - 1, "beta 1": "beta 68"].sum()) / p_b
    gibbs.loc[n, "beta"] = norm.rvs(loc=mu_b, scale=np.sqrt(1 / p_b))

    # Sample alpha_i and beta_i
    S = np.diag([sig_a2, sig_b2])
    invS = inv(S)
    for i in range(1, I + 1):
        y = df_dna[df_dna['patient'] == i]['weight_gain'].values
        x = sm.add_constant(df_dna[df_dna['patient'] == i]['gestational_age']).values
        xx = x.T @ x
        xy = x.T @ y
        Lamb = inv(invS + (1 / sig_y2) * xx)
        theta = np.array([gibbs.loc[n, "alpha"], gibbs.loc[n, "beta"]])
        mu_i = Lamb @ ((1 / sig_y2) * xy + invS @ theta)
        ri = multivariate_normal.rvs(mean=mu_i, cov=Lamb)
        gibbs.loc[n, f"alpha {i}"] = ri[0]
        gibbs.loc[n, f"beta {i}"] = ri[1]

gibbs.to_csv('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/samples.csv')

