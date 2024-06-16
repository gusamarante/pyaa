"""
Random Multivariate t-dist
https://github.com/statsmodels/statsmodels/blob/main/statsmodels/sandbox/distributions/multivariate.py#L90

--- The Pods ---
Each pod has the same sharpe and Vol, and the same correlation with each other
Each of them charges a Pfee over benchmark

--- The Fund ---
Compiles the pods and leverages them to the desired vol. Charges a Pfee over
the performance of the fund over the benchmark. Charges admin fee over the
total size of the fund.

--- The client ---
What is the final performance for the client?
How much of the maximum available sharpe of the pods gets to the client?
How does the maximum sharpe (of the fund and of the client) change as a
function of the number of pods and their average correlation.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sharpe = 0.8
vol = 0.04
rf = 0.05
rho = 0.1
n_periods = 1
pfee_pods = 0.18
admin_fee = 0.016
pfee_admin = 0.12
simulations = 1000


def multivariate_t_rvs(m, S, df=np.inf, n_simul=1):
    """
    generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n_simul : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    """
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = np.ones(n_simul)
    else:
        x = np.random.chisquare(df, n_simul) / df
    z = np.random.multivariate_normal(np.zeros(d), S, n_simul)
    return m + z / np.sqrt(x)[:, None]


df_scenarios = []
for n_pods in [1, 3, 5, 10, 15, 20]:
    for leverage in [1, 3, 5, 7]:
        for sharpe in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5]:
            for rho in [0, 0.1, 0.25, 0.5, 0.75, 0.9]:

                # Given all parameter compute static moments
                mu = rf + vol * sharpe
                mu = mu * np.ones(n_pods)

                corr_mat = np.eye(n_pods) * (1 - rho) + np.ones((n_pods, n_pods)) * rho
                cov_mat = np.diag(vol * np.ones(n_pods)) @ corr_mat @ np.diag(vol * np.ones(n_pods))
                inv_c = np.linalg.inv(corr_mat)

                iota = np.ones(n_pods)
                max_sharpe = sharpe * np.sqrt(iota @ inv_c @ iota)
                w = (1 / n_pods) * np.ones(n_pods)
                quota_vol = np.sqrt(w.T @ cov_mat @ w) * leverage

                df_simuls = pd.DataFrame()
                case_name = f"N={n_pods}, lev={leverage}, S={sharpe}, rho={rho}"
                for ss in tqdm(range(simulations), case_name):  # TODO add messege

                    # Sample the returns
                    rets = multivariate_t_rvs(mu, cov_mat, 10, n_periods)
                    trackers = np.vstack([np.ones((1, n_pods)), 1 + rets])
                    trackers = trackers.cumprod(axis=0)
                    trackers = pd.DataFrame(data=trackers, columns=[f"Pod {p+1}" for p in range(n_pods)])

                    bonus_pods = np.maximum(trackers.diff(1) - trackers.shift(1) * rf, 0) * pfee_pods
                    quota_gross = trackers.sum(axis=1)

                    quota_after_podfee = (trackers.diff(1) - trackers.shift(1) * rf).sum(axis=1) * leverage
                    quota_after_podfee.iloc[0] = quota_gross.iloc[0]
                    quota_after_podfee = quota_after_podfee.cumsum()

                    admin_cost = quota_after_podfee.shift(1) * admin_fee
                    bonus_admin = np.maximum(quota_after_podfee.diff(1) - quota_after_podfee.shift(1) * rf, 0) * pfee_admin
                    quota_net = quota_after_podfee.diff(1) - admin_cost - bonus_admin
                    quota_net.iloc[0] = quota_gross.iloc[0]
                    quota_net = quota_net.cumsum()

                    # compute single stats
                    df_simuls.loc[ss, 'Pods Avg Sharpe'] = ((trackers.pct_change(1) - rf) / vol).mean(axis=1).iloc[-1]
                    df_simuls.loc[ss, 'Quota Sharpe'] = (quota_gross.pct_change(1) / quota_vol).iloc[-1]
                    df_simuls.loc[ss, 'Net Sharpe'] = (quota_net.pct_change(1) / quota_vol).iloc[-1]

                # compute simul stats
                aux_simul = pd.Series(
                    data={
                        'Average Sharpe of Pods': df_simuls['Pods Avg Sharpe'].mean(),
                        'Sharpe of Port of Pods': df_simuls['Quota Sharpe'].mean(),
                        'Net Sharpe': df_simuls['Net Sharpe'].mean(),
                        "Alpha Share Mean": (df_simuls['Net Sharpe'] / df_simuls['Quota Sharpe']).mean(),
                        "Alpha Share Median": (df_simuls['Net Sharpe'] / df_simuls['Quota Sharpe']).median(),
                        "Number of Pods": n_pods,
                        "Leverage Factor": leverage,
                        "Exante Sharpe of Pods": sharpe,
                        "Avg Correl of Pods": rho,
                    },
                )
                df_scenarios.append(aux_simul)

df_scenarios = pd.concat(df_scenarios, axis=1)
df_scenarios = df_scenarios.T

print(df_scenarios)
df_scenarios.to_clipboard()

df_scenarios.to_csv("/Users/gustavoamarante/Library/CloudStorage/Dropbox/multipod DATA.csv")
