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

sharpe = 0.5
vol = 0.04
rf = 0.05
n_pods = 10
rho = 0
n_periods = 10

mu = rf + vol * sharpe
mu = mu * np.ones(n_pods)

corr_mat = np.eye(n_pods) * (1 - rho) + np.ones((n_pods, n_pods)) * rho
cov_mat = np.diag(vol * np.ones(n_pods)) @ corr_mat @ np.diag(vol * np.ones(n_pods))


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


# Sample the returns
rets = multivariate_t_rvs(mu, cov_mat, 10, n_periods)
trackers = np.vstack([np.ones((1, n_pods)), 1 + rets])
trackers = trackers.cumprod(axis=0)
trackers = pd.DataFrame(data=trackers,
                        columns=[f"Pod {p+1}" for p in range(n_pods)])
trackers.plot()
plt.show()

trackers.to_clipboard()

# TODO Now I have the trajectories, compute performances and fees
