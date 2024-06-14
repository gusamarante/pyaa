import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sharpe = 0.8
vol = 0.04
rf = 0.05
n_pods = 10
rho = 0.1
n_periods = 1
pfee_pods = 0.18
leverage = 5
admin_fee = 0.016
pfee_admin = 0.12
simulations = 1000

# Portfolio of pods
mu = rf + vol * sharpe
mu = mu * np.ones(n_pods)

corr_mat = np.eye(n_pods) * (1 - rho) + np.ones((n_pods, n_pods)) * rho
cov_mat = np.diag(vol * np.ones(n_pods)) @ corr_mat @ np.diag(vol * np.ones(n_pods))
inv_c = np.linalg.inv(corr_mat)

iota = np.ones(n_pods)
max_sharpe = sharpe * np.sqrt(iota @ inv_c @ iota)

w = (1/n_pods) * np.ones(n_pods)
vol_quota = np.sqrt(w.T @ cov_mat @ w) * leverage