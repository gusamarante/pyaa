import matplotlib.pyplot as plt
from allocation import MeanVar
from getpass import getuser
from pathlib import Path
import pandas as pd
import numpy as np

file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation")
df = pd.read_excel(file_path.joinpath('Commodities Total Return.xlsx'), index_col=0)
df.index = pd.to_datetime(df.index)
df = df.resample('M').last()
df = df.pct_change(1, fill_method=None)
df = df.dropna()

cov = df.cov()
vols = df.std()
mu = df.mean() + 0.05/12


# Irrestricted
invcov = np.linalg.inv(cov)
iota = np.ones(len(mu))

a = mu.T @ invcov @ mu
b = iota.T @ invcov @ mu
c = iota.T @ invcov @ iota

mu_bar = np.linspace(min(mu) - 0.02, max(mu) + 0.02, 100)
sigma_bar = np.sqrt((c * (mu_bar ** 2) - 2 * b * mu_bar + a) / (c * a - b ** 2))

# ===================
# ===== Chart 1 =====
# ===================
plt.scatter(vols, mu, label='Assets')
plt.plot(sigma_bar, mu_bar, label='MVF Irrestrita')
plt.axhline(0, color='black', lw=0.5)
plt.xlim(0, None)
plt.legend()
plt.tight_layout()
plt.show()


# ===================
# ===== Chart 2 =====
# ===================
mv = MeanVar(
    mu=mu,
    cov=cov,
    rf=0,
    short_sell=True,
)
mv.plot(
    assets=True,  # plot individual assets
    gmvp=True,  # plot global min var
    max_sharpe=True,  # Max Sharpe port
    risk_free=True,  # plot rf
    mvf=True,  # MinVar Frontier
    mvfnoss=True,  # MinVar Frontier no short selling
    cal=True,  # Capital Allocation Line
    investor=False,  # Investor's indifference, portfolio, and CE
)
