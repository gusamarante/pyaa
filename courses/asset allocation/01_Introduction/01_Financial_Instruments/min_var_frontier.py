from allocation import MeanVar
from getpass import getuser
from pathlib import Path
import pandas as pd
import numpy as np

mu = pd.Series(
    data={
        "A": 0.12,
        "B": 0.2,
        "C": 0.15,
    }
)

vols = pd.Series(
    data={
        "A": 0.23,
        "B": 0.3,
        "C": 0.25,
    }
)

corr = pd.DataFrame(
    columns=["A", "B", "C"],
    index=["A", "B", "C"],
    data=[[1, 0.4, 0.4],
          [0.4, 1, 0],
          [0.4, 0, 1]],
)


cov = np.diag(vols) @ corr @ np.diag(vols)
cov = pd.DataFrame(
    columns=["A", "B", "C"],
    index=["A", "B", "C"],
    data=cov.values,
)

mv = MeanVar(
    mu=mu,
    cov=cov,
    rf=0.03,
    short_sell=False,
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
    save_path=fr"C:\Users\{getuser()}\Dropbox\Aulas\Insper - Asset Allocation\Figures\Fiancial Instruments - MeanVar Frontier.pdf"
)
