from allocation import MeanVar
import pandas as pd
import numpy as np
from pathlib import Path
from getpass import getuser
from simulation.randomcov import random_correlation


# TODO preencher valores corretos
mu = pd.Series(
    data={
        "A": 0.12,
        "B": 0.2,
        "C": 0.15,
        "D": 0.10,
        "E": 0.18,
        "F": 0.08,
    }
)

vols = pd.Series(
    data={
        "A": 0.18,
        "B": 0.25,
        "C": 0.20,
        "D": 0.22,
        "E": 0.30,
        "F": 0.18,
    }
)

corr = pd.DataFrame(
    columns=vols.index,
    index=vols.index,
    data=random_correlation(len(vols), len(vols)),
)


cov = np.diag(vols) @ corr @ np.diag(vols)
cov = pd.DataFrame(
    columns=vols.index,
    index=vols.index,
    data=cov.values,
)

mv = MeanVar(
    mu=mu,
    cov=cov,
    rf=0.03,
    short_sell=False,
)

file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation/Figures")
mv.plot(
    assets=True,  # plot individual assets
    gmvp=False,  # plot global min var
    max_sharpe=False,  # Max Sharpe port
    risk_free=False,  # plot rf
    mvf=True,  # MinVar Frontier
    mvfnoss=False,  # MinVar Frontier no short selling
    cal=False,  # Capital Allocation Line
    investor=False,  # Investor's indifference, portfolio, and CE
    save_path=file_path.joinpath("Static Portfolio Choice - Many risky.pdf")
)
