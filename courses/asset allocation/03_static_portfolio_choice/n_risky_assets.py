from allocation import MeanVar
import pandas as pd
import numpy as np
from pathlib import Path
from getpass import getuser


mu = pd.Series(
    data={
        "A": 0.0606,
        "B": 0.1065,
        "C": 0.11,
        "D": 0.085,
    }
)

vols = pd.Series(
    data={
        "A": 0.0888,
        "B": 0.2255,
        "C": 0.1064,
        "D": 0.15,
    }
)

corr = pd.DataFrame(
    columns=vols.index,
    index=vols.index,
    data=[[1, -0.3, -0.2, 0],
          [-0.3, 1, 0.5, 0],
          [-0.2, 0.5, 1, 0],
          [0, 0, 0, 1]],
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
