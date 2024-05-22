import numpy as np
import pandas as pd

from allocation import RiskBudgetVol


# Declarations
asset_names = [f'Asset {i+1}' for i in range(4)]

vols = np.array([0.15, 0.20, 0.30, 0.10])
corr = np.array([[1, 0.5, 0.0, -0.1],
                 [0.5, 1, 0.2, 0.4],
                 [0.0, 0.2, 1, 0.7],
                 [-0.1, 0.4, 0.7, 1]])
cov = np.diag(vols) @ corr @ np.diag(vols)
cov = pd.DataFrame(data=cov, index=asset_names, columns=asset_names)

budget = pd.Series(data=[0.2, 0.2, 0.4, 0.2],
                   index=asset_names)

rbv = RiskBudgetVol(cov=cov, budget=None)

print(rbv.risk_contribution_ratio)
