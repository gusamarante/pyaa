import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import chi2, f
from getpass import getuser

# User parameters
file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Doutorado - Empirical Finance/Project 1")
show_charts = False


# ================
# ===== Data =====
# ================
# --- read portfolios ---
ff25 = pd.read_excel(file_path.joinpath("Dados.xlsx"),
                     skiprows=2, header=[0, 1], index_col=0, sheet_name="FF25")
ff25.index = pd.to_datetime(ff25.index)

# --- read factors ---
ff5f = pd.read_excel(file_path.joinpath("Dados.xlsx"),
                     index_col=0, sheet_name="Factors")
ff5f.index = pd.to_datetime(ff5f.index)

# --- Execess Returns of the FF25 ---
ff25 = ff25.sub(ff5f['RF'], axis=0)

# --- summary statistics ---
means = ff25.mean()
stds = ff25.std()


# =======================================================
# ===== First Stage - The 25 Timeseries Regressions =====
# =======================================================
betas = pd.Series(index=means.index)
df_resids = pd.DataFrame()

for s in range(1, 6):
    for v in range(1, 6):
        reg_data = pd.concat([ff25[s][v].rename('Y'), ff5f["Mkt"].rename('X')], axis=1)
        reg_data = reg_data.dropna()

        Y = reg_data["Y"]
        X = sm.add_constant(reg_data["X"])
        model = sm.OLS(Y, X)
        res = model.fit()

        betas.loc[s, v] = res.params["X"]
        df_resids[f"S{s}V{v}"] = res.resid


# =======================================================
# ===== Second Stage - The Cross-Section Regression =====
# =======================================================
for add_cons in [True, False]:
    for estimator in ['OLS', 'GLS']:
        for test_assets in [True, False]:
            pass

# TODO CS regression
# TODO Test