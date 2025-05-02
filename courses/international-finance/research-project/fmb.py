import statsmodels.api as sm
import pandas as pd


# ========================
# ===== Fama-MacBeth =====
# ========================
factors = portfolios.iloc[:, n_portfolios:]
test_assets = portfolios.iloc[:, :n_portfolios]

# ff5f = get_ff5f()
# factors = pd.concat([factors, ff5f["Mkt-RF"]], axis=1)


# Time series regression
coeffs = []
stders = []
r2 = pd.Series(name="R2")
df_resids = pd.DataFrame()

for col in test_assets.columns:
    reg_data = pd.concat([test_assets[col], factors], axis=1)
    reg_data = reg_data.dropna()

    model = sm.OLS(reg_data[col], sm.add_constant(reg_data[factors.columns]))
    res = model.fit()

    coeffs.append(res.params.rename(col))
    stders.append(res.bse.rename(col))
    r2.loc[col] = res.rsquared

coeffs = pd.concat(coeffs, axis=1).T
stders = pd.concat(stders, axis=1).T

# Cross Section Regression
betas = coeffs.iloc[:, 1:]
means = test_assets.mean()

model = sm.OLS(means, sm.add_constant(betas))
res = model.fit()