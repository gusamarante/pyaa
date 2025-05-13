import statsmodels.api as sm
import pandas as pd
import getpass
import numpy as np


n_portfolios = 5
size = 5
port_labels = [f"Port {i+1}" for i in range(n_portfolios)]
username = getpass.getuser()

trackers = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
    sheet_name='CDS Trackers',
    index_col=0,
)
trackers = trackers.resample('M').last()
is_available = ~trackers.isna()
rets = np.log(trackers).diff(1)

spreads = pd.read_excel(
    f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
    sheet_name='CDS Spread',
    index_col=0,
)
spreads = spreads.resample('M').last()
spreads = spreads.reindex(trackers.index)


# Assign countries to portfolios
def assign_portfolio(x):
    x = x[is_available.loc[x.name]]
    x = pd.qcut(x, q=n_portfolios, labels=port_labels)
    return x

portfolios = spreads.apply(assign_portfolio, axis=1)
portfolios = portfolios.shift(1)  # Today's selection is tomorrow's portfolio

portfolios = pd.DataFrame({"returns": rets.stack(), "portfolio": portfolios.stack()})
portfolios = portfolios.groupby(['date', 'portfolio']).mean()
portfolios = portfolios.unstack("portfolio")["returns"]

portfolios[f"Port {n_portfolios - 1}-1"] = portfolios[f"Port {n_portfolios - 1}"] - portfolios[f"Port 1"]
portfolios[f"Port {n_portfolios}-{n_portfolios - 1}"] = portfolios[f"Port {n_portfolios}"] - portfolios[f"Port {n_portfolios - 1}"]

port_trackers = (1 + portfolios).cumprod()
port_trackers = 100 * port_trackers / port_trackers.iloc[0]


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

# Cross-Section Regression
betas = coeffs.iloc[:, 1:]
means = test_assets.mean()

model = sm.OLS(means, sm.add_constant(betas))
res = model.fit()
