import numpy as np
import pandas as pd
from utils.performance import Performance


n_portfolios = 5

port_labels = [f"Port {i+1}" for i in range(n_portfolios)]


trackers = pd.read_excel(
    '/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
    sheet_name='CDS Trackers',
    index_col=0,
)
trackers = trackers.resample('M').last()
is_available = ~trackers.isna()
rets = np.log(trackers).diff(1)

spreads = pd.read_excel(
    '/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
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

portfolios_trackers = (1 + portfolios).cumprod()
portfolios_trackers = 100 * portfolios_trackers / portfolios_trackers.iloc[0]

# TODO Add HML portfolio

perf = Performance(portfolios_trackers, skip_dd=True)
perf.table.to_clipboard()

# TODO plot the portfolio trackers










a = 1