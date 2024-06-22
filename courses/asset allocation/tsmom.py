import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from utils import Performance

df = pd.read_excel(r"/Users/gamarante/Library/CloudStorage/Dropbox/Aulas/Insper - Asset Allocation/Commodities Total Return.xlsx",
                   index_col=0)

df = df[df.index >= "2011-01-01"]

rets = df.pct_change()
vols = rets.ewm(com=60, min_periods=60).std() * np.sqrt(21)
vols = vols.dropna(how='all')
vols = vols.resample('M').last()
vols = vols.shift(1)

retm = df.resample('M').last().pct_change(1)

srets = retm / vols

tstats = pd.DataFrame()
for comm in srets.columns:
    for h in range(1, 13):
        sretsl = srets.shift(h)  # H

        aux = pd.DataFrame(
            {'Y': srets[comm],
             'X': sretsl[comm]}
        )
        aux = aux.dropna()

        model = sm.OLS(aux['Y'], sm.add_constant(aux['X']))
        results = model.fit()

        tstats.loc[h, comm] = results.tvalues['X']

print(tstats)

# Strategy
bos = np.sign(df.resample('M').last().pct_change(12).shift(1))
strat = bos * (0.1/(vols*np.sqrt(12))) * retm
strat = (1+strat).cumprod()
strat = 100 * strat / strat.bfill().iloc[0]

port = strat.pct_change().mean(axis=1)
port = (1+port).cumprod()
port = 100 * port / port.bfill().iloc[0]

perf = Performance(port.to_frame('Portfolio TSMOM'))
print(perf.table)



a = 1
