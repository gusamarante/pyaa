from utils import Performance
from data import trackers_di
import numpy as np
import matplotlib.pyplot as plt
from utils import BLUE, RF_LECTURE
from plottable import ColDef, Table
import pandas as pd

size = 5
window = 252
index_start = 100

# Grab data and compute performance
df = trackers_di()

ret = df.pct_change(window)
vol = df.pct_change(1).rolling(252).std().dropna()
sharpe = (ret / vol).dropna(how='all')

df[['DI 0.5y', 'DI 10y']].plot()
plt.show()

contract_ratio = sharpe['DI 0.5y'] / sharpe['DI 10y']
weights = pd.concat(
    [
        -contract_ratio.rename('DI 0.5y'),
        pd.Series(index=contract_ratio.index, data=1, name='DI 10y'),
    ],
    axis=1,
)
ret_bab = (df[['DI 0.5y', 'DI 10y']].pct_change(1) * weights).dropna().sum(axis=1)
ret_bab = (1 + ret_bab).cumprod()

ret_bab.plot()
plt.show()


