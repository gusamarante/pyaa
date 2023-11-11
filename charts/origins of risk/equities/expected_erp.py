"""
Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
FFBA08 - Yellow
F25F5C - Red
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
import numpy as np

from bwbbgdl import GoGet
from bwmktdata import Macrobond
from bwsecrets.api import get_secret

from utils import Performance

size = 5

# ===== Read the Data =====
# file_path = r"C:/Users/gamarante/Dropbox/Aulas/Asset Allocation/Dados BBG AA Course.xlsx"
# df = pd.read_excel(file_path, sheet_name='TOT_RETURN_INDEX_GROSS_DVDS', skiprows=4, index_col=0)
# df = df.sort_index()
# df = df.dropna(how='all')
#
# rename_tickers = {
#     "SPX Index": "S&P500",
#     "SXXP Index": "EuroStoxx 600",
#     "TPX Index": "Topix",
#     "IBOV Index": "Ibovespa",
# }
# df = df.rename(rename_tickers, axis=1)

# Macrobond
passwords = get_secret("macrobond")
mb = Macrobond(client_id=passwords["client_id"], client_secret=passwords["client_secret"])
q_tickers = {
    "usnaac0169": "US GDP",
    "eueunaac0149": "Europe GDP",
    "jpnaac0004": "Japan GDP",
    "brnaac0016": "Brazil GDP",
}
d_tickers = {
    "us10yzcyz": "US 10y",
    "eu10yspyc": "Europe 10y",
    "jp10yycm": "Japan 10y",
    # "": "Brazil 10y",
}
df_q = mb.fetch_series(q_tickers)

df_hp = pd.DataFrame()
for col in df_q.columns:
    _, trend = hpfilter(df_q[col].dropna())
    df_hp = pd.concat([df_hp, trend], axis=1)

df_growth = df_hp.rolling(4).mean().pct_change(4)
df_growth.plot()
plt.show()
