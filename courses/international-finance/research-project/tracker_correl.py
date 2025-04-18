import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.performance import Performance
from allocation import HRP
import numpy as np

df = pd.read_excel('/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                   sheet_name='CDS Trackers',
                   index_col=0)
df.index = pd.to_datetime(df.index)
df.loc["2017-11-15":, 'VENZ'] = np.nan
df.loc["2022-09-12":, 'RUSSIA'] = np.nan
df.loc["2022-09-28":, 'UKRAIN'] = np.nan
df = df.resample("M").last()
# TODO resize the timeseries


cov = df.pct_change(1).cov()
hrp = HRP(cov)
hrp.plot_dendrogram()
hrp.plot_corr_matrix(figsize=(11, 11))
