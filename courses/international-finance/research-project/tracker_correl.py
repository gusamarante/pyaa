import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.performance import Performance
from allocation import HRP

df = pd.read_excel('/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                   sheet_name='CDS Trackers',
                   index_col=0)
df.index = pd.to_datetime(df.index)
df = df.resample("M").last()
# TODO resize the timeseries


cov = df.pct_change(1).cov()
hrp = HRP(cov)
hrp.plot_dendrogram()
hrp.plot_corr_matrix(figsize=(11, 11))
