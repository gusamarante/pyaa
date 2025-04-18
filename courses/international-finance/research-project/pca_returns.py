import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.performance import Performance
from allocation import HRP
from sklearn.decomposition import PCA

df = pd.read_excel('/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Research Project/Data.xlsx',
                   sheet_name='CDS Trackers',
                   index_col=0)
df.index = pd.to_datetime(df.index)
df = df.resample("M").last()
df = df.dropna()
# TODO resize the timeseries

pca = PCA(n_components=5)
pca.fit(df)

var_raio = pd.DataFrame(data=pca.explained_variance_ratio_)
loadings = pd.DataFrame(data=pca.components_.T,
                        columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'],
                        index=df.columns)
pcs = pd.DataFrame(data=pca.transform(df.values),
                   columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'],
                   index=df.index)

# TODO Make a better chart / eliminate problems
loadings['PC 1'].plot(kind='bar')
plt.show()
