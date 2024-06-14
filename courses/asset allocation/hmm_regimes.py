import pandas as pd
import numpy as np
from hmm import GaussianHMM
from utils import AA_LECTURE
import matplotlib.pyplot as plt

df = pd.read_excel(AA_LECTURE.joinpath("Data HMM.xlsx"), sheet_name="Sheet1",
                   index_col=0)
df.index = pd.to_datetime(df.index)
df = df.resample('Q').last()
rets = df.pct_change(1).dropna()

hmm = GaussianHMM(rets, seed=14)
# n_regimes = hmm.select_order(show_chart=True, max_regimes=6, select_iter=1000)


hmm.fit(n_regimes=3)
print("Number of Regimes", hmm.n_regimes)
print("LogL", hmm.score)
print("AIC", hmm.aic)
print("BIC", hmm.bic)
print("Transition Matrix \n", hmm.trans_mat.round(2))
print("Average Duration \n", hmm.avg_duration)
print("Stationary Dist \n", hmm.stationary_dist)
print("Means \n", hmm.means)
print("Vols \n", hmm.vols)
print("Covars \n", hmm.covars)
print("Correls \n", hmm.corrs)

hmm.plot_series(df['Equities'], log_scale=True)

hmm.plot_densities()

(hmm.means*4).plot(kind='bar', grid=True, title="Means")
plt.tight_layout()
plt.show()

(hmm.vols * np.sqrt(4)).plot(kind='bar', grid=True, title="Vols")
plt.tight_layout()
plt.show()

sharpe = (hmm.means / hmm.vols) * np.sqrt(4)
sharpe.plot(kind='bar', grid=True, title="Sharpe")
plt.tight_layout()
plt.show()
