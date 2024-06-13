import pandas as pd
from hmm import GaussianHMM
from utils import AA_LECTURE

df = pd.read_excel(AA_LECTURE.joinpath("Data HMM.xlsx"), sheet_name="Sheet1",
                   index_col=0)
df.index = pd.to_datetime(df.index)
df = df[['Equities', 'Commodities']]
df = df.resample('Q').last()
rets = df.pct_change(1).dropna()

hmm = GaussianHMM(rets)
# n_regimes = hmm.select_order(show_chart=True, max_regimes=6, select_iter=1000)


hmm.fit()
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
print("Covars \n", hmm.corrs)







