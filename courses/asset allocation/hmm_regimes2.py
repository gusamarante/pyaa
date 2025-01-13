import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import torch
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM

# User defined parameters
n_states = 4

# Read data
file_path = r"C:\Users\gamarante\Dropbox\NAVs.xlsx"
df = pd.read_excel(file_path, index_col=0)
df.index = pd.to_datetime(df.index)
rets = df.resample("M").last().pct_change(1).dropna()

# The model
hmm = DenseHMM(
    distributions=[Normal() for _ in range(n_states)],
    verbose=True,
)
hmm = hmm.fit(X=np.array([rets.values]))

trans_prob = pd.DataFrame(np.exp(np.array(hmm.edges)))
trans_prob = trans_prob.div(trans_prob.sum(axis=1), axis=0)  # Reduce numerical error


# Stationary distribution
vals, vecs = np.linalg.eig(trans_prob)
stat_dist = pd.Series(vecs[:, np.argmax(vals)])
stat_dist = stat_dist * np.sign(stat_dist)
stat_dist = stat_dist / stat_dist.sum()

stat_dist.plot(kind="bar")
plt.show()


# States probabilities
state_probs = pd.DataFrame(
    data=hmm.predict_proba(np.array([rets.values]))[0],
    index=rets.index,
)

state_probs.plot()
plt.show()


# Predicted / Most likely State
state_pred = pd.Series(
    data=hmm.predict(np.array([rets.values]))[0],
    index=rets.index,
)
state_pred.plot()
plt.show()

# TODO Normal Mixtures

