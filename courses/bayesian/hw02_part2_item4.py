from scipy.stats import t, norm, laplace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

m_values = [500, 5000, 50000, 500000]
n_values = [100, 1000, 10000, 100000]
r_times = 1000

df = []
for m in m_values:
    for n in n_values:
        for r in range(r_times):
            sample = t.rvs(df=3, size=m)
            w_a = norm.pdf(sample, loc=0, scale=1) / t.pdf(sample, df=3)
            w_a = w_a / w_a.sum()
            draws_a = np.random.choice(sample, p=w_a, size=n, replace=True)

            # TODO parei aqui
            aux = pd.Series({
                "M": m,
                "N": n,
                "prob": (draws_a >= 2).mean(),  # TODO VECTOR
                "var prob": 2,
            })
            df.append(aux)

df = pd.concat(df, axis=0)

print(df)





# ===== LAPLACE =====
# compute weights for target \pi_b
# w_b = laplace.pdf(sample, loc=0, scale=1) / t.pdf(sample, df=3)
# w_b = w_b / w_b.sum()
# draws_b = np.random.choice(sample, p=w_b, size=n, replace=True)