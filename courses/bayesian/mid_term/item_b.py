import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm


df = pd.read_excel('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/Dados.xlsx',
                   sheet_name="com_na")

df_dna = df.dropna()

# ===== Chart =====
sns.lmplot(
    data=df_dna,
    x="gestational_age",
    y="weight_gain",
    height=4,
    aspect=16 / 9,
    ci=False,
    legend=False,
    scatter_kws={'alpha': 0.4},
)
plt.grid(True, color="grey", alpha=0.1)
plt.tight_layout()
plt.savefig("/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/figures/Item B 1.pdf")
plt.show()
plt.close()


# ===== OLS for pool of patients =====
y = df_dna['weight_gain']
x = df_dna['gestational_age']

mod = sm.OLS(endog=y, exog=sm.add_constant(x))
res = mod.fit()

print(res.params)

print(np.sqrt(res.scale))
