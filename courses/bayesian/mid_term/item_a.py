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
    hue="patient",
    height=4,
    aspect=16 / 9,
    ci=False,
    legend=False,
    scatter_kws={'alpha': 0.4,
                 's': 4},
    line_kws={'lw': 0.6},
)
plt.grid(True, color="grey", alpha=0.1)
plt.tight_layout()
plt.savefig("/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/figures/Item A 1.pdf")
plt.show()
plt.close()


# ===== OLS for each patient =====
ols_all = pd.DataFrame()
for p in tqdm(df_dna['patient'].unique()):
    aux = df_dna[df_dna['patient'] == p]
    y = aux['weight_gain']
    x = aux['gestational_age']

    mod = sm.OLS(endog=y, exog=sm.add_constant(x))
    res = mod.fit()

    ols_all.loc[p, "alpha"] = res.params['const']
    ols_all.loc[p, "beta"] = res.params['gestational_age']
    ols_all.loc[p, "sigma_eps"] = np.sqrt(res.scale)


print(ols_all.describe())

ols_all.hist()
plt.tight_layout()
plt.savefig("/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/figures/Item A 2.pdf")
plt.show()
plt.close()
