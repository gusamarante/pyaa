import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/Dados.xlsx',
                   sheet_name="com_na")

df_dna = df.dropna()

# ===== Chart =====
size = 6
ax = sns.lmplot(
    data=df_dna,
    x="gestational_age",
    y="weight_gain",
    hue="patient",
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


