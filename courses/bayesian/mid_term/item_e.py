import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

data = pd.read_excel('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/Dados.xlsx',
                   sheet_name="com_na")
data = data.set_index(['patient', 'visit'])

gibbs = pd.read_csv('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/samples.csv',
                 index_col=0)
gibbs = gibbs.iloc[1000:].reset_index(drop=True)


patients = [1, 2, 4]
pred = pd.DataFrame()
for row in tqdm(gibbs.iterrows()):
    for p in patients:
        pred.loc[row[0], f"patient {p}"] = row[1].loc[f"alpha {p}"] + row[1].loc[f"beta {p}"] * data.loc[(p, 7), "gestational_age"]

pred.hist()
plt.tight_layout()
plt.show()
