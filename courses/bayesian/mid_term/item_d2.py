"""
This routine analyses the samples from the posterior, it does not generate them
"""
import pandas as pd

df = pd.read_csv('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/samples.csv',
                 index_col=0)
print(df)