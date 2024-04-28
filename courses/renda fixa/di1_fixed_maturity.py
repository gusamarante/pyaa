from pathlib import Path
from tqdm import tqdm
import getpass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# User defined parameters
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 250)

# Path to save outputs
username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2024')

# get data
df_raw = pd.DataFrame()
for year in tqdm(range(2006, 2023 + 1), 'Reading Files'):
    aux = pd.read_csv(f'/Users/{username}/PycharmProjects/pyaa/trackers/input data/dados_di1 {year}.csv',
                      sep=';')
    df_raw = pd.concat([df_raw, aux], axis=0)
df_raw = df_raw.drop(['Unnamed: 0'], axis=1)
df_raw['reference_date'] = pd.to_datetime(df_raw['reference_date'])
df_raw['maturity_date'] = pd.to_datetime(df_raw['maturity_date'])

df_curve = df_raw.pivot(index='reference_date', columns='du', values='rate')


# ================================
# ===== Linear Interpolation =====
# ================================
df_linear = df_curve.interpolate(method='linear', axis=1, limit_area='inside')
df_linear = df_linear.dropna(how='any', axis=1)

df_linear[[252, 252*5]].plot(title='Linear Interpolation', legend=True, grid=True)
plt.show()


# ======================================
# ===== Cubic Spline Interpolation =====
# ======================================
df_cubic = df_curve.interpolate(method='cubic', axis=1, limit_area='inside')
df_cubic = df_cubic.dropna(how='any', axis=1)

df_cubic[[252, 252*5]].plot(title='Cubic Spline Interpolation', legend=True, grid=True)
plt.show()


# ======================================
# ===== Flat-Forward Interpolation =====
# ======================================
lndisc = np.log(1 / ((1 + df_curve) ** (df_curve.columns/252)))
lndisc = lndisc.interpolate(method='linear', axis=1, limit_area='inside')
df_ff = (1 / np.exp(lndisc)) ** (252 / lndisc.columns) - 1

df_ff[[252, 252*5]].plot(title='Flat-Forward Interpolation', legend=True, grid=True)
plt.show()


# ===========================
# ===== Compare Methods =====
# ===========================
df2plot = pd.concat([df_linear[5*252].rename('Linear'),
                     df_cubic[5*252].rename('Cubic Splines'),
                     df_ff[5*252].rename('Flat-Forward')],
                    axis=1)
df2plot.plot(title='Compare Interpolation Methods', legend=True, grid=True)
plt.show()
