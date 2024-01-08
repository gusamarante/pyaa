"""
genarates the rolling window PCA of the DI curve. Includes an entry
level and an exit level based on the PC levels.
"""

from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
import getpass
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# User defined parameters
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 250)
start_date = '2023-11-01'

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


# ===========================
# ===== Build the curve =====
# ===========================
df_curve = df_raw.pivot(index='reference_date', columns='du', values='rate')
df_curve = df_curve.interpolate(axis=1, method='cubic')
df_curve = df_curve.dropna(how='any', axis=1)
df_curve.index = pd.to_datetime(df_curve.index)

# ======================
# ===== Build DV01 =====
# ======================
df_dv01 = df_raw.pivot(index='reference_date', columns='du', values='dv01')
df_dv01 = df_dv01 * 10_000  # PCA-DV01 requires move per unit of PC, so the DV01 has to be per unit of rate


# ================================
# ===== Backtested PC Signal =====
# ================================
df_pca = pd.DataFrame(columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])
df_loadings = pd.DataFrame(columns=['date', 'du', 'PC 1', 'PC 2', 'PC 3', 'PC 4'])
dates2loop = df_curve.index[df_curve.index >= start_date]

for date in tqdm(dates2loop, 'Generating Signal'):
    pca = PCA(n_components=4)
    data = df_curve.loc[:date].tail(252 * 5)
    pca.fit(data.values)

    current_loadings = pd.DataFrame(data=pca.components_,
                                    index=['PC 1', 'PC 2', 'PC 3', 'PC 4'],
                                    columns=df_curve.columns).T
    current_pca = pd.DataFrame(data=pca.transform(data.values),
                               columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'],
                               index=data.index)

    # Normalize the signal
    signal = np.sign(current_loadings.iloc[-1])
    current_loadings = current_loadings * signal
    current_pca = current_pca * signal

    df_pca.loc[date] = current_pca.iloc[-1]

    current_loadings = current_loadings.reset_index()
    current_loadings['date'] = date
    df_loadings = pd.concat([df_loadings, current_loadings])

df_loadings['date'] = pd.to_datetime(df_loadings['date'])



# ===== Custom Function =====
def get_portfolio(current_date, pcadv01):
    """
    given a date and the desired exposition vector, returns the chosen contracts
    """
    current_date = pd.to_datetime(current_date)

    current_loadings = df_loadings[df_loadings['date'] == current_date]
    current_loadings = current_loadings.drop('date', axis=1).set_index('du')
    current_pca = df_pca.loc[current_date]

    available_maturities = df_dv01.loc[current_date].dropna().index
    available_maturities = available_maturities[df_curve.columns.min() <= available_maturities]
    available_maturities = available_maturities[df_curve.columns.max() >= available_maturities]

    aux_pcadv01 = current_loadings.loc[available_maturities]
    aux_pcadv01 = aux_pcadv01.multiply(df_dv01.loc[current_date, available_maturities], axis=0)
    aux_pcadv01 = aux_pcadv01.astype(float)

    # Choose 4 contracts
    vertices_du = aux_pcadv01.idxmax().sort_values().values

    selected_portfolio = pd.DataFrame(index=vertices_du)
    cond_date = df_raw['reference_date'] == current_date
    cond_du = df_raw['du'].isin(vertices_du)
    current_data = df_raw[cond_date & cond_du].sort_values('du')

    # TODO parei aqui
    selected_portfolio['contracts'] = current_data['contract'].values
    selected_portfolio['pu'] = current_data['theoretical_price'].values
    selected_portfolio[['Loadings 1', 'Loadings 2', 'Loadings 3', 'Loadings 4']] = current_loadings.loc[vertices_du]
    # selected_portfolio[['PCADV01 1', 'PCADV01 2', 'PCADV01 3']] = aux_pcadv01.loc[vertices_du]
    # selected_portfolio[['PC 1', 'PC 2', 'PC 3']] = current_pca.values
    #
    # coeff = selected_portfolio[['PCADV01 1', 'PCADV01 2', 'PCADV01 3']].T.values
    # constants = np.array(pcadv01)
    # selected_portfolio['quantities'] = np.linalg.inv(coeff) @ constants

    # return selected_portfolio
    return 2


get_portfolio(start_date, 2)
