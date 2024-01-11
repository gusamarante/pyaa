"""
Calculate the rolling PCA signals and loadings
"""

from sklearn.decomposition import PCA
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import sqlite3
import getpass

# User Defined Parameters
username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2024')
window = 5  # in years

# Create Connection to DB
db_file = save_path.joinpath(r"di1pca.db")
with sqlite3.connect(db_file) as conn:
    # Query all the raw data
    query = """
    SELECT * FROM di1_raw;
    """
    df = pd.read_sql(query, conn)

df['maturity_date'] = pd.to_datetime(df['maturity_date'])
df['reference_date'] = pd.to_datetime(df['reference_date'])


# ===========================
# ===== Build the curve =====
# ===========================
df_curve = df.pivot(index='reference_date', columns='du', values='rate')
df_curve = df_curve.interpolate(axis=1, method='cubic')
df_curve = df_curve.dropna(how='any', axis=1)
df_curve.index = pd.to_datetime(df_curve.index)


# ======================
# ===== Build DV01 =====
# ======================
df_dv01 = df.pivot(index='reference_date', columns='du', values='dv01')
df_dv01 = df_dv01 * 10_000  # PCA-DV01 requires move per unit of PC, so the DV01 has to be per unit of rate


# ================================
# ===== Backtested PC Signal =====
# ================================
df_pca = pd.DataFrame(columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])
df_loadings = pd.DataFrame(columns=['reference_date', 'du', 'PC 1', 'PC 2', 'PC 3', 'PC 4'])
df_var = pd.DataFrame(columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])
dates2loop = df_curve.index

for d in tqdm(dates2loop):

    aux = df_curve.loc[:d].tail(252 * 5)

    if len(aux) < 252:
        continue

    pca = PCA(n_components=4)
    pca.fit(aux.values)

    current_loadings = pd.DataFrame(data=pca.components_,
                                    index=['PC 1', 'PC 2', 'PC 3', 'PC 4'],
                                    columns=df_curve.columns).T
    current_pca = pd.DataFrame(data=pca.transform(aux.values),
                               columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'],
                               index=aux.index)

    # Normalize the signal: all effect on the longer end are positive
    signal = np.sign(current_loadings.iloc[-1])
    current_loadings = current_loadings * signal
    current_pca = current_pca * signal

    df_pca.loc[d] = current_pca.iloc[-1]

    current_loadings = current_loadings.reset_index()
    current_loadings['reference_date'] = d
    df_loadings = pd.concat([df_loadings, current_loadings])

    df_var.loc[d] = pca.explained_variance_

# Organize DFs to upload
df_loadings = df_loadings.melt(id_vars=['date', 'du'], var_name='pc', value_name='loading')
df_loadings['window_type'] = f'rolling {window}y'

df_pca.index.name = 'reference_date'
df_pca = df_pca.reset_index()
df_pca = df_pca.melt(id_vars='reference_date', var_name='pc', value_name='pc_value')
df_pca['window_type'] = f'rolling {window}y'

df_var.index.name = 'reference_date'
df_var = df_var.reset_index()
df_var = df_var.melt(id_vars='reference_date', var_name='pc', value_name='pc_variance')
df_var['window_type'] = f'rolling {window}y'


# Create Connection to DB
db_file = save_path.joinpath(r"di1pca.db")
with sqlite3.connect(db_file) as conn:

    # Create PCA table
    create_table = """
    CREATE TABLE di1_pca (
                         reference_date    date not null,
                         du                integer not null,
                         rate              real not null,
                         theoretical_price real not null,
                         pnl               real not null,
                         dv01              real not null,
                         contract          text not null,
                         du                integer not null,
                         volume            integer not null,
                         open_interest     integer not null,

    CONSTRAINT pk_di1_raw PRIMARY KEY (reference_date, contract)
    );
    """
    conn.cursor().execute(create_table)

    # Upload data
    df_raw.to_sql(name='di1_raw', con=conn, if_exists='append', index=False)






