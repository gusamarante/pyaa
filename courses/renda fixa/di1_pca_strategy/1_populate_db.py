"""
Populate the Database with the raw DI data.

Before running this routine:
- Make sure that all raw csv files are in the correct folder
- Drop the table "di1_raw" from the sqlite db file
"""

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sqlite3
import getpass

# User Defined Parameters
username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2024')

# Read raw files
df_raw = pd.DataFrame()
for year in tqdm(range(2006, 2024 + 1), 'Reading Files'):
    aux = pd.read_csv(save_path.joinpath(f'data/DI1/dados_di1 {year}.csv'),
                      sep=';')
    df_raw = pd.concat([df_raw, aux], axis=0)

df_raw = df_raw.drop(['Unnamed: 0'], axis=1)
df_raw['reference_date'] = pd.to_datetime(df_raw['reference_date'])
df_raw['maturity_date'] = pd.to_datetime(df_raw['maturity_date'])


# Create Connection to DB
db_file = save_path.joinpath(r"di1pca.db")
with sqlite3.connect(db_file) as conn:

    # --- Create the table for raw data ---
    create_table = """
    CREATE TABLE di1_raw (
                         reference_date    date not null,
                         maturity_date     date not null,
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
    # Upload raw data
    df_raw.to_sql(name='di1_raw', con=conn, if_exists='append', index=False)

    # --- Create table for PCs ---
    create_table = """
        CREATE TABLE di1_pca (
                             reference_date    date not null,
                             pc                text not null,
                             pc_value          real not null,
                             window_type       text not null,

        CONSTRAINT pk_di1_pca PRIMARY KEY (reference_date, pc, window_type)
        );
        """
    conn.cursor().execute(create_table)

    # --- Create table for Loadings ---
    create_table = """
        CREATE TABLE di1_loadings (
                             reference_date    date not null,
                             du                int  not null,
                             pc                text not null,
                             loading           real not null,
                             window_type       text not null,

        CONSTRAINT pk_di1_loading PRIMARY KEY (reference_date, du, pc, window_type)
        );
        """
    conn.cursor().execute(create_table)

    # --- Create table for PC Variance ---
    create_table = """
        CREATE TABLE di1_variance (
                             reference_date    date not null,
                             pc                text not null,
                             pc_variance       real not null,
                             window_type       text not null,

        CONSTRAINT pk_di1_loading PRIMARY KEY (reference_date, pc, window_type)
        );
        """
    conn.cursor().execute(create_table)
