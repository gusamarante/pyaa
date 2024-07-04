from .utils import output_path, input_path
from tqdm import tqdm
import pandas as pd


# ======================
# ===== DI Futures =====
# ======================
def raw_di():
    last_year = 2024
    data = pd.DataFrame()
    for year in tqdm(range(2006, last_year + 1), 'Reading DI files'):
        aux = pd.read_csv(input_path.joinpath(f'dados_di1 {year}.csv'), sep=';')
        data = pd.concat([data, aux])

    data['reference_date'] = pd.to_datetime(data['reference_date'])
    data['maturity_date'] = pd.to_datetime(data['maturity_date'])
    data = data.drop('Unnamed: 0', axis=1)
    return data


def trackers_di():
    file_path = output_path.joinpath("trackers_di1.csv")
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def curve_di():
    file_path = output_path.joinpath("di monthly maturities.csv")
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


# ================
# ===== NTNB =====
# ================
def raw_ntnb():
    last_year = 2024
    ntnb = pd.DataFrame()
    for year in tqdm(range(2003, last_year + 1), 'Reading NTNB files'):
        aux = pd.read_csv(input_path.joinpath(f'dados_ntnb {year}.csv'), sep=';')
        ntnb = pd.concat([ntnb, aux])

    ntnb['reference date'] = pd.to_datetime(ntnb['reference date'])
    ntnb['maturity'] = pd.to_datetime(ntnb['maturity'])
    return ntnb


def trackers_ntnb():
    file_path = output_path.joinpath("trackers_ntnb.csv")
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df
