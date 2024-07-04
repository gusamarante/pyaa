from .utils import output_path, input_path
from tqdm import tqdm
import pandas as pd


last_year = 2024  # Year of the last file available

# ======================
# ===== DI Futures =====
# ======================
def raw_di():
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


# ========================
# ===== NTNF and LTN =====
# ========================
def raw_ltn_ntnf():
    # Read the Data - LTN
    ltn = pd.DataFrame()
    for year in tqdm(range(2003, last_year + 1), 'Reading LTN files'):
        aux = pd.read_csv(input_path.joinpath(f'dados_ltn {year}.csv'), sep=';')
        ltn = pd.concat([ltn, aux])

    ltn['reference date'] = pd.to_datetime(ltn['reference date'])
    ltn['maturity'] = pd.to_datetime(ltn['maturity'])
    ltn = ltn.drop(['Unnamed: 0', 'index'], axis=1)

    # Read the Data - NTNF
    ntnf = pd.DataFrame()
    for year in tqdm(range(2003, last_year + 1), 'Reading NTNF files'):
        aux = pd.read_csv(input_path.joinpath(f'dados_ntnf {year}.csv'), sep=';')
        ntnf = pd.concat([ntnf, aux])

    ntnf['reference date'] = pd.to_datetime(ntnf['reference date'])
    ntnf['maturity'] = pd.to_datetime(ntnf['maturity'])
    ntnf = ntnf.drop(['Unnamed: 0', 'index'], axis=1)

    # Put both bonds together
    ntnf = pd.concat([ntnf, ltn])

    return ntnf


def trackers_ntnf():
    """
    Although the name is "ntnf", some of the trackers use ltns
    """
    file_path = output_path.joinpath("trackers_ntnf.csv")
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df
