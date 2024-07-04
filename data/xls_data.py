from .utils import output_path, intput_path
from tqdm import tqdm
import pandas as pd


def raw_di():
    last_year = 2024
    data = pd.DataFrame()
    for year in tqdm(range(2006, last_year + 1), 'Reading files'):
        aux = pd.read_csv(intput_path.joinpath(f'dados_di1 {year}.csv'), sep=';')
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
