import pandas as pd
from pathlib import Path
from getpass import getuser


dir_path = Path(f"/Users/{getuser()}/PycharmProjects/pyaa/trackers/output data")


def trackers_di():
    file_path = dir_path.joinpath("trackers_di1.xlsx")
    df = pd.read_excel(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df
