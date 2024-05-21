from allocation import HRP
import pandas as pd
from pathlib import Path
from getpass import getuser

file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation")
df = pd.read_excel(file_path.joinpath('Commodities Total Return.xlsx'), index_col=0)
df.index = pd.to_datetime(df.index)
df = df.resample('M').last()
df = df.pct_change(1, fill_method=None)
df = df.dropna()
cov = df.cov()

hrp = HRP(cov)
hrp.plot_corr_matrix(show_chart=True,
                     save_path=file_path.joinpath('Figures/HRP Correlation Matrix.pdf'))
hrp.plot_dendrogram(show_chart=True,
                    save_path=file_path.joinpath('Figures/HRP Dendrogram.pdf'),
                    )
