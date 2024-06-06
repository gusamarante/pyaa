import pandas as pd
from data.data_api import SGS


# ================
# ===== DATA =====
# ================
# Read the Names
names = pd.read_excel(r"C:\Users\gamarante\Dropbox\Personal Portfolio\data\Gestores.xlsx",
                      sheet_name='Tickers', index_col=0)
names = names['Fund Name']
for term in [' FIC ', ' FIM ', ' FI ', ' MULT ', ' LP ']:
    names = names.str.replace(term, ' ')


# Read Managers
df = pd.read_excel(r"C:\Users\gamarante\Dropbox\Personal Portfolio\data\Gestores.xlsx",
                   sheet_name='BBG', skiprows=3, index_col=0)
df.index = pd.to_datetime(df.index)
df = df.rename(names, axis=1)
df = df.dropna(how='all').ffill()

# CDI
cdi = SGS().fetch({12: "CDI"})
cdi = cdi["CDI"] / 100

# Excess Returns
df_xr = []
for fund in df.columns:
    ret = df[fund].pct_change(1).dropna()
    ret = ret - cdi
    df_xr.append(ret.rename(fund))

df_xr = pd.concat(df_xr, axis=1)
df_trackers = (1 + df_xr).cumprod()
df_trackers = df_trackers / df_trackers.bfill().iloc[0]
df_trackers = df_trackers.dropna(how='all')

df_trackers.to_clipboard()

# TODO performance from the start
# TODO Correlation Linkage
# TODO Performance Table
