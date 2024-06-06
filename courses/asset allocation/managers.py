import pandas as pd
from data.data_api import SGS
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from allocation import HRP
from utils import Performance


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


# All cumulative ERI
df_cum = []
for fund in df_trackers.columns:
    df_cum.append(df_trackers[fund].dropna().reset_index(drop=True))

df_cum = pd.concat(df_cum, axis=1)
df_cum.index = df_cum.index / 252


# =======================
# ===== Performance =====
# =======================
perf = Performance(df_trackers, skip_dd=True)
a = 1


# ===========================================
# ===== Chart - Cumulative ERI Together =====
# ===========================================
fig = plt.figure(figsize=(7 * (16 / 7.3), 7))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Brazilian Hedge Funds - Excess Return Index")
ax.plot(df_cum)
ax.axhline(1, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5, which="both")
ax.set_yscale('log')
ax.set_ylabel("Index (log-scale)")
ax.set_xlabel("Years since start of series")
ax.get_yaxis().set_major_formatter(ScalarFormatter())


plt.tight_layout()

plt.show()
plt.close()


# ============================================
# ===== Chart - Correlation + Dendrogram =====
# ============================================
cov = df_trackers.dropna().pct_change(21).cov()
hrp = HRP(cov)
hrp.plot_dendrogram()
hrp.plot_corr_matrix()
