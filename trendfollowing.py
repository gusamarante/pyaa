import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(io=r"C:\Users\gamarante\Dropbox\Personal Portfolio\data\FIPs.xlsx",
                   sheet_name='Trackers',
                   index_col=0)

df = df['XPIE'].dropna()
ret = df.pct_change(1)

k_fast = 21 * 3
k_slow = 21 * 6
df_fast = ret.rolling(k_fast).mean().shift(1)  # 1-month lag
df_slow = ret.rolling(k_slow).mean().shift(1)

is_bull = (df_fast >= 0) & (df_slow >= 0)
is_bear = (df_fast < 0) & (df_slow < 0)
is_corr = (df_fast < 0) & (df_slow >= 0)
is_rebo = (df_fast >= 0) & (df_slow < 0)

ret_fast = ret * (df_fast >= 0) + (-ret) * (df_fast < 0)
ret_slow = ret * (df_slow >= 0) + (-ret) * (df_slow < 0)

aco = 0.5
are = 0.5

ret_dyn = is_bull * ret + is_bear * (-ret) + is_corr * ((1 - aco) * ret_slow + aco * ret_fast) + is_rebo * ((1 - are) * ret_slow + are * ret_fast)
ret_dyn = ret_dyn.fillna(0)
dyn = (1 + ret_dyn).cumprod()
dyn = df.iloc[0] * dyn / dyn.iloc[0]
df = pd.concat([df, dyn.rename('Dynamic Trend')], axis=1)

df.plot()
plt.show()
