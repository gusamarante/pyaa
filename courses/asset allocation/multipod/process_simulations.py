import pandas as pd
import matplotlib.pyplot as plt

"""
Statistics
'Average Sharpe of Pods', 
'Sharpe of Port of Pods', 
'Net Sharpe',
       
'Alpha Share Mean', 
'Alpha Share Median', 

Parameters
'Number of Pods',
'Leverage Factor', 
'Exante Sharpe of Pods', 
'Avg Correl of Pods',
"""
df = pd.read_csv("/Users/gustavoamarante/Library/CloudStorage/Dropbox/multipod DATA.csv")
df = df.drop("Unnamed: 0", axis=1)


# =====================
# ===== BAR CHART =====
# =====================
# 3 parameters fixed, 1 parameter varying in the X axis, chosen variable in the Y axis.
# fixed = {
#     'Number of Pods': 10,
#     # 'Leverage Factor': 1,
#     'Exante Sharpe of Pods': 0.75,
#     'Avg Correl of Pods': 0.1,
# }
# varying = 'Leverage Factor'
# variable = 'Alpha Share Median'
#
# # Filter data
# df_plot = df.copy()
# title = ''
# for k, v in fixed.items():
#     df_plot = df_plot[df_plot[k] == v]
#     title = title + f"{k} = {v}   "
#
# df_plot = df_plot.pivot_table(
#     index=varying,
#     values=variable,
# )
# df_plot = df_plot[variable]
#
# # Chart
# fig = plt.figure(figsize=(5 * (16 / 9), 5))
# ax = plt.subplot2grid((1, 1), (0, 0))
#
# ax.bar(df_plot.index, df_plot.values)
#
# ax.axhline(color='black', lw=0.5)
# ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.set_ylabel(variable)
# ax.set_xlabel(varying)
# ax.set_title(title)
# plt.tight_layout()
# plt.show()


# ==========================
# ===== HUE LINE CHART =====
# ==========================
# 2 parameters fixed, 2 parameter varying in the X axis and Hue lines, chosen variable in the Y axis
fixed = {
    # 'Number of Pods': 5,
    'Leverage Factor': 3,
    'Exante Sharpe of Pods': 0.75,
    # 'Avg Correl of Pods': 0.1,
}
varying_x = 'Number of Pods'
varying_hue = 'Avg Correl of Pods'
variable = 'Net Sharpe'

# Filter data
df_plot = df.copy()
title = ''
for k, v in fixed.items():
    df_plot = df_plot[df_plot[k] == v]
    title = title + f"{k} = {v}   "

df_plot = df_plot.pivot_table(
    index=varying_x,
    columns=varying_hue,
    values=variable,
)

# Chart
fig = plt.figure(figsize=(5 * (16 / 9), 5))
ax = plt.subplot2grid((1, 1), (0, 0))

for col in df_plot:
    ax.plot(df_plot[col].index, df_plot[col].values, label=col)

ax.legend(loc='best', frameon=True)
ax.get_legend().set_title(varying_hue)

ax.axhline(color='black', lw=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_ylabel(variable)
ax.set_xlabel(varying_x)
ax.set_title(title)
plt.tight_layout()
plt.show()
