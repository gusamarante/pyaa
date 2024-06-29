"""
Plots the ACM term premium for the US
"""
from data.nyfed import acmtp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import AA_LECTURE

size = 5
df = acmtp()
df = df[df.index >= "1985-01-01"]

# =================
# ===== Chart =====
# =================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

# 24 months
mat = str(2).zfill(2)
ax = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
ax.plot(df[f"ACMY{mat}"].dropna(), label="Zero Yield", color="#3333B2", lw=2)
ax.plot(df[f"ACMRNY{mat}"].dropna(), label="Risk Neutral Yield", color="#F25F5C", lw=2)
ax.set_title("Yields - 24 months")
ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

ax = plt.subplot2grid((3, 2), (2, 0))
ax.plot(df[f"ACMTP{mat}"].dropna(), label="Zero Yield", color="#3333B2", lw=2)
ax.set_title("Term Premium - 24 months")
ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

# 24 months
mat = str(10).zfill(2)
ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
ax.plot(df[f"ACMY{mat}"].dropna(), label="Zero Yield", color="#3333B2", lw=2)
ax.plot(df[f"ACMRNY{mat}"].dropna(), label="Risk Neutral Yield", color="#F25F5C", lw=2)
ax.set_title("Yields - 120 months")
ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

ax = plt.subplot2grid((3, 2), (2, 1))
ax.plot(df[f"ACMTP{mat}"].dropna(), label="Zero Yield", color="#3333B2", lw=2)
ax.set_title("Term Premium - 120 months")
ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")

plt.tight_layout()

plt.savefig(AA_LECTURE.joinpath("Figures/Bonds - US Term Premium ACM.pdf"))
plt.show()
plt.close()
