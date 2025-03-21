import numpy as np
import pandas as pd
from calendars import DayCounts
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import Slider

start_rate = 0.1435
grid_size = 100
rate_eps = 0.0001
dc = DayCounts(dc="bus/252", calendar='anbima')

# Function that returns the price of a NTN-F
def pu(
        rate,
        coupon_rate=0.1,
        ref_date=pd.to_datetime("2025-03-20"),
        maturity=pd.to_datetime("2029-01-01"),
):
    # Find the next coupon date, which will be the start of the stream
    # of cashflows
    if ref_date.month < 7:
        next_coupon_date = pd.to_datetime(datetime(ref_date.year, 7, 1))
    else:
        next_coupon_date = pd.to_datetime(datetime(ref_date.year + 1, 1, 1))

    # Generates cashflow dates
    cf_dates = pd.date_range(
        start=next_coupon_date,
        end=maturity,
        freq="6MS",
        inclusive="both",
    )
    cf_dates = dc.following(cf_dates)
    dus = dc.days(ref_date, cf_dates)

    df_bond = pd.DataFrame(
        index=cf_dates,
        data={
            'cashflows': ((1 + coupon_rate) ** 0.5 - 1) * 1000,
            'du': dus,
        },
    )
    df_bond.loc[cf_dates[-1], "cashflows"] = df_bond.loc[cf_dates[-1], "cashflows"] + 1000
    df_bond["discount"] = 1 / ((1 + rate)**(df_bond["du"] / 252))
    price = (df_bond["cashflows"] * df_bond["discount"]).sum()
    return price


def dv01(
        rate,
        coupon_rate=0.1,
        ref_date=pd.to_datetime("2025-03-20"),
        maturity=pd.to_datetime("2029-01-01"),
):
    pu_plus = pu(rate - rate_eps, coupon_rate, ref_date, maturity)
    pu_minus = pu(rate + rate_eps, coupon_rate, ref_date, maturity)
    dv = (pu_minus - pu_plus) / (2 * rate_eps)
    return dv


print(pu(start_rate))


# =================
# ===== Chart =====
# =================
size = 5
fig = plt.figure(figsize=(size * (16 / 7.3), size))

max_rate = start_rate + 0.1
rate2plot = np.arange(0, max_rate, max_rate / grid_size)

ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(rate2plot, [pu(r) for r in rate2plot])
ax.set_xlim(0, max_rate)
ax.set_ylim(0, None)
ax.set_ylabel("Price")
ax.set_xlabel("Yield to Maturity")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.show()


# TODO add dot for the bond
# TODO função de DV01
# TODO plot DV01/duration approx
# TODO função de Duration
# TODO função de Convexity
# TODO plot convex approximation
# TODO add rate slider
# TODO add coupon slider
# TODO add maturity slider

