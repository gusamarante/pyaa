from scipy.optimize import minimize
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np


def bootstrapp(cashflows, prices):
    """
    Bootstrap method to build a zero curve from a set of bond cashflows and prices.

    Parameters
    __________
    cashflows : pandas.DataFrame
        Index are the cashflow dates and columns are the name of the bonds

    prices : pd.Series
        prices of bonds. The index names must match the columns of `cashflows`
    """

    # Find the DUs that we can change
    du_dof = cashflows.idxmax().values

    def objective_function(disc):
        dus = np.insert(du_dof, 0, 0)  # add the first value, which will be fixed at zero
        disc = np.insert(disc, 0, 1)  # add the first value, which will be fixed at one
        f = interp1d(dus, np.log(disc))  # Interpolation of the log of discounts
        disc = pd.Series(index=cashflows.index, data=np.exp(f(cashflows.index)))  # Populate the discounts to a series
        sum_dcf = cashflows.multiply(disc, axis=0).sum()  # get the sum of discounted cashflows
        erros = prices.subtract(sum_dcf, axis=0)  # Difference between actual prices and sum of DCF
        erro_total = (erros ** 2).sum()  # Sum of squarred errors

        try:
            erro_total = erro_total.values[0]
        except AttributeError:
            erro_total = erro_total

        return erro_total

    # Run optimization
    # Initial gues for the vector of disccounts
    init_discount = 0.8 * np.ones(len(du_dof))
    res = minimize(fun=objective_function,
                   x0=init_discount,
                   method=None,
                   tol=1e-16,
                   options={'disp': False})

    dus = np.insert(du_dof, 0, 0)  # add the first value, which will be fixed at zero
    discount = np.insert(res.x, 0, 1)  # add the first value, which will be fixed at one
    f = interp1d(dus, np.log(discount))  # Interpolation of the log of disccounts
    discount = pd.Series(index=cashflows.index, data=np.exp(f(cashflows.index)))

    curve = (1 / discount) ** (252 / discount.index) - 1

    return curve
