import pandas as pd
import numpy as np


def cov2corr(cov):
    # TODO Documentation
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1  # correct for numerical error
    corr[corr > 1] = 1
    return corr, std


def corr2cov(corr, std):
    # TODO Documentation
    corr_a = np.array(corr)
    std = np.array(std)

    cov = np.diag(std) @ corr_a @ np.diag(std)

    if isinstance(corr, pd.DataFrame):
        cov = pd.DataFrame(data=cov, index=corr.index, columns=corr.columns)

    return cov