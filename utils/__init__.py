from .charts import (
    BLUE,
    GREEN,
    RED,
    YELLOW,
)
from .paths import (
    AA_LECTURE,
    ASSET_ALLOCATION,
    EMPIRICAL_FINANCE,
    STATISTICS_BAYESIAN,
    STOCHASTIC_CALCULUS,
)
from .performance import Performance, compute_eri
from .stats import cov2corr, corr2cov

__all__ = [
    # Lectures
    "AA_LECTURE",

    # Book
    "ASSET_ALLOCATION",
    "EMPIRICAL_FINANCE",
    "STATISTICS_BAYESIAN",
    "STOCHASTIC_CALCULUS",

    # Custom Colors
    "BLUE",
    "GREEN",
    "RED",
    "YELLOW",

    # Others
    'compute_eri',
    'corr2cov',
    'cov2corr',
    'Performance',
]
