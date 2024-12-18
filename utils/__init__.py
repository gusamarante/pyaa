from .charts import color_palette
from .paths import (
    AA_LECTURE,
    EMPIRICAL_FINANCE,
    STATISTICS_BAYESIAN,
    STOCHASTIC_CALCULUS,
)
from .performance import Performance, compute_eri
from .stats import cov2corr, corr2cov

__all__ = [
    "AA_LECTURE",
    "EMPIRICAL_FINANCE",
    "STATISTICS_BAYESIAN",
    "STOCHASTIC_CALCULUS",
    'color_palette',
    'compute_eri',
    'corr2cov',
    'cov2corr',
    'Performance',
]
