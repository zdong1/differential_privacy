from typing import Tuple
import numpy as np
import scipy.stats as stats

def clopper_pearson(k: int, n: int, alpha=0.05) -> Tuple[float, float]:
    """
    Compute the Clopper-Pearson confidence interval for a binomial proportion.
    
    Args:
    k (int): Number of successes.
    n (int): Number of trials.
    alpha (float): Significance level (default: 0.05).
    
    Returns:
    tuple: Lower and upper bounds of the Clopper-Pearson confidence interval.
    """
    assert 0 <= k <= n, "k must be between 0 and n, inclusive"

    if k == 0:
        lower = 0
    else:
        lower = stats.beta.ppf(alpha / 2, k, n - k + 1)

    if k == n:
        upper = 1
    else:
        upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)

    return lower, upper
