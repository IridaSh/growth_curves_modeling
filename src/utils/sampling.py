from scipy.stats import truncnorm
import numpy as np

def sample_truncated_normal(mean, sd, low, upp, size=None):
    """
    Samples from a truncated normal distribution.

    Parameters:
    - mean (float): Mean of the distribution.
    - sd (float): Standard deviation of the distribution.
    - low (float): Lower bound of the distribution.
    - upp (float): Upper bound of the distribution.
    - size (int or tuple of ints, optional): Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.

    Returns:
    - samples (float or ndarray): Random variates from the truncated normal distribution.
    """
    a, b = (low - mean) / sd, (upp - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)