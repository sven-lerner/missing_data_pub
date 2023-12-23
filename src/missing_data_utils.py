import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def get_cov_mat(char_matrix, ct=None, ct_int=None):
    """
    get estimate of covariance matrix with potentially missing data
    Parameters
    ----------
        char_matrix : input matrix
        ct : input matrix, nans replaced with zero
        ct_int : 0-1 integer matrix indicating present data in char_matrix
    """
    if ct is None:
        ct_int = (~np.isnan(char_matrix)).astype(int)
        ct = np.nan_to_num(char_matrix)
    temp = ct.T.dot(ct)
    temp_counts = ct_int.T.dot(ct_int)
    sigma_t = temp / temp_counts
    return sigma_t
