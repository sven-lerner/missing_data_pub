import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
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


def get_cor_mat(char_matrix, ct=None, ct_int=None):
    """
    
    """
    C = char_matrix.shape[1]
    ret_mat = np.zeros((C, C))
    for i in range(C):
        ret_mat[i,i] = 1
        for j in range(i+1, C):
            available = np.logical_and(~np.isnan(char_matrix[:,i]),
                                       ~np.isnan(char_matrix[:,j]))
            corr = np.corrcoef(char_matrix[available, i], y=char_matrix[available, j])[0,1]
            ret_mat[i,j] = corr
            ret_mat[j,i] = corr
    return ret_mat


def get_optimal_A(B, A, present, cl, idxs=[], reg=0, min_chars=1, infer_lr_entries=False,
                 require_2x_present=False, weight=None, adaptive_reg=False, min_reg=None, max_reg=None):
    """
    Get optimal A for cl = AB given that X is (potentially) missing data
    Parameters
    ----------
        B : matrix B
        A : matrix A, will be overwritten
        present: boolean mask of present data
        cl: matrix cl
        idxs: indexes which to impute
        reg: optinal regularization penalty
        min_chars: minimum number of entries to require present
        infer_lr_entries: optionally require fewer entries present, regress on first
            i columns of B where i is the number of observed entries
    """
    for i in idxs:
        present_i = present[i,:]
        Xi = cl[i,present_i]
        Bi = B[:,present_i]
        if weight is None:
            bitbi = Bi.dot(Bi.T)
        else:
            W_i = np.diag(weight[present_i])
            bitbi = Bi @ W_i @ Bi.T
            Bi = Bi @ W_i 
            
        assert np.sum(np.isnan(bitbi)) == 0, "should have no nans"
        assert np.sum(np.isinf(bitbi)) == 0, "should have no infs"
        if adaptive_reg:
#             effective_reg = np.log(np.logspace(reg, max_reg, 36)[np.sum(~present_i)])
            effective_reg = np.logspace(min_reg, max_reg, 36)[np.sum(~present_i)]
#             effective_reg = np.square(np.linspace(0.1, np.sqrt(10), 36))[np.sum(~present_i)]
        else:
            effective_reg = reg
        A[i,:] = np.linalg.solve(bitbi + np.eye(Bi.shape[0])*effective_reg, Bi.dot(Xi.T))
    return A
