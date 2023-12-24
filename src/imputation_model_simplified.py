import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import numpy as np
import os
import missing_data_utils 
import scipy as sp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError

############### Core Factor Imputation Model ####################

def conditional_mean(Sigma, mu, i_mask, i_data):
    '''
    return the conditional mean of missing data give the covariance matrix and mean of the characteristcs
    '''
    Sigma_11 = Sigma[~i_mask, :][:, ~i_mask]
    Sigma_12 = Sigma[~i_mask, :][:, i_mask]
    Sigma_22 = Sigma[i_mask, :][:, i_mask]
    mu1, mu2 = mu[~i_mask], mu[i_mask]
    
    conditional_mean = mu1 + Sigma_12 @ np.linalg.solve(Sigma_22, i_data[i_mask] - mu2)

    assert np.all(~np.isnan(conditional_mean))
    
    return conditional_mean


def get_optimal_A(B, A, present, cl, idxs=[], reg=0, 
                 mu=None,
                 resid_regs=None):
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
    A[:,:] = np.nan
    for i in idxs:
        present_i = present[i,:]
        Xi = cl[i,:]
        Xi = Xi[present_i]
        Bi = B[:,present_i]
        assert np.all(~np.isnan(Bi)) and np.all(~np.isinf(Bi))
        effective_reg = reg 
        if resid_regs is None:
            lmbda = effective_reg * np.eye(Bi.shape[1])
        else:
            lmbda = np.diag(resid_regs[present_i])

        if mu is not None:
            Xi = Xi - mu[present_i]
        try:
            A[i,:] = Bi @ np.linalg.lstsq(Bi.T @ Bi + lmbda, Xi, rcond=0)[0]
        except LinAlgError as e:
            lmbda = np.eye(Bi.shape[1])
            A[i,:] = Bi @ np.linalg.lstsq(Bi.T @ Bi + lmbda, Xi, rcond=0)[0]
    return A


def impute_panel_xp_lp(char_panel, return_panel, min_chars, K, num_months_train,
                      reg=0.01,
                      time_varying_lambdas=False,
                      window_size=1, 
                      n_iter=1,
                      eval_data=None,
                      allow_mean=False,
                      eval_weight_lmbda=True,
                      resid_reg=False,
                      shrink_lmbda=False,
                      run_in_parallel=True):
    '''
    Run the XS imputation as described in the paper
    Default arguments represent configuration as described in the paper
    '''
    char_panel = np.copy(char_panel)
    missing_mask_overall = np.isnan(char_panel)
    char_panel[np.sum(~np.isnan(missing_mask_overall), axis=2) < min_chars] = np.nan
    missing_mask_overall = np.isnan(char_panel)
    imputed_chars = np.copy(char_panel)
    
    lmbda_minchars = min_chars
    T, N, L = char_panel.shape
    
    resid_cov_mats = [None for _ in range(T)]
    mu = np.zeros((T, L), dtype=float)
    resid_regs = None
    for i in range(n_iter):
        lmbda, cov_mat = estimate_lambda(imputed_chars, return_panel, num_months_train=num_months_train,
                                K=K, min_chars=min_chars, window_size=window_size,
                                        time_varying_lambdas=time_varying_lambdas,
                                        eval_weight_lmbda=eval_weight_lmbda,
                                        shrink_lmbda=shrink_lmbda, reg=reg)

        assert np.sum(np.isnan(lmbda)) == 0, f"lambda should contain no nans, {np.argwhere(np.isnan(lmbda))}"

        
        
        gamma_ts = np.zeros((char_panel.shape[0], char_panel.shape[1], K))
        gamma_ts[:,:] = np.nan
        missing_counts = np.sum(np.isnan(char_panel), axis=2)

        return_mask = np.sum(~np.isnan(char_panel), axis=2) >= min_chars
                                     
        
        
            
        def get_gamma_t(ct, present, to_impute, lmbda, time_varying_lambdas, t,
                        mean, resid_regs):
            gamma_func = get_optimal_A
            weight = None
            if time_varying_lambdas:
                gamma_t = lmbda[t].T.dot(ct.T).T
                gamma_t = gamma_func(lmbda[t].T, gamma_t, present, ct, 
                                        idxs=to_impute, reg=reg,
                                       mu=mean,
                                    resid_regs=resid_regs)
            else:

                gamma_t = lmbda.T.dot(ct.T).T
                gamma_t = gamma_func(lmbda.T, gamma_t, present, ct, idxs=to_impute, reg=reg,
                                        mu=mean,
                                    resid_regs=resid_regs)
            return gamma_t
        
        if run_in_parallel:
            gammas = [x for x in Parallel(n_jobs=30, verbose=0)(delayed(get_gamma_t)(
                ct = char_panel[t], 
                present = ~np.isnan(char_panel[t]),
                to_impute = np.argwhere(return_mask[t]).squeeze(),
                lmbda=lmbda,
                time_varying_lambdas=time_varying_lambdas, t=t,
                mean=mu[t],
                resid_regs = resid_regs[t] if resid_reg and resid_regs is not None else None
            ) for t in range(T))]
        else:
            gammas = [get_gamma_t(
                ct = char_panel[t], 
                present = ~np.isnan(char_panel[t]),
                to_impute = np.argwhere(return_mask[t]).squeeze(),
                lmbda=lmbda,
                time_varying_lambdas=time_varying_lambdas, t=t,
                mean=mu[t],
                resid_regs = resid_regs[t] if resid_reg and resid_regs is not None else None
            ) for t in range(T)]

        for t in range(T):
            gamma_ts[t, return_mask[t]] = gammas[t][return_mask[t]]
#             gamma_ts[t, ~return_mask[t]] = np.nan        

        use_cond_first_time = False
        
        
        if time_varying_lambdas:
            new_imputation = np.concatenate([np.expand_dims(x @ l.T + m, axis=0) for x,l, m in zip(gamma_ts, lmbda, mu)], axis=0)
        else:
            new_imputation = np.concatenate([np.expand_dims(x @ lmbda.T + m, axis=0) for x,m in zip(gamma_ts, mu)], axis=0)
            
        resids = char_panel - new_imputation
        if resid_reg:
            resid_regs = np.square(np.nanstd(resids, axis=1))

        imputed_chars[missing_mask_overall] = new_imputation[missing_mask_overall]
        
        if allow_mean:
            mu = np.nanmean(imputed_chars, axis=1)
            
        if not allow_mean:
            assert np.all(mu == 0)
            
    return gamma_ts, lmbda


def estimate_lambda(char_panel, return_panel, num_months_train, K, min_chars,
                    window_size=1, time_varying_lambdas=False,
                   mu=None, eval_weight_lmbda=True,
                   shrink_lmbda=False, reg=0):
    """
    Parameters
    ----------
        char_panel : input TxNxC panel of characteristics
        return_panel : input TxN panel of returns
        num_months_train : number of months to fit lambda over
        K : number of factors
        gamma_0, gamma_1, gamma_2: penalty weights of model losses
        phi: basis function for the characteristics
        intercept: whether or not to include an intercept / "market" characteristic
        hardcoded_mask: Optional TxN mask to exclude stocks from computation
        weighting: Optional TxN stock-wise weighting mask
        t_lag_maps: Optional, mapping from time index to lags to include
    """
    chars = np.array(['A2ME', 'AC', 'AT', 'ATO', 'B2M', 'BETA_d', 'BETA_m', 'C2A',
       'CF2B', 'CF2P', 'CTO', 'D2A', 'D2P', 'DPI2A', 'E2P', 'FC2Y',
       'HIGH52', 'INV', 'IdioVol', 'LEV', 'ME', 'NI', 'NOA', 'OA', 'OL',
       'OP', 'PCM', 'PM', 'PROF', 'Q', 'R12_2', 'R12_7', 'R2_1', 'R36_13',
       'R60_13', 'RNA', 'ROA', 'ROE', 'RVAR', 'S2P', 'SGA2S', 'SPREAD',
       'SUV', 'TURN', 'VAR'])
    monthly_chars = ['BETA_d', 'BETA_m', 'D2P', 'IdioVol', 'ME', 'TURN',
                     'R2_1', 'R12_2', 'R12_7', 'R36_13', 'R60_13', 'HIGH52', 'RVAR', 'SPREAD',  'SUV',  'VAR']


    min_char_mask = np.expand_dims(np.logical_and(np.sum(~np.isnan(char_panel), axis=2) >= min_chars,
                                                  ~np.isnan(return_panel)), axis=2)


    cov_mats = []
    first_warn = True

    for t in range(num_months_train):
        cov_mats.append(get_cov_mat(char_panel[t], None if mu is None else mu[t]))
        
    cov_mats_sum = sum(cov_mats) * (1 / len(cov_mats))

    if time_varying_lambdas:
        lmbda = []
        cov_mat = []

        for t in range(len(cov_mats)):
            cov_mats_sum = sum(cov_mats[max(0, t-window_size+1):t+1]) * (1 / (window_size))
            eig_vals, eig_vects = np.linalg.eigh(cov_mats_sum)
            idx = np.abs(eig_vals).argsort()[::-1]
            if eval_weight_lmbda:
                if shrink_lmbda:
                    lmbda.append(eig_vects[:, idx[:K]] * 
                                 np.maximum(np.sqrt(np.sqrt(np.maximum(eig_vals[idx[:K]].reshape(1, -1), 0))) - reg, 0))
                else:
                    lmbda.append(eig_vects[:, idx[:K]] * np.sqrt(np.maximum(eig_vals[idx[:K]].reshape(1, -1), 0)))
                
            else:
                lmbda.append(eig_vects[:, idx[:K]])
            assert np.all(~np.isnan(lmbda[-1])), lmbda
            cov_mat.append(cov_mats_sum)
    else:
        tgt_mat = cov_mats_sum


        eig_vals, eig_vects = np.linalg.eigh(tgt_mat)

        idx = np.abs(eig_vals).argsort()[::-1]
        if eval_weight_lmbda:
            if shrink_lmbda:
                lmbda = eig_vects[:, idx[:K]] * np.maximum(np.sqrt(np.sqrt(eig_vals[idx[:K]].reshape(1, -1))) - reg, 0)
            else:
                lmbda = eig_vects[:, idx[:K]] * np.sqrt(np.maximum(0, eig_vals[idx[:K]].reshape(1, -1)))
        else:
            lmbda = eig_vects[:, idx[:K]]
        cov_mat = tgt_mat

    return lmbda, cov_mat


def get_cov_mat(char_matrix, mu=None):
    '''
    utility method to get the covariance matrix of a partially observed set of chatacteristics
    '''
    ct_int = (~np.isnan(char_matrix)).astype(float)
    ct = np.nan_to_num(char_matrix)
    if mu is None:
        mu = np.nanmean(char_matrix, axis=0).reshape(-1, 1)
    temp = ct.T.dot(ct) 
    temp_counts = ct_int.T.dot(ct_int)
    sigma_t = temp / temp_counts - mu @ mu.T
    return sigma_t



def get_sufficient_statistics_xs(gamma_ts, characteristics_panel):
    return gamma_ts, None


def get_sufficient_statistics_last_val(characteristics_panel, max_delta=None,
                                      residuals=None):
    '''
    utility method to get the last observed value of a characteristic if it has been previously observed
    '''
    T, N, L = characteristics_panel.shape
    last_val = np.copy(characteristics_panel[0])
    if residuals is not None:
        last_resid = np.copy(residuals[0])
    lag_amount = np.zeros_like(last_val)
    lag_amount[:] = np.nan
    if residuals is None:
        sufficient_statistics = np.zeros((T,N,L, 1), dtype=float)
    else:
        sufficient_statistics = np.zeros((T,N,L, 2), dtype=float)
    sufficient_statistics[:,:,:,:] = np.nan
    deltas = np.copy(sufficient_statistics[:,:,:,0])
    for t in range(1, T):
        lag_amount += 1
        sufficient_statistics[t, :, :, 0] = np.copy(last_val)
        deltas[t] = np.copy(lag_amount)
        present_t = ~np.isnan(characteristics_panel[t])
        last_val[present_t] = np.copy(characteristics_panel[t, present_t])
        if residuals is not None:
            sufficient_statistics[t, :, :, 1] = np.copy(last_resid)
            last_resid[present_t] = np.copy(residuals[t, present_t])
        lag_amount[present_t] = 0
        if max_delta is not None:
            last_val[lag_amount >= max_delta] = np.nan
        
    return sufficient_statistics, deltas

def get_sufficient_statistics_next_val(characteristics_panel, max_delta=None, residuals=None):
    '''
    utility method to get the next observed value of a characteristic if it is observed in the future
    '''
    suff_stats, deltas = get_sufficient_statistics_last_val(characteristics_panel[::-1], max_delta=max_delta,
                                      residuals=None if residuals is None else residuals[::-1])
    return suff_stats[::-1], deltas[::-1]


def impute_chars(char_data, imputed_chars, residuals, 
                 suff_stat_method, constant_beta, beta_weight=True, noise=None):
    '''
    run imputation as described in the paper, based on the type of sufficient statistics
    - last-val B-XS
    - next-val F-XS
    - fwbw BF-XS
    '''
    if suff_stat_method == 'last_val':
        suff_stats, _ = get_sufficient_statistics_last_val(char_data, max_delta=None,
                                                                           residuals=residuals)
        if len(suff_stats.shape) == 3:
            suff_stats = np.expand_dims(suff_stats, axis=3)
        beta_weights = None
        
    elif suff_stat_method == 'next_val':
        suff_stats, _ = get_sufficient_statistics_next_val(char_data, max_delta=None, residuals=residuals)
        beta_weights = None
        
    elif suff_stat_method == 'fwbw':
        next_val_suff_stats, fw_deltas = get_sufficient_statistics_next_val(char_data, max_delta=None,
                                                                                            residuals=residuals)
        prev_val_suff_stats, bw_deltas = get_sufficient_statistics_last_val(char_data, max_delta=None,
                                                                                            residuals=residuals)
        suff_stats = np.concatenate([prev_val_suff_stats, next_val_suff_stats], axis=3)
        
        if beta_weight:            
            beta_weight_arr = np.concatenate([np.expand_dims(fw_deltas, axis=3), 
                                                  np.expand_dims(bw_deltas, axis=3)], axis=3)
            beta_weight_arr = 2 * beta_weight_arr / np.sum(beta_weight_arr, axis=3, keepdims=True)
            beta_weights = {}
            one_arr = np.ones((1, 1))
            for t, i, j in np.argwhere(np.logical_and(~np.isnan(fw_deltas), ~np.isnan(bw_deltas))):
                beta_weights[(t,i,j)] = np.concatenate([one_arr, beta_weight_arr[t,i,j].reshape(-1, 1)], axis=0).squeeze()
        else:
            beta_weights = None
        
    elif suff_stat_method == 'None':
        suff_stats = None
        beta_weights = None
            
    if suff_stats is None:
        return imputed_chars
    else:
        return impute_beta_combined_regression(
            char_data, imputed_chars, sufficient_statistics=suff_stats, 
            beta_weights=None, constant_beta=constant_beta,
            noise=noise
        )


def impute_beta_combined_regression(characteristics_panel, xs_imps, sufficient_statistics=None, 
                           beta_weights=None, constant_beta=False, get_betas=False, gamma_ts=None,
                                   use_factors=False, noise=None, reg=None, switch_off_on_suff_stats=False):
    '''
    Run the regression to fit the parameters of the regression model combining time series with cross-sectional information
    '''
    T, N, L = characteristics_panel.shape
    K = 0
    if xs_imps is not None:
        K += 1
    if sufficient_statistics is not None:
        K += sufficient_statistics.shape[3]
    if use_factors:
        K += gamma_ts.shape[-1]
    betas = np.zeros((T, L, K))
    imputed_data = np.copy(characteristics_panel)
    imputed_data[:,:,:]=np.nan
    
    if reg is not None and not switch_off_on_suff_stats and use_factors:
        gamma_ts = gamma_ts * np.sqrt(45)
    
    for l in range(L):
        fit_suff_stats = []
        fit_tgts = []
        inds = []
        curr_ind = 0
        all_suff_stats = []
        
        for t in range(T):
            inds.append(curr_ind)
            
            if xs_imps is not None:
                if noise is not None:
                    suff_stat = np.concatenate([xs_imps[t,:,l:l+1] + noise[t,:,l:l+1], sufficient_statistics[t,:,l]], axis=1)
                else:
                    suff_stat = np.concatenate([xs_imps[t,:,l:l+1], sufficient_statistics[t,:,l]], axis=1)
                
            else:
                if use_factors:
                    suff_stat = np.concatenate([gamma_ts[t,:,l,:], sufficient_statistics[t,:,l]], axis=1)
                else:
                    suff_stat = sufficient_statistics[t,:,l]
            
            available_for_imputation = np.all(~np.isnan(suff_stat), axis=1)
            available_for_fit = np.logical_and(~np.isnan(characteristics_panel[t,:,l]),
                                                  available_for_imputation)
            all_suff_stats.append(suff_stat)

            fit_suff_stats.append(suff_stat[available_for_fit, :])
            fit_tgts.append(characteristics_panel[t,available_for_fit,l])
            curr_ind += np.sum(available_for_fit)
        
        
        inds.append(curr_ind)
        fit_suff_stats = np.concatenate(fit_suff_stats, axis=0)
        fit_tgts = np.concatenate(fit_tgts, axis=0)
        
        if constant_beta:
            if reg is None:
                beta = np.linalg.lstsq(fit_suff_stats, fit_tgts, rcond=None)[0]
            else:
                X = fit_suff_stats
                lmbda = np.eye(fit_suff_stats.shape[1]) * reg * fit_suff_stats.shape[0]
                if switch_off_on_suff_stats:
                    skip_reg_num = 0 if sufficient_statistics is None else sufficient_statistics.shape[-1]
                    for i in range(1, skip_reg_num+1):
                        lmbda[-i, -i] = 0

                
                beta = np.linalg.lstsq(X.T@ X + lmbda, X.T@fit_tgts, rcond=None)[0]
                
            betas[:,l,:] = beta.reshape(1, -1)
        else:
            for t in range(T):
                
                
                if reg is None:
                    beta_l_t = np.linalg.lstsq(fit_suff_stats[inds[t]:inds[t+1]],
                                           fit_tgts[inds[t]:inds[t+1]], rcond=None)[0]
                else:
                    X = fit_suff_stats[inds[t]:inds[t+1]]
                    y = fit_tgts[inds[t]:inds[t+1]]
                    lmbda = np.eye(X.shape[1]) * reg * X.shape[0]
                    
                    if switch_off_on_suff_stats:
                        skip_reg_num = 0 if sufficient_statistics is None else sufficient_statistics.shape[-1]
                        for i in range(1, skip_reg_num+1):
                            lmbda[-i, -i] = 0
                    
                    beta_l_t = np.linalg.lstsq(X.T@X + lmbda, 
                                           X.T@y, rcond=None)[0]
                betas[t,l,:] = beta_l_t
                
        for t in range(T):
            beta_l_t = betas[t,l]
            suff_stat = all_suff_stats[t]
            
            available_for_imputation = np.all(~np.isnan(suff_stat), axis=1)
            
            if beta_weights is None:
                imputed_data[t,available_for_imputation,l] = suff_stat[available_for_imputation,:] @ beta_l_t
            else:
                for i in np.argwhere(available_for_imputation).squeeze():
                    if (t,i,l) in beta_weights:
                        assert np.all(~np.isnan(beta_weights[(t,i,l)]))
                        imputed_data[t,i,l] = suff_stat[i,:] @ np.diag(beta_weights[(t,i,l)]) @ betas[l]
                    else:
                        imputed_data[t,i,l] = suff_stat[i,:] @ betas[l]
    if get_betas:
        return imputed_data, betas
    else:
        return imputed_data

def simple_imputation(gamma_ts, char_data, suff_stat_method, monthly_update_mask, char_groupings,
                                 eval_char_data=None, num_months_train=None, median_imputation=False,
                                 industry_median=False, industries=None):
    '''
    utility method to do either previous value, median or industry median imputation
    '''
    if eval_char_data is None:
        eval_char_data = char_data
    imputed_chars = simple_impute(char_data)
    if median_imputation:
        imputed_chars[:,:,:] = 0
    elif industry_median:
        imputed_chars = xs_industry_median_impute(char_panel=char_data, industry_codes=industries)
        
    return imputed_chars

def simple_impute(char_panel):
    """
    imputes using the last value of the characteristic time series
    """
    imputed_panel = np.copy(char_panel)
    imputed_panel[:,:,:] = np.nan
    imputed_panel[0] = np.copy(char_panel[0])
    for t in range(1, imputed_panel.shape[0]):
        present_t_l = ~np.isnan(char_panel[t-1])
        imputed_t_1 = ~np.isnan(imputed_panel[t-1])
        imputed_panel[t, present_t_l] = char_panel[t-1, present_t_l]
        imputed_panel[t, np.logical_and(~present_t_l, 
                                     imputed_t_1)] = imputed_panel[t-1, 
                                                                   np.logical_and(~present_t_l, imputed_t_1)]
        imputed_panel[t, ~np.logical_or(imputed_t_1, present_t_l)] = np.nan
        
    return imputed_panel

def xs_industry_median_impute(char_panel, industry_codes):
    """
    imputes using the last value of the characteristic time series
    """
    imputed_panel = np.copy(char_panel)
    for t in range(imputed_panel.shape[0]):
        for c in range(imputed_panel.shape[2]):
            for x in np.unique(industry_codes):
                industry_filter = industry_codes==x
                present_t_l_i = np.logical_and(~np.isnan(char_panel[t,:, c]), industry_filter)
                imputed_panel[t, industry_filter, c] = np.median(char_panel[t,present_t_l_i, c])        
    return imputed_panel

def get_all_xs_vals(chars, reg, Lmbda, time_varying_lmbda=False, get_factors=False):
    '''
    utility method to get the "out of sample estimate" of an observed characteristic based on the XP method
    '''
    C = chars.shape[-1]
    def impute_t(t_chars, reg, C, Lmbda, get_factors=False):
        if not get_factors:
            imputation = np.copy(t_chars) * np.nan
        else:
            imputation = np.zeros((t_chars.shape[0], t_chars.shape[1], Lmbda.shape[1])) * np.nan
        mask = ~np.isnan(t_chars)
        net_mask = np.sum(mask, axis=1)
        K = Lmbda.shape[1]
        for n in range(t_chars.shape[0]):
            if net_mask[n] == 1:
                imputation[n,:] = 0
            elif net_mask[n] > 1:
                for i in range(C):
                    tmp = mask[n, i]
                    mask[n,i] = False
                    y = t_chars[n, mask[n]]
                    X = Lmbda[mask[n], :]
                    L = np.eye(K) * reg
                    params = np.linalg.lstsq(X.T @ X + L, X.T @ y, rcond=None)[0]
                    if get_factors:
                        imputation[n,i] = params
                    else:
                        imputation[n,i] = Lmbda[i] @ params
                    
                    mask[n,i] = tmp
        return np.expand_dims(imputation, axis=0)
    chars = [chars_t for chars_t in chars]
    
    if time_varying_lmbda:
        imputation = list(Parallel(n_jobs=60)(delayed(impute_t)(chars_t, reg, C, l, get_factors=get_factors) 
                                              for chars_t, l in zip(chars, Lmbda)))
    else:
        imputation = list(Parallel(n_jobs=60)(delayed(impute_t)(chars_t, reg, C, Lmbda, get_factors=get_factors)
                                              for chars_t in chars))
    return np.concatenate(imputation, axis=0)