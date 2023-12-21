from time import time
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from unicodedata import name
from python import missing_data_utils, imputation_utils
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm 
from joblib import Parallel, delayed
from collections import defaultdict
import numpy as np
from itertools import chain, combinations
from py_imports import *
from python.models import FactorModel, PCAFactorModel, IpcaModelV2
from matplotlib.lines import Line2D

char_groupings = {
    "Past Returns" : ['R2_1', 'R12_2', 'R12_7', 'R36_13', 'R60_13'],
    "Investment": ['INV', 'NOA', 'DPI2A', 'NI'],
    "Profitability": ['PROF', 'ATO', 'CTO', 'FC2Y', 'OP', 'PM', 'RNA', 'ROA', 'ROE', 'SGA2S', 
                     'D2A'],
    "Intangibles": ['AC', 'OA', 'OL', 'PCM'],
    "Value": ['A2ME', 'B2M',  'C2A', 'CF2B', 'CF2P', 'D2P', 'E2P', 'Q',  'S2P', 'LEV'],
    "Trading Frictions": ['AT', 'BETA_d', 'IdioVol', 'ME', 'TURN', 'BETA_m',
             'HIGH52',  'RVAR', 'SPREAD', 'SUV', 'VAR']
}
base_colors = ['r', 'b', 'g', 'y', 'c', 'orange']
def plot_factor(lmbda, index, chars, tag):
    fig, ax = plt.subplots(figsize=(15,10))
    bars = []
    bar_locations = [val for val in[3*x for x in range(45)]]
    tick_locations = [3*x for x in range(45)]

    ticks = []
    colors = []
    k = 0

    groups = []
    group_colors = []

    bar_maxs = []

    for i, grouping in enumerate(char_groupings):
        groups.append(grouping)
        group_colors += [base_colors[i]]
        for char in char_groupings[grouping]:
            ticks.append(char)
            colors += [base_colors[i]]

            char_ind = np.argwhere(chars==char)[0]

            bars += [lmbda[char_ind, index][0]]
            bar_maxs.append(abs(lmbda[char_ind, index][0]))
#     bar_locations.append(46*3)
#     tick_locations.append(46*3)
#     ticks.append("intercept")
#     colors.append("black")
#     bars.append(lmbda[-1, index])
#     bar_maxs.append(np.abs(lmbda[-1, index]))
#     groups.append("Intercept")
#     group_colors.append("black")

    plt.bar(bar_locations, bars, color=colors)
    plt.xticks(ticks=tick_locations, labels=ticks)
    plt.xticks(rotation=90)
    fig.patch.set_facecolor('white')
#     fig.autofmt_xdate()

    for i in np.where((np.max(bar_maxs) - bar_maxs) / np.max(bar_maxs) < 0.5 )[0]:
        ax.get_xticklabels()[i].set_color("green")

    for i in np.where((np.max(bar_maxs) - bar_maxs) / np.max(bar_maxs) < 0.1 )[0]:
        ax.get_xticklabels()[i].set_color("red")

    custom_lines = [Line2D([0], [0], color=group_colors[i], lw=4) for i in range(len(groups))]

    ax.legend(custom_lines, groups)
    save_path = f'../images-pdfs/revision/'
    plt.savefig(save_path + f'factor_vis_{index}_{tag}.pdf', bbox_inches='tight')
    plt.show()
    


def get_sparse_lmbda(lmbda, gammas, gamma_tildes, L,K,reg,group_masks):
    start = time()
    print(L,K,reg)
    lmbda_tilde = cp.Variable((L,K))
    cost = 0
    scale = 0
    for gamma_t, gamma_tilde_t in zip(gammas, gamma_tildes):
        tgt = gamma_t @ lmbda.T
        scale += np.linalg.norm(tgt)
        cost += cp.sum_squares(tgt - gamma_tilde_t @ lmbda_tilde.T)
    for k in range(K):
        for mask in group_masks:
            cost += reg * cp.norm(lmbda_tilde[mask, k], p=2) * scale / np.sum(mask)
    print("constructed problem")
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver='GUROBI', verbose=False)
    print("solved problem", time() - start)
    return lmbda_tilde.value


def sparsify_lmbda(lmbda, gamma_ts, group_masks, maxiter=10, tol=1e-3, reg=1):
    '''
    get a sparse approximation of lmbda / gamma 
    '''
    delta = 1
    lmbda_tilde = np.copy(lmbda)
    lmbda_norms = np.linalg.norm(lmbda_tilde, axis=0, keepdims=True)
    print("lmbda_norms", lmbda_norms)
    lmbda_tilde = lmbda_tilde / lmbda_norms
    initial_lmbda_tilde = np.copy(lmbda_tilde)
    
    for gamma in gamma_ts:
        assert np.all(~np.isnan(gamma))
    assert np.all(~np.isnan(lmbda))
    
    gamma_tildes = [np.copy(x) * lmbda_norms for x in gamma_ts]
    i = 0
    while delta > tol:
        i += 1
        print("getting sparse lmbda")
        lmbda_tilde_new = get_sparse_lmbda(lmbda, gamma_ts, gamma_tildes, L=lmbda.shape[0],
                                           K=lmbda.shape[1],
                                           reg=reg,
                                           group_masks=group_masks)
        delta = np.max(np.abs(lmbda_tilde - lmbda_tilde_new))
        lmbda_tilde = lmbda_tilde_new
        
        print("before regression", [np.linalg.norm(gamma_t @ lmbda.T - gamma_tilde_t @ lmbda_tilde.T) 
                                    for  gamma_t, gamma_tilde_t in zip(gamma_ts, gamma_tildes)
                                   ])
        gamma_tildes_new = [
            np.linalg.lstsq(lmbda_tilde, lmbda@gamma_t.T)[0].T for gamma_t in gamma_ts
        ]
        
        print([(gamma_t.shape, gamma_tilde_t.shape)  for  gamma_t, gamma_tilde_t in zip(gamma_ts, gamma_tildes_new)
                                   ])
              
        print("after regression", [np.linalg.norm(gamma_t @ lmbda.T - gamma_tilde_t @ lmbda_tilde.T) 
                                    for  gamma_t, gamma_tilde_t in zip(gamma_ts, gamma_tildes_new)
                                   ])
              
        delta = max(np.max([np.max(np.abs(g - g_new)) for g, g_new in zip(gamma_tildes, gamma_tildes_new)]),
                    delta)
        gamma_tildes = gamma_tildes_new
        
        lmbda_norms = np.linalg.norm(lmbda_tilde, axis=0, keepdims=True)
        lmbda_tilde = lmbda_tilde / lmbda_norms
        gamma_tildes = [x * lmbda_norms for x in gamma_tildes]
        
        if i >= maxiter:
            break
            
        print(f"iteration {i} delta was {delta}, lmbda net delta was {np.max(np.abs(initial_lmbda_tilde - lmbda_tilde))}")
            
    return lmbda_tilde, gamma_tildes

########### alternative models

def impute_given_mu_sigma(data_t, mu_t, sigma_t, min_chars):
    imputed_data = np.copy(data_t)
    for i in range(data_t.shape[0]):
        i_data = data_t[i]
        i_mask = ~np.isnan(i_data)
        if np.sum(i_mask) > min_chars:
            conditional_mean, conditional_var = conditional_mean_and_var(sigma_t, mu_t, i_mask, 
                                                                         i_data.reshape(-1, 1))
            assert np.all(~np.isnan(conditional_mean))
            imputed_data[i,~i_mask] = conditional_mean.squeeze()
    
    return imputed_data

def impute_andrew_chen(missing_data_type, min_chars, maxiter, percentile_rank_chars):
    
    fit_maps = {
        'MAR': "MAR_fit_data",
        'BLOCK': "prob_block_fit_data",
        'logit': "logit_fit_data"
    }
    
    if missing_data_type in fit_maps:
        masked_lagged_chars = imputation_utils.load_imputation(fit_maps[missing_data_type])
    else:
        masked_lagged_chars = percentile_rank_chars

    def impute_t(data_t):
        mu, sigma = em(data_t, eps=1e-4, min_chars=min_chars, max_iter=maxiter)
        return np.expand_dims(impute_given_mu_sigma(data_t, mu, sigma, min_chars), axis=0), sigma
    
    res = [x for x in Parallel(n_jobs=15, verbose=5)(delayed(impute_t)(masked_lagged_chars[t]) 
                                                                 for t in range(masked_lagged_chars.shape[0]))]
    imputed_data = np.concatenate([x[0] for x in res], axis=0)
        
    return imputed_data, [x[1] for x in res]


def conditional_mean_and_var(Sigma, mu, i_mask, i_data):
    Sigma_11 = Sigma[~i_mask, :][:, ~i_mask]
    Sigma_12 = Sigma[~i_mask, :][:, i_mask]
    Sigma_22 = Sigma[i_mask, :][:, i_mask]
    mu1, mu2 = mu[~i_mask], mu[i_mask]

    conditional_var =  Sigma_11 - Sigma_12 @ np.linalg.inv(Sigma_22) @ Sigma_12.T
    assert np.all(~np.isnan(conditional_var))

    
    conditional_mean = mu1 + Sigma_12 @ np.linalg.inv(Sigma_22) @ (i_data[i_mask] - mu2)
#     print(np.linalg.cond(Sigma_22), conditional_mean)
    assert np.all(~np.isnan(conditional_mean))
    
    return conditional_mean, conditional_var
    


def em(data, eps=0.1, min_chars=10, max_iter=100, log=False):    
    ret_mat = np.copy(data)
    observed_mask = ~np.isnan(ret_mat)
    in_sample = np.sum(observed_mask, axis=1) >= min_chars

    sample_data = ret_mat[in_sample]
    sample_mask = ~np.isnan(sample_data)
    N, C = sample_data.shape
    Sigma = np.eye(C)
    mu = np.zeros((C, 1))
    
    j = 0
    S_hats = np.zeros((N, C, C))
    e_hats = np.zeros((N, C))
    while j < max_iter:
        for i in range(N):
            i_mask = sample_mask[i]
            i_data = sample_data[i:i+1].T
            conditional_mean, conditional_var = conditional_mean_and_var(Sigma, mu, i_mask, i_data)
            
            m1 = i_mask.reshape(-1, 1) @ i_mask.reshape(1, -1)
            np.place(S_hats[i], m1, i_data[i_mask].reshape(-1, 1) @ i_data[i_mask].reshape(1,-1))
            
            m2 = i_mask.reshape(-1, 1) @ ~i_mask.reshape(1, -1)
            np.place(S_hats[i], m2, i_data[i_mask] @ conditional_mean.T)
            np.place(S_hats[i], m2.T, conditional_mean @ i_data[i_mask].T)
            m3 = ~i_mask.reshape(-1, 1) @ ~i_mask.reshape(1, -1)
            np.place(S_hats[i], m3, conditional_var + conditional_mean @ conditional_mean.T)
#             np.place(S_hats[i], m3, conditional_mean @ conditional_mean.T)
            
            e_hats[i:i+1,i_mask] = i_data[i_mask].T
            e_hats[i:i+1,~i_mask] = conditional_mean.T
        
        
        mu_new = np.mean(e_hats, axis=0).reshape(-1, 1)
        Sigma_new = np.mean(S_hats, axis=0) - mu_new @ mu_new.T
        delta_m = np.max(np.abs(mu_new - mu))
        delta_s = np.max(np.abs(Sigma_new - Sigma))
        
        if log: 
            log_like = -1*np.log(np.linalg.det(Sigma_new)) - \
                             np.trace(np.linalg.solve(Sigma_new,
                                                              np.mean([S_hats[i] 
                                                               - mu_new.reshape(-1, 1) @ e_hats[i].reshape(1, -1)
                                                              - e_hats[i].reshape(-1, 1) @ mu_new.reshape(1, -1)
                                                              - mu_new.reshape(-1, 1) @ mu_new.reshape(1, -1)
                                                            for i in range(N)], axis=0)))
            print(f"iteration {j} delta_m was {delta_m} delta_s was {delta_s} log like was {log_like}")
        mu = mu_new
        Sigma = Sigma_new
        if max(delta_s, delta_m) < eps:
            break
        j += 1
    
    return mu, Sigma


def impute_chars_freyweb(chars, regr_chars, missing_data_type, percentile_rank_chars):
    
    fit_maps = {
        'MAR': "MAR_fit_data",
        'BLOCK': "prob_block_fit_data",
        'logit': "logit_fit_data"
    }
    
    if missing_data_type in fit_maps:
        masked_lagged_chars = imputation_utils.load_imputation(fit_maps[missing_data_type])
    else:
        masked_lagged_chars = percentile_rank_chars
    
    T = masked_lagged_chars.shape[0]
    
    char_mask = np.isin(chars, regr_chars)
    gamma_ts = masked_lagged_chars[:,:,char_mask]
    mask = np.all(~np.isnan(gamma_ts), axis=2)
    gamma_ts[~mask] = np.nan
    imputed_chars = imputation_utils.impute_chars(gamma_ts, masked_lagged_chars, 
                                             "None", None, char_groupings, num_months_train=T,
                                                            eval_char_data=None,
                                                               window_size=1)
    
    return imputed_chars


