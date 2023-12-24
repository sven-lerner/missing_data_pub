import data_loading
import imputation_utils, imputation_model_simplified
import importlib
import numpy as np
import pandas as pd

columns = ["EM", "EM", "XS", "XS", "XS>K", "XS>K"]


def get_factor_model_realization(N, L, K, snr=0.1, factor_scales=None, lmbda_orth=True):
    loadings = np.random.normal(size=(L, K))
    if lmbda_orth:
        loadings, _ = np.linalg.qr(loadings)
    factors = np.random.normal(size=(N, K))
    if factor_scales is not None:
        factors = factors @ np.diag(factor_scales) 
    noise = np.random.normal(size=(N, L), scale=snr)
    data = noise + factors @ loadings.T
    return loadings, factors, data, noise
def standard_missing_mask(N,L,percentage):
    samples= np.random.uniform(low=0.0, high=1.0, size=(N,L))
    return samples < percentage
def factor_dependant_mask(N, L, factors, percentage):
    pi = 1 - 1 / (1 + np.exp(np.linalg.norm(factors, axis=1)))
    pi /= (np.mean(pi) / percentage)
    mask = np.concatenate([np.random.binomial(1, pi).reshape(-1, 1) for _ in range(L)], axis=1) == 1
    return mask

def loading_dependant_mask(N, L, loadings, percentage):
    pi = 1 - 1 / (1 + np.exp(np.linalg.norm(loadings, axis=1)))
    pi /= (np.mean(pi) / percentage)
    mask = np.concatenate([np.random.binomial(1, pi).reshape(1, -1) for _ in range(N)], axis=0) == 1
    print(percentage, np.sum(mask) / (N * L))
    return mask


from joblib import Parallel, delayed
def run_scheme(numiter, N,K,L,perc,missingtype,num_factors_impute,
               min_chars=10,snr=0.1,factor_scales=None, impute_reg=0.01,
              lmbda_orth=True, estimate_reg=False):
    
    def run_inner(N,K,L,perc,missingtype,num_factors_impute,
               min_chars=min_chars,snr=snr,factor_scales=factor_scales,
                  impute_reg=impute_reg,
              lmbda_orth=lmbda_orth, estimate_reg=estimate_reg):
        Lmbda, Ft, X, epsilon = get_factor_model_realization(N,L,K, snr=snr, factor_scales=factor_scales, 
                                                             lmbda_orth=lmbda_orth)
        standard_mask = standard_missing_mask(N,L,perc)
        X_standard_masked = np.copy(X)
        X_standard_masked[standard_mask] = np.nan
        ft_mask = factor_dependant_mask(N, L, Ft, perc)
        X_ft_masked = np.copy(X)
        X_ft_masked[ft_mask] = np.nan

        if missingtype == 'MAR':
            X_masked = X_standard_masked
            mask = standard_mask
        elif missingtype == 'FT':
            X_masked = X_ft_masked
            mask = ft_mask
        elif missingtype == 'threshold':
            cutoff = np.quantile(np.abs(X), 1 - perc)
            X_masked = np.copy(X)
            mask = np.abs(X) > cutoff
            X_masked[mask] = np.nan
        elif missingtype == 'Lmbda':
            X_masked = np.copy(X)
            mask = loading_dependant_mask(N, L, Lmbda, perc)
            X_masked[mask] = np.nan

        if factor_scales is None:
            Sigma = Lmbda@ Lmbda.T + np.square(snr)*np.eye(L)
        else:
            Sigma = Lmbda@ np.diag(np.square(factor_scales)) @ Lmbda.T + np.square(snr)*np.eye(L)

        optimal_impute = impute_given_mu_sigma(X_masked, np.zeros((L,1)), Sigma, min_chars)

        em_imp = impute_em(X_masked, min_chars=10, maxiter=20)


        xp_K, std_gamma_ts, std_lmbda = impute_xs_simplified(X_masked, 
                                                               use_cond_exp=False, niter=3,
                                                               use_cond_first_time=False,
                                                               cov_weight_regr=False, 
                                                               eval_data=None, reg=impute_reg, 
                                                               nfactors=K, 
                                                               prespec_cov_mats=None,
                                                              eval_weight_lmbda=True,
                                                            estimate_reg=estimate_reg,
                                                            KMax=None)


        xp_nfactor, std_gamma_ts, std_lmbda = impute_xs_simplified(X_masked, 
                                                               use_cond_exp=False, niter=4,
                                                               use_cond_first_time=False,
                                                               cov_weight_regr=False, 
                                                               eval_data=None, reg=impute_reg, 
                                                               nfactors=None, 
                                                               prespec_cov_mats=None,
                                                              eval_weight_lmbda=True,
                                                            estimate_reg=estimate_reg,
                                                            KMax=num_factors_impute)

        return (
            np.nanmean(np.square(em_imp[0][0][mask] - X[mask])), 
            np.nanmean(np.square(em_imp[0][0][mask] - optimal_impute[mask])),
            np.nanmean(np.square(xp_K[0][mask] - X[mask])),
            np.nanmean(np.square(xp_K[0][mask] - optimal_impute[mask])),
            np.nanmean(np.square(xp_nfactor[0][mask] - X[mask])),
            np.nanmean(np.square(xp_nfactor[0][mask] - optimal_impute[mask]))
        )
    
    
    
    results = list(Parallel(n_jobs=4)(delayed(run_inner)(N,K,L,perc,missingtype,num_factors_impute,
               min_chars=min_chars,snr=snr,factor_scales=factor_scales,
                  impute_reg=impute_reg,
              lmbda_orth=lmbda_orth, estimate_reg=estimate_reg) for _ in range(numiter)))
    return results
        
    
def impute_xs_simplified(data, use_cond_exp, niter=1, use_cond_first_time=False,
                        cov_weight_regr=False, eval_data=None, reg=0, nfactors=6, prespec_cov_mats=None,
                        allow_mean=True, 
                        tv_lambdas=False, eval_weight_lmbda=True, estimate_reg=False,
                        KMax=None):
    
    data = np.expand_dims(data, axis=0)
    
    if nfactors is None:
        mask = standard_missing_mask(data.shape[1],data.shape[2],percentage=0.1)
#         print("number masked", np.sum(mask * np.isnan(data)))
        masked_data = np.copy(data)
        masked_data[:,mask] = np.nan
        kvals = np.arange(1, KMax + 1)
        results = []
        for k in kvals:
            gamma_ts, lmbdas = imputation_model_simplified.impute_panel_xp_lp(
                    char_panel=masked_data, 
                            return_panel= np.ones(data.shape[0]), min_chars=10,
                              K=k, 
                            num_months_train=data.shape[0],
                    reg=reg,
                    time_varying_lambdas=True,
                    window_size=1, 
                    n_iter=niter,
                    eval_data=None,
                    allow_mean=False,
            eval_weight_lmbda=eval_weight_lmbda,
                        resid_reg=estimate_reg,
            run_in_parallel=False)
            recon = np.concatenate([np.expand_dims(x @ l.T, axis=0) for x,l in zip(gamma_ts, lmbdas)], axis=0)
            results.append(np.nanmean(np.square(recon[:,mask] - data[:,mask])))
        nfactors = kvals[np.argmin(results)]
        
    gamma_ts, lmbdas = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=data, 
                        return_panel= np.ones(data.shape[0]), min_chars=10,
                          K=nfactors, 
                        num_months_train=data.shape[0],
                reg=reg,
                time_varying_lambdas=True,
                window_size=1, 
                n_iter=niter,
                eval_data=None,
                allow_mean=False,
    eval_weight_lmbda=eval_weight_lmbda,
                resid_reg=estimate_reg,
                run_in_parallel=False)
    
    
    recon = np.concatenate([np.expand_dims(x @ l.T, axis=0) for x,l in zip(gamma_ts, lmbdas)], axis=0)
    
    return recon, gamma_ts, lmbdas


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

def impute_em(data, min_chars, maxiter):
    
   

    def impute_t(data_t):
        mu, sigma = em(data_t, eps=1e-4, min_chars=min_chars, max_iter=maxiter)
        return np.expand_dims(impute_given_mu_sigma(data_t, mu, sigma, min_chars), axis=0), sigma, mu
    

    res = impute_t(data)
    
    
    return res


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


def factor_conditional_mean(imask, idata, Sigma, k, include_d=False, reg=0):
    evs, evects = np.linalg.eigh(Sigma)
    evs, evects = evs[::-1], evects[:,::-1]
    
    lmbda = evects[:,:k]
    
    if include_d:
        lmbda = lmbda @ np.diag(np.sqrt(evs[:k]))
        
    if reg == 0:
        return lmbda[~imask,:] @ (np.linalg.lstsq(lmbda[imask,:], idata[imask])[0])
    else:
        return lmbda[~imask,:] @ (np.linalg.lstsq(lmbda[imask,:].T @ lmbda[imask,:] + np.eye(k) * reg,
                                                  lmbda[imask,:].T @ idata[imask])[0])
        
