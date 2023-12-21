import os 
from python.plots_and_tables import plot_base
from python import missing_data_utils
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm 
from python import imputation_utils, imputation_model, imputation_metrics, imputation_model_simplified
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import scipy as scp
from scipy.optimize import LinearConstraint
from joblib import Parallel, delayed

class AppendixPlotBase(plot_base.PaperPlot, ABC):
    section = 'appendix'

class AppendixTableBase(plot_base.PaperTable, ABC):
    section = 'appendix'

class AvgMissingGaps(AppendixTableBase):
    name = "AvgMissingGaps"
    description = 'AvgMissingGaps'
    sigfigs = 3
    norm_func = np.sqrt
    
    def setup(self, percentile_rank_chars, return_panel, dates, permnos, chars, char_groupings):
   
        missing_length = np.ones(percentile_rank_chars.shape[1:]) * -1
        prev_obs = ~np.isnan(percentile_rank_chars[0])
        for t in tqdm(range(1, percentile_rank_chars.shape[0])):
            present_at_t = ~np.isnan(percentile_rank_chars[t])
            to_add = np.logical_and(present_at_t, missing_length > 0)
            for n,c in np.argwhere(to_add):
                missing_gaps[c].append(missing_length[n,c])
            missing_length[present_at_t] = 0
            missing_length[np.logical_and(prev_obs, ~present_at_t)] += 1
            prev_obs = np.logical_or(prev_obs, present_at_t)


        print("characteristic & number of gaps & mean gap length & median gap length \\\\")
        print("\\midrule")
        result_data = []
        for c, gaps in zip(chars, missing_gaps):
            print(c, " & ", len(gaps), " & ", round(np.mean(gaps), 2)," & ", round(np.median(gaps), 2), "\\\\")
            result_data.append([len(gaps), round(np.mean(gaps), 2), round(np.median(gaps), 2)])
        #     print("\\midrule")
        self.data_df = pd.DataFrame(data=result_data, 
                                    columns = ['characteristic', 'number of gaps', 'mean gap length', 'median gap length'],
                                   index=chars)
        

    
def get_imputation_metrics_by_size(gamma_ts, char_data, suff_stat_method, monthly_update_mask, char_groupings,
                                  nyse_buckets, return_panel,
                          eval_char_data=None, num_months_train=None):
    if eval_char_data is None:
        eval_char_data = char_data
    if suff_stat_method == 'last_val':
        suff_stats = np.expand_dims(imputation_model.get_sufficient_statistics_last_val(char_data, max_delta=None)[0], axis=3)
    elif suff_stat_method == 'next_val':
        suff_stats = np.expand_dims(imputation_model.get_sufficient_statistics_next_val(char_data, max_delta=None)[0], axis=3)
    elif suff_stat_method == 'fwbw':
        next_val_suff_stats = imputation_model.get_sufficient_statistics_next_val(char_data, max_delta=None)[0]
        prev_val_suff_stats = imputation_model.get_sufficient_statistics_last_val(char_data, max_delta=None)[0]
        suff_stats = np.concatenate([np.expand_dims(prev_val_suff_stats, axis=3), 
                                              np.expand_dims(next_val_suff_stats, axis=3)], axis=3)
    elif suff_stat_method == 'None':
        suff_stats = None
    imputed_chars, betas = imputation_model.impute_fixed_beta_regression(char_data, gamma_ts, suff_stats,
                                                                        num_months_train=num_months_train)
    
    ret_metrics = []
    for i in range(10):
        masked_eval_data = np.copy(eval_char_data)
        masked_eval_data[nyse_buckets[:,:,i] != 1] = np.nan
        by_char_metrics, by_char_q_metrics, by_char_m_metrics  = imputation_metrics.get_aggregate_imputation_metrics(imputed_chars,
                                                          masked_eval_data, None, monthly_update_mask, char_groupings)
        ret_metrics.append((by_char_metrics, by_char_q_metrics, by_char_m_metrics))
    return ret_metrics

from collections import defaultdict
def get_nyse_permnos_mask(dates_ap, permnos):
    permnos_nyse_data = pd.io.parsers.read_csv('nyse_permnos.csv')
    permnos_nyse_data[['permno', 'date']].to_numpy()
    permnos_to_ind = {}
    for i, p in enumerate(permnos):
        permnos_to_ind[p] = i

    dates_to_ind = defaultdict(list)
    for i, date in enumerate(dates_ap):
        dates_to_ind[date // 10000].append(i)
    T,N,_ = percentile_rank_chars.shape
    permno_mask = np.zeros((T,N), dtype=bool)
    for permno, date in permnos_nyse_data[['permno', 'date']].to_numpy():
        date = int(date.replace('-', ''))
        if date//10000 in dates_to_ind and permno in permnos_to_ind:

            pn_ind = permnos_to_ind[permno]
            for d_ind in dates_to_ind[date//10000]:
                permno_mask[d_ind, pn_ind] = 1
    return permno_mask

def get_deciles_nyse_cutoffs(permno_mask, size_chars):
    print(size_chars.shape)
    T, N = size_chars.shape
    decile_data = np.zeros((T, N, 10))
    to_decide_deciles = np.logical_and(~np.isnan(size_chars), permno_mask)
    for t in tqdm(range(T)):
        valid_values_sorted = np.sort(size_chars[t, to_decide_deciles[t]])
        interval_size = int(valid_values_sorted.shape[0] / 10)
        cutoffs = [[valid_values_sorted[i * interval_size], 
                    valid_values_sorted[min((i+1) * interval_size, 
                       valid_values_sorted.shape[0] - 1)]] for i in range(10)]
        cutoffs[-1][1] = 2 # don't ignore the biggest stock lol
        cutoffs[0][0] = -1
        for i in range(10):
            in_bucket = np.logical_and(size_chars[t,:] > cutoffs[i][0], 
                                      size_chars[t,:] <= cutoffs[i][1])
            decile_data[t,in_bucket,i] = 1
    return decile_data

def bucket_company_age(return_panel, age_buckets):
    first_occurance = np.argmax(~np.isnan(return_panel), axis=0)
    company_buckets = np.zeros_like(return_panel)
    company_buckets[:,:] = -1
    for t in range(return_panel.shape[0]):
        age = t - first_occurance
        for i, b in enumerate(age_buckets):
            lower, upper = b
            in_bucket = np.logical_and(age >= lower, age <= upper)
            company_buckets[t, in_bucket] = i 
    return company_buckets

def get_imputation_metrics_by_age(gamma_ts, char_data, suff_stat_method, monthly_update_mask, char_groupings,
                                  age_buckets, return_panel,
                          eval_char_data=None, num_months_train=None):
    if eval_char_data is None:
        eval_char_data = char_data
    if suff_stat_method == 'last_val':
        suff_stats = np.expand_dims(imputation_model.get_sufficient_statistics_last_val(char_data, max_delta=None)[0], axis=3)
    elif suff_stat_method == 'next_val':
        suff_stats = np.expand_dims(imputation_model.get_sufficient_statistics_next_val(char_data, max_delta=None)[0], axis=3)
    elif suff_stat_method == 'fwbw':
        next_val_suff_stats = imputation_model.get_sufficient_statistics_next_val(char_data, max_delta=None)[0]
        prev_val_suff_stats = imputation_model.get_sufficient_statistics_last_val(char_data, max_delta=None)[0]
        suff_stats = np.concatenate([np.expand_dims(prev_val_suff_stats, axis=3), 
                                              np.expand_dims(next_val_suff_stats, axis=3)], axis=3)
    elif suff_stat_method == 'None':
        suff_stats = None
    imputed_chars, betas = imputation_model.impute_fixed_beta_regression(char_data, gamma_ts, suff_stats,
                                                                        num_months_train=num_months_train)
    company_buckets = bucket_company_age(return_panel, age_buckets)
    ret_metrics = []
    for i,_ in enumerate(age_buckets):
        masked_eval_data = np.copy(eval_char_data)
        masked_eval_data[company_buckets != i] = np.nan
        by_char_metrics, by_char_q_metrics, by_char_m_metrics  = imputation_metrics.get_aggregate_imputation_metrics(imputed_chars,
                                                          masked_eval_data, None, monthly_update_mask, char_groupings)
        ret_metrics.append((by_char_metrics, by_char_q_metrics, by_char_m_metrics))
    return ret_metrics

def get_imputation_metrics_by_mask(gamma_ts, char_data, suff_stat_method, monthly_update_mask, char_groupings,
                                  permno_masks, return_panel,
                          eval_char_data=None, num_months_train=None):
            
            
    

    ret_metrics = []
    for fit_mask in permno_masks:
        
        gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=char_data, 
                return_panel= return_panel, min_chars=10, K=20, 
                num_months_train=char_data.shape[0],
                reg=0.01,
                time_varying_lambdas=False,
                window_size=1, 
                n_iter=3,
                eval_data=None,
                allow_mean=False,
                use_alternate_gamma_estm=False)

        oos_xs_imputations = imputation_model_simplified.get_all_xs_vals(char_data, reg=0.01, 
                                         Lmbda=lmbda, time_varying_lmbda=False)
        residuals = char_data - oos_xs_imputations

        imputed_chars = imputation_model_simplified.impute_chars(
            char_data, oos_xs_imputations, residuals, 
            suff_stat_method=suff_stat_method, 
            constant_beta=True, beta_weight=False
        )
        

        mask_metrics = []
        for mask in permno_masks:
            masked_eval_data = np.copy(eval_char_data)
            masked_eval_data[:, ~mask, :] = np.nan
            by_char_metrics, by_char_q_metrics, by_char_m_metrics  =\
                imputation_metrics.get_aggregate_imputation_metrics(imputed_chars,
                                                              masked_eval_data, None, monthly_update_mask, 
                                                                    char_groupings)
            mask_metrics.append((by_char_metrics, by_char_q_metrics, by_char_m_metrics))
        ret_metrics.append(mask_metrics)
    return ret_metrics


class ExcludingFinancialFirms(AppendixTableBase):
    name = 'ExcludingFinancialFirms'
    description = ''
    sigfigs = 4
    
    def setup(self, percentile_rank_chars, return_panel, chars, char_groupings, permnos, monthly_updates, char_map):
        self.data_string = []
        

        
        sic_fic = pd.read_csv("../data/sic_fic.csv")
        financial_permnos = sic_fic.loc[sic_fic.sic // 1000 == 6].LPERMNO.unique()
        financial_firm_filter = np.isin(permnos, financial_permnos)

        masks = [financial_firm_filter, ~financial_firm_filter]
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan

        by_char_metrics = get_imputation_metrics_by_mask(None, percentile_rank_chars, "last_val", None, 
                                                         char_groupings, masks,
                                                        return_panel, num_months_train=return_panel.shape[0],
                                                eval_char_data=update_chars)
        
        
        
        self.data_string.append((" fit & ", " eval & ", 'aggregate & ', "quarterly & ", 'monthly & '))
        labels = ["financial firms", "non financial firms"]

        for i in range(2):
            self.data_string.append(("eval on " + labels[i]))
            metrics = [[round(np.sqrt(np.nanmean(x)), 5) for x in y] for y in [x[i] for x in by_char_metrics]]
            for j,m in enumerate(metrics):
                if j == 0:
                    start_str = "\\multirow{2}{*}{" + labels[i] + "} & "
                else:
                    start_str = ' & '
                self.data_string.append((start_str + labels[j] + ' & ' + ' & '.join(["{:.2f}".format(x) for x in m]) + ' \\'))
            self.data_string.append(())
            

            
class PrevValXsGMMSciPy:
    
    def __init__(self, lag_moments, prev_val_suff_stats, deltas, gts, xs_imputed_values):
        moment_functions = []
        T = prev_val_suff_stats.shape[0]
        for lag in lag_moments:
            all_gts, all_xs_vals, all_prev_vals = [], [], []
            for t in range(lag):
                correct_delta = (deltas + t)[t:] == lag
                available = np.logical_and(~np.isnan(prev_val_suff_stats[:T-t]), correct_delta)
                available = np.logical_and(~np.isnan(xs_imputed_values[t:]), available)
                available = np.logical_and(~np.isnan(gts[t:]), available)
                all_gts.append(gts[t:][available])
                all_xs_vals.append(xs_imputed_values[t:][available])
                all_prev_vals.append(prev_val_suff_stats[t:][available])
            all_gts = np.concatenate(all_gts, axis=0)
            all_xs_vals = np.concatenate(all_xs_vals, axis=0)
            all_prev_vals = np.concatenate(all_prev_vals, axis=0)
            assert np.all(~np.isnan(all_gts))
            assert np.all(~np.isnan(all_xs_vals))
            assert np.all(~np.isnan(all_prev_vals))
            print(lag, all_gts.shape)
            for inst in [all_xs_vals, all_prev_vals]:
                moment_functions.append(moment(all_prev_vals, all_xs_vals, lag, all_gts, inst))
        
        self.moments = moment_functions
    
    def solve(self, param_vect):
        W = np.eye(len(self.moments))
        
        def G(W):
            def inner(pv):
                moment_means = np.array([np.mean(g(pv)) for g in self.moments])
                retval = moment_means @ W @ moment_means
                return retval
            return inner
        
        res = scp.optimize.minimize(G(W), x0 = param_vect, method = 'SLSQP', tol = 1.e-9,
                                   constraints=LinearConstraint(np.eye(len(param_vect)), 
                                                                -100*np.ones(len(param_vect)),
                                                                100*np.ones(len(param_vect)),
                                                                keep_feasible=False))
        return res
def moment(prev_vals, xs_preds, lag, gts, inst):
    def _apply(params):
        alpha_xs, beta_xs, gamma_xs, alpha_ts, beta_ts, gamma_ts = params
        return (gts - (alpha_xs + beta_xs * np.exp(-lag * gamma_xs)) * xs_preds\
                   - (alpha_ts + beta_ts * np.exp(-lag * gamma_ts)) * prev_vals) * inst
    return _apply
                    

def GMM(moments, param_vect, n_iter=1):
    results = []
    W = np.eye(len(moments))
    
    def G(W):
        def inner(pv):
            moment_means = np.array([np.mean(g(pv)) for g in moments])
            retval = moment_means @ W @ moment_means
            return retval
        return inner
    
    res = scp.optimize.minimize(G(W), x0 = param_vect, method = 'BFGS', tol = 1.e-9)
    
class ExponentialWeights(AppendixTableBase):
    name = 'ExponentialWeights'
    description = ''
    sigfigs = 4
    
    def setup(self, return_panel, chars, char_map, char_groupings):
        self.data_string = []
        eval_maps = {
            'MAR': "MAR_eval_data",
            'block': "prob_block_eval_data",
            'logit': "logit_eval_data",
        }
        fit_maps = {
            'MAR': "MAR_fit_data",
            'block': "prob_block_fit_data",
            'logit': "logit_fit_data",
        }
        for missing_type in tqdm(["MAR", 'block', 'logit']):
            self.data_string.append((missing_type))
            fit_data = imputation_utils.load_imputation(fit_maps[missing_type])
            eval_data = imputation_utils.load_imputation(eval_maps[missing_type])
#             return_panel = return_panel[-100:]

            T = fit_data.shape[0]

            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=fit_data, 
                return_panel= return_panel, min_chars=10, K=20, 
                num_months_train=T,
                reg=0.01,
                time_varying_lambdas=False,
                window_size=548, 
                n_iter=3,
                eval_data=None,
                allow_mean=False,
                use_alternate_gamma_estm=False)

            xs_imputed = imputation_model_simplified.get_all_xs_vals(fit_data, reg=0.01, 
                                         Lmbda=lmbda, time_varying_lmbda=False)
            
            prev_vals, lags = imputation_model_simplified.get_sufficient_statistics_last_val(fit_data, max_delta=None,
                                                                           residuals=None)
            print(prev_vals.shape, lags.shape)

            vals = [3, 6, 13, 21, 30]
            m_vals = [1, 3, 9, 10, 10, 21]
            params = []
            
            def run_GMM_FIT(c, prev_vals_i, eval_data_i, lags_i, xs_imputed_i):
                if char_map[c] == 'M':
                    vals_i = m_vals
                else:
                    vals_i = vals
                gmm_a2me = PrevValXsGMMSciPy(vals_i, prev_vals_i,
                                         lags_i, eval_data_i, 
                                         xs_imputed_i)
                res = gmm_a2me.solve(np.array([1, 1, 0.5, 1, 1, 0.5]))
                print(c, res.x)
                return res.x
            
#             for i, c in enumerate(tqdm(chars)):
#                 if char_map[c] == 'M':
#                     vals_i = m_vals
#                 else:
#                     vals_i = vals
#                 gmm_a2me = PrevValXsGMMSciPy(vals_i, prev_vals[:,:,i],
#                                          lags[:,:,i], eval_data[:,:,i], 
#                                          xs_imputed[:,:,i])
#                 res = gmm_a2me.solve(np.array([1, 1, 0.5, 1, 1, 0.5]))
#                 print(c, res.x)
#                 params.append(res.x)
            params = list(Parallel(n_jobs=4)(delayed(run_GMM_FIT)(c, prev_vals[:,:,i,0], 
                                                                  eval_data[:,:,i], lags[:,:,i], 
                                                         xs_imputed[:,:,i]) for i, c in enumerate(tqdm(chars))))

            imputed_arr = []
            for i, p in enumerate(params):
                alpha_xs, beta_xs, gamma_xs, alpha_ts, beta_ts, gamma_ts = p
                imputed = (alpha_xs + beta_xs * np.exp(-lags[:,:,i] * gamma_xs)) * xs_imputed[:,:,i] \
                        + (alpha_ts + beta_ts * np.exp(-lags[:,:,i] * gamma_ts)) * prev_vals[:,:,i,0]

                imputed_arr.append(imputed)

            imputed_arr = [np.expand_dims(x, axis=2) for x in imputed_arr]
            imputation = np.concatenate(imputed_arr, axis=2)


            metrics = imputation_utils.get_imputation_metrics(imputation, 
                                                 eval_char_data=eval_data, 
                                                 monthly_update_mask=None, 
                                                 char_groupings=char_groupings)
            self.data_string.append(['exponential_impute'] + [round(np.sqrt(np.nanmean(x)), 5) for x in metrics])
            
            
class ExcludingSmallFirms(AppendixTableBase):
    name = 'ExcludingSmallFirms'
    description = ''
    sigfigs = 4
    
    def setup(self, percentile_rank_chars, return_panel, chars, dates, char_groupings, permnos, char_map,
          monthly_updates):
        self.data_string = []
        gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=percentile_rank_chars, 
                return_panel= return_panel, min_chars=10, K=20, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01,
                time_varying_lambdas=False,
                window_size=548, 
                n_iter=3,
                eval_data=None,
                allow_mean=False,
                use_alternate_gamma_estm=False)
        
        base_path = "/home/svenl/repos/research/pelger/stock_characteristics"
        prices_and_permnos_and_dates  = pd.read_csv(os.path.join(base_path, "CRSP_MONTHLY_RAW_2.csv"))[['date', 
                                                                                                        'PERMNO','PRC']]
        bad_price_mask = []
        for i,t in enumerate(dates):
            bad_permnos = prices_and_permnos_and_dates.loc[(prices_and_permnos_and_dates.date == t) & 
                                                          (prices_and_permnos_and_dates.PRC <= 1)].PERMNO.unique()
            bad_price_mask.append(np.logical_or(np.isin(permnos, bad_permnos).reshape([1, -1]),
                                                np.isnan(return_panel[i])))
        bad_price_mask = np.vstack(bad_price_mask)
        bad_price_mask = np.all(bad_price_mask, axis=0)
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        masks = [bad_price_mask, ~bad_price_mask, np.ones_like(bad_price_mask, dtype=bool)]
        
        by_char_metrics = get_imputation_metrics_by_mask(gamma_ts, percentile_rank_chars, "last_val", None, 
                                                         char_groupings, masks,
                                                return_panel, num_months_train=return_panel.shape[0],
                                                eval_char_data=update_chars)
        
        self.data_string.append((" fit & ", " eval & ", 'aggregate & ', "quarterly & ", 'monthly & '))
        labels = ["$<$ \\$ 1 firms", "$\geq$ \\$ 1 firms", 'all']

        for i in range(3):
        #     print("eval on " + labels[i])
            metrics = [[round(np.sqrt(np.nanmean(x)), 5) for x in y] for y in [x[i] for x in by_char_metrics]]
            for j,m in enumerate(metrics):
                if j == 0:
                    print("\\midrule")
                    start_str = "\\multirow{3}{*}{" + labels[i] + "} & "
                else:
                    start_str = ' & '
                self.data_string.append((start_str + labels[j] + ' & ' + ' & '.join(["{:.2f}".format(x) for x in m]) + '\\\\'))
            self.data_string.append(())
            
            
            
class MetricsByNumberOfChars(AppendixTableBase):
    name = "MetricsByNumberOfChars"
    sigfigs = 2
    description = ''
    
    def setup(self, percentile_rank_chars, return_panel, char_groupings):
        self.data_string = []
        block_flag_panel = imputation_utils.load_imputation("prob_block_flag_panel")
        block_eval_data = imputation_utils.load_imputation("prob_block_eval_data")
        block_fit_data = imputation_utils.load_imputation("prob_block_fit_data")
        
        T = percentile_rank_chars.shape[0]

        block_metrics_by_num_factors = [[], [], []]
        for k in range(1, 10):    

            
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=block_fit_data, 
                return_panel= return_panel, min_chars=10, K=k, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01,
                time_varying_lambdas=True,
                window_size=548, 
                n_iter=3,
                eval_data=None,
                allow_mean=False,
                use_alternate_gamma_estm=False)

        #     imputation = impute_chars(gamma_ts, masked_lagged_chars[100:], 
        #                                          "last_val", None, char_groupings, num_months_train=T,
        #                                                         eval_char_data=update_chars[100:],
        #                                                            window_size=1)
            imputation = imputation_utils.impute_chars(gamma_ts, block_fit_data, 
                                                 "None", None, char_groupings, num_months_train=T,
                                                                eval_char_data=block_eval_data,
                                                                   window_size=1)

            by_char_metrics = imputation_utils.get_imputation_metrics(imputation, 
                                                     eval_char_data=block_eval_data, 
                                                     monthly_update_mask=None, 
                                                     char_groupings=char_groupings)
            block_metrics_by_num_factors[0].append(by_char_metrics)
            
            imputation = imputation_utils.impute_chars(gamma_ts, block_fit_data, 
                                                 "last_val", None, char_groupings, num_months_train=T,
                                                                eval_char_data=block_eval_data,
                                                                   window_size=1)

            by_char_metrics = imputation_utils.get_imputation_metrics(imputation, 
                                                     eval_char_data=block_eval_data, 
                                                     monthly_update_mask=None, 
                                                     char_groupings=char_groupings)
            block_metrics_by_num_factors[1].append(by_char_metrics)
            
            
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=block_fit_data, 
                return_panel= return_panel, min_chars=10, K=k, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01,
                time_varying_lambdas=False,
                window_size=548, 
                n_iter=3,
                eval_data=None,
                allow_mean=False,
                use_alternate_gamma_estm=False)
            
            imputation = imputation_utils.impute_chars(gamma_ts, block_fit_data, 
                                                 "last_val", None, char_groupings, num_months_train=T,
                                                                eval_char_data=block_eval_data,
                                                                   window_size=None)

            by_char_metrics = imputation_utils.get_imputation_metrics(imputation, 
                                                     eval_char_data=block_eval_data, 
                                                     monthly_update_mask=None, 
                                                     char_groupings=char_groupings)
            block_metrics_by_num_factors[2].append(by_char_metrics)
        
        
        agg_mmse = [[np.sqrt(np.nanmean(x[0])) for x in y] for y in block_metrics_by_num_factors]
        monthly_mmse = [[np.sqrt(np.nanmean(x[2])) for x in y] for y in block_metrics_by_num_factors]
        quarterly_mmse = [[np.sqrt(np.nanmean(x[1])) for x in y] for y in block_metrics_by_num_factors]

        labels = ['local XS', 'local B-XS', 'global B-XS']
        for i in range(3):
            self.data_string.append((labels[i]))
            for k in range(9):
                self.data_string.append((k+1, '&', agg_mmse[i][k], '&', quarterly_mmse[i][k], '&', monthly_mmse[i][k]))
                

def normalize_chars(chars, max_std_devs=5):
    T, N , C = chars.shape
    return_chars = np.copy(chars)
    for t in tqdm(range(T)):
        for c in range(C):
            inf_filter = np.isinf(return_chars[t,:,c])
            inf_sign = np.sign(return_chars[t,inf_filter,c])
            return_chars[t,inf_filter,c] = np.nan
            mu, sigma = np.nanmean(return_chars[t,:,c]), np.nanstd(return_chars[t,:,c])
            assert ~np.isinf(mu) and ~np.isinf(sigma) and ~np.isnan(mu) and ~np.isnan(sigma)
            windsorize_filter = np.abs(return_chars[t,:,c] - mu) > max_std_devs * sigma
            windworize_sign = np.sign(return_chars[t,windsorize_filter,c] - mu)
            return_chars[t,windsorize_filter,c] = windworize_sign * max_std_devs * sigma
            return_chars[t,inf_filter,c] = inf_sign * max_std_devs * sigma
            mu_wind, sigma_wind = np.nanmean(return_chars[t,:,c]), np.nanstd(return_chars[t,:,c])
            assert ~np.isinf(mu_wind) and ~np.isinf(sigma_wind) and ~np.isnan(mu_wind) and ~np.isnan(sigma_wind)
            return_chars[t,:,c] = (return_chars[t,:,c] - mu_wind) / sigma_wind
    return return_chars

import scipy as scp
def map_ranks_to_gaussian_space(percentile_rank_chars, scaling=0.95):
    return scp.stats.norm.ppf(percentile_rank_chars * 0.95 + 0.5)
def map_gaussian_space_to_ranks(norm_chars, scaling=0.95):
    return (scp.stats.norm.cdf(norm_chars) - 0.5) / scaling
def invert_percentiles_lognormal(percentiles, regular_chars, std_devs_filter = 3):
    imputed_values = np.copy(regular_chars)
    imputed_values[:,:,:] = np.nan
    for t in range(regular_chars.shape[0]):
        chars_t = np.copy(regular_chars[t])
        chars_t[np.isinf(chars_t)] = np.nan
        mu, sigma = np.nanmean(chars_t), np.nanstd(chars_t)
        chars_t[np.abs(chars_t - mu) > std_devs_filter * sigma] = np.nan
        mu, sigma = np.nanmean(chars_t), np.nanstd(chars_t)
        imputed_value[t] = scp.stats.lognorm(percentiles[t], s=sigma, loc=mu)
    

class NormalizationType(AppendixTableBase):
    name = 'NormalizationType'
    description = ''
    sigfigs = 4
    
    def setup(self, return_panel, chars, char_map, char_groupings, percentile_rank_chars, regular_chars):
        self.data_string = []
        
        eval_maps = {
            'MAR': "MAR_eval_data",
            'block': "prob_block_eval_data",
            'logit': "logit_eval_data",
        }
        fit_maps = {
            'MAR': "MAR_fit_data",
            'block': "prob_block_fit_data",
            'logit': "logit_fit_data",
        }
        missing_type  = 'block'
        print(missing_type)
        fit_mask = ~np.isnan(imputation_utils.load_imputation(fit_maps[missing_type]))
        eval_mask = ~np.isnan(imputation_utils.load_imputation(eval_maps[missing_type]))
        
        factor_range = list(range(10, 21))
        
        
        ### config 4, normed from raw
        normed_chars = normalize_chars(regular_chars)
        
        scalar = np.nanmean(np.square(percentile_rank_chars)) / np.nanmean(np.square(normed_chars))
        print("scalar was", scalar)
        
        names = [f"out of sample {i}"  for i in factor_range]

        only_mimissing_chars = np.copy(normed_chars)
        only_mimissing_chars[~eval_mask] = np.nan

        masked_lagged_chars = np.copy(normed_chars)
        masked_lagged_chars[~fit_mask] = np.nan
        

        zero_pred_metrics = imputation_metrics.get_aggregate_imputation_metrics(np.zeros_like(only_mimissing_chars),
                                                                  only_mimissing_chars,
                                                                   None, None, char_groupings)
        zero_pred_scale = [np.sqrt(np.nanmean(x)) for x in zero_pred_metrics]

        oos_metrics = []
        for i in tqdm(factor_range):
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=masked_lagged_chars, 
                return_panel= return_panel, min_chars=1, K=i, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01 / scalar,
                time_varying_lambdas=True,
                window_size=548, 
                n_iter=3,
                eval_data=None,
                allow_mean=False,
                use_alternate_gamma_estm=False)
            imputed = np.concatenate([np.expand_dims(g @ l.T, axis=0) for g,l in zip(gamma_ts, lmbda)], axis=0)
            
            oos_metrics.append(imputation_utils.get_imputation_metrics(imputed, 
                                      eval_char_data=only_mimissing_chars, monthly_update_mask=None, 
                                      char_groupings=char_groupings, norm_func=None))
            
        self.data_string.append(("normed chars local xs oos"))
        self.data_string.append(("  ", 'aggregate', "quarterly", 'monthly'))
        
        metrics = [[round(np.sqrt(np.nanmean(x)), 5) for x in y] for y in oos_metrics]
        for i, m in enumerate(metrics):
            self.data_string.append((names[i], '&', ' & '.join(["{:.3f}".format(x/y) for x,y in zip(m, zero_pred_scale)]) + ' \\\\'))


        oos_metrics = []
        for i in tqdm(factor_range):
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=masked_lagged_chars, 
                return_panel= return_panel, min_chars=10, K=i, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01 / scalar,
                time_varying_lambdas=False,
                window_size=548, 
                n_iter=3,
                eval_data=None,
                allow_mean=False,
                use_alternate_gamma_estm=False)
            imputed = np.concatenate([np.expand_dims(g @ lmbda.T, axis=0) for g in gamma_ts], axis=0)
            
            oos_metrics.append(imputation_utils.get_imputation_metrics(imputed, 
                                      eval_char_data=only_mimissing_chars, monthly_update_mask=None, 
                                      char_groupings=char_groupings, norm_func=None))

        self.data_string.append(("normed chars gloabl xs oos"))
        self.data_string.append(("  ", 'aggregate', "quarterly", 'monthly'))
        
        metrics = [[round(np.sqrt(np.nanmean(x)), 5) for x in y] for y in oos_metrics]
        for i, m in enumerate(metrics):
            self.data_string.append((names[i], '&', ' & '.join(["{:.3f}".format(x/y) for x,y in zip(m, zero_pred_scale)]) + ' \\\\'))
            
        #### config 4, normed from ranks
        
        normed_chars = map_ranks_to_gaussian_space(percentile_rank_chars)
        only_mimissing_chars = np.copy(normed_chars)
        only_mimissing_chars[~eval_mask] = np.nan

        masked_lagged_chars = np.copy(normed_chars)
        masked_lagged_chars[~fit_mask] = np.nan
        
        scalar = np.nanmean(np.abs(percentile_rank_chars / normed_chars))
        
        zero_pred_metrics = imputation_metrics.get_aggregate_imputation_metrics(np.zeros_like(only_mimissing_chars),
                                                                  only_mimissing_chars,
                                                                   None, None, char_groupings)
        zero_pred_scale = [np.sqrt(np.nanmean(x)) for x in zero_pred_metrics]
        
        oos_metrics = []
        
        oos_metrics = []
        for i in tqdm(factor_range):
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=masked_lagged_chars, 
                return_panel= return_panel, min_chars=1, K=i, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01 / scalar,
                time_varying_lambdas=True,
                window_size=548, 
                n_iter=3,
                eval_data=None,
                allow_mean=False,
                use_alternate_gamma_estm=False)
            imputed = np.concatenate([np.expand_dims(g @ l.T, axis=0) for g,l in zip(gamma_ts, lmbda)], axis=0)
            
            oos_metrics.append(imputation_utils.get_imputation_metrics(imputed, 
                                      eval_char_data=only_mimissing_chars, monthly_update_mask=None, 
                                      char_groupings=char_groupings, norm_func=None))
            
        self.data_string.append(("kernel transformation chars local xs oos"))
        self.data_string.append(("  ", 'aggregate', "quarterly", 'monthly'))

        metrics = [[round(np.sqrt(np.nanmean(x)), 5) for x in y] for y in oos_metrics]
        for i, m in enumerate(metrics):
            self.data_string.append((names[i], '&', ' & '.join(["{:.3f}".format(x/y) for x,y in zip(m, zero_pred_scale)]) + ' \\\\'))
            


        oos_metrics = []
        for i in tqdm(factor_range):
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=masked_lagged_chars, 
                return_panel= return_panel, min_chars=1, K=i, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01 / scalar,
                time_varying_lambdas=False,
                window_size=548, 
                n_iter=3,
                eval_data=None,
                allow_mean=False,
                use_alternate_gamma_estm=False)
            imputed = np.concatenate([np.expand_dims(g @ lmbda.T, axis=0) for g in gamma_ts], axis=0)
            
            oos_metrics.append(imputation_utils.get_imputation_metrics(imputed, 
                                      eval_char_data=only_mimissing_chars, monthly_update_mask=None, 
                                      char_groupings=char_groupings, norm_func=None))

        self.data_string.append(("kernel transformation chars gloabl xs oos"))
        self.data_string.append(("  ", 'aggregate', "quarterly", 'monthly'))

        metrics = [[round(np.sqrt(np.nanmean(x)), 5) for x in y] for y in oos_metrics]
        for i, m in enumerate(metrics):
            self.data_string.append((names[i], '&', ' & '.join(["{:.3f}".format(x/y) for x,y in zip(m, zero_pred_scale)]) + ' \\\\'))


            
            
        only_mimissing_chars = np.copy(percentile_rank_chars)
        only_mimissing_chars[~eval_mask] = np.nan

        masked_lagged_chars =  np.copy(percentile_rank_chars)
        masked_lagged_chars[~fit_mask] = np.nan
        
        zero_pred_metrics = imputation_metrics.get_aggregate_imputation_metrics(np.zeros_like(only_mimissing_chars),
                                                                  only_mimissing_chars,
                                                                   None, None, char_groupings)
        zero_pred_scale = [np.sqrt(np.nanmean(x)) for x in zero_pred_metrics]
        
        oos_metrics = []

        for i in tqdm(range(20, 21)):
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=masked_lagged_chars, 
                return_panel= return_panel, min_chars=1, K=i, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01,
                time_varying_lambdas=True,
                window_size=548, 
                n_iter=3,
                eval_data=None,
                allow_mean=False,
                use_alternate_gamma_estm=False)
            imputed = np.concatenate([np.expand_dims(g @ l.T, axis=0) for g,l in zip(gamma_ts, lmbda)], axis=0)
            
            oos_metrics.append(imputation_utils.get_imputation_metrics(imputed, 
                                      eval_char_data=only_mimissing_chars, monthly_update_mask=None, 
                                      char_groupings=char_groupings, norm_func=None))
        
        
        self.data_string.append(("local xs oos"))
        self.data_string.append(("  ", 'aggregate', "quarterly", 'monthly'))

        metrics = [[round(np.sqrt(np.nanmean(x)), 5) for x in y] for y in oos_metrics]
        print(metrics)
        print(zero_pred_scale)
        for i, m in enumerate(metrics):
            self.data_string.append((names[i], '&', ' & '.join(["{:.3f}".format(x/y) for x,y in zip(m, zero_pred_scale)]) + ' \\\\'))

                                       
        







