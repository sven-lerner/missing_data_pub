import numpy as np
import imputation_metrics
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


line_styles = [
     'solid',      # Same as (0, ()) or '-'
     'dotted',    # Same as (0, (1, 1)) or ':'
     'dashed',    # Same as '--'
     'dashdot',  # Same as '-.'
     (5, (10, 3)),
     (0, (5, 10)),
     (0, (3, 10, 1, 10)),
     (0, (3, 5, 1, 5, 1, 5)),
     (0, (3, 1, 1, 1, 1, 1))
]

def get_imputation_metrics_by_size(imputed_chars, monthly_update_mask, char_groupings,
                                  nyse_buckets, return_panel,
                          eval_char_data=None, num_months_train=None,
                                  local_estimation=False):
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

def get_imputation_metrics_by_age(imputed_chars, monthly_update_mask, char_groupings,
                                  age_buckets, return_panel,
                          eval_char_data=None, num_months_train=None):
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
    
    ret_metrics = []
    for fit_mask in permno_masks:

        imputed_chars, betas = imputation_model.impute_fixed_beta_regression(char_data, gamma_ts, suff_stats,
                                                                            num_months_train=num_months_train,
                                                                            fit_mask=fit_mask)

        mask_metrics = []
        for mask in permno_masks:
            masked_eval_data = np.copy(eval_char_data)
            masked_eval_data[:, ~mask, :] = np.nan
            by_char_metrics, by_char_q_metrics, by_char_m_metrics  = imputation_metrics.get_aggregate_imputation_metrics(imputed_chars,
                                                              masked_eval_data, None, monthly_update_mask, char_groupings)
            mask_metrics.append((by_char_metrics, by_char_q_metrics, by_char_m_metrics))
        ret_metrics.append(mask_metrics)
    return ret_metrics


def plot_metrics_over_time(metrics, names, dates, save_name=None, extra_line=None, nans_ok=False):
    save_base = '../images-pdfs/section5/metrics_over_time-'
    

    date_vals = np.array(dates) // 10000 + ((np.array(dates) // 100) % 100) / 12
    
    start_idx = metrics[0][0][0].shape[0] - date_vals.shape[0]
    print(start_idx)

    plot_names = ["aggregate", "quarterly_chars", "monthly_chars"]
    
    for i, plot_name in enumerate(plot_names):
        print(plot_name)
        plt.tight_layout() 
        fig, axs = plt.subplots(1, 1, figsize=(20,10))
        fig.patch.set_facecolor('white')

        for j, (data, label) in enumerate(zip(metrics, names)):

            metrics_i = data[i]
            
            label = f'{label}'
            if nans_ok:
                plt.plot(dates, np.sqrt(np.nanmean(np.array(metrics_i), axis=0))[start_idx:], label=label,
                        linestyle=line_styles[j])
            else:
                plt.plot(dates, np.sqrt(np.mean(np.array(metrics_i), axis=0))[start_idx:], label=label,
                        linestyle=line_styles[j])

        if extra_line is not None:
            ax2 = axs.twinx()
            ax2.plot(dates, extra_line, label="extra_line", c='red')
            ax2.legend(prop={'size': 14})
        if i == 0:
            axs.legend(prop={'size': 20}, loc='upper center', bbox_to_anchor=(0.5, 1.2),
              ncol=4, framealpha=1)



#         plt.ylabel("RMSE")
#         plt.xticks(fontsize=25)
#         plt.ylim(0.08, 0.32)
#         plt.yticks([0.1, 0.15, 0.2, 0.25, 0.3], fontsize=25)
        
        if save_name is not None:
            print("saving, theoretically... ")
            fig.savefig(save_base + save_name + f'-{plot_name}.pdf', bbox_inches='tight')
        plt.show()
        
        
def plot_metrics_by_mean_vol(mean_vols, input_metrics, names, chars, save_name=None, ylabel=None):
    
    char_names = []
    metrics_by_type = [[] for _ in input_metrics] 

    for i in np.argsort(mean_vols):
        metrics = [round(np.sqrt(np.nanmean(y[0][i])), 5) for y in input_metrics]
        char_names.append(chars[i])
        for j, m in enumerate(metrics):
            metrics_by_type[j].append(m)
    plt.tight_layout() 
    fig = plt.figure(figsize=(20,10))
    fig.patch.set_facecolor('white')
    mycolors = ['#152eff', '#e67300', '#0e374c', '#6d904f', '#8b8b8b', '#30a2da', '#e5ae38', '#fc4f30', '#6d904f', '#8b8b8b', '#0e374c']
    for j, (c, line_name, metrics_series) in enumerate(zip(mycolors, names,
                                 metrics_by_type)):
        plt.plot(np.arange(45), metrics_series, label=line_name, c=c, linestyle=line_styles[j])
    plt.plot(np.arange(45), np.array(mean_vols)[np.argsort(mean_vols)], label="mean volatility of char", c='black')
    plt.xticks(np.arange(45), chars[np.argsort(mean_vols)], rotation='vertical')
    if ylabel is None:
        plt.ylabel("RMSE")
    else:
        plt.ylabel("ylabel")
    plt.legend(prop={'size': 20}, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, framealpha=1)
    plt.minorticks_off()
    
    if save_name is not None:
        save_base = '../images-pdfs/section5/metrics_by_char_vol_sort-'
        save_path = save_base + save_name + '.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.show()

def save_metrics(metrics, name):
    base_path = '../data/metrics_cache/'
    result_file_name = base_path + name + '.pkl'
    with open(result_file_name, 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result_file_name
def load_metrics(name):
    base_path = '../data/metrics_cache/'
    result_file_name = base_path + name + '.pkl'
    with open(result_file_name, 'rb') as handle:
        result = pickle.load(handle)
    return result
def save_imputation(imputed_data, dates, permnos, chars, name):
    base_path = '../data/imputation_cache/'
    result_file_name = base_path + name + '.npz'
    np.savez(result_file_name, data=imputed_data, dates=dates, permnos=permnos, chars=chars)
    
def load_imputation(name, full=False):
    base_path = '../data/imputation_cache/'
    result_file_name = base_path + name + '.npz'
    res = np.load(result_file_name)
    if not full:
        return res['data']
    else:
        return res['data'], res['dates'], res['permnos'], res['chars']

    
def impute_chars(gamma_ts, char_data, suff_stat_method, monthly_update_mask, char_groupings,
                          eval_char_data=None, num_months_train=None, window_size=None,
                        lmbda=None, tv_lmbda=False, beta_weight=True,
                        use_xs_imp_for_regression=False, residuals=None, include_residual_in_regression=False,
                        only_use_residuals=False):
    if eval_char_data is None:
        eval_char_data = char_data
    if suff_stat_method == 'last_val':
        if include_residual_in_regression:
            suff_stats, _ = imputation_model.get_sufficient_statistics_last_val(char_data, max_delta=None,
                                                                               residuals=residuals)
            if only_use_residuals:
                suff_stats = suff_stats[:,:,:,1:]
        else:
            suff_stats, _ = imputation_model.get_sufficient_statistics_last_val(char_data, max_delta=None,
                                                                               residuals=None)
        if len(suff_stats.shape) == 3:
            suff_stats = np.expand_dims(suff_stats, axis=3)
        beta_weights = None
    elif suff_stat_method == 'next_val':
        suff_stats = np.expand_dims(imputation_model.get_sufficient_statistics_next_val(char_data, max_delta=None)[0], axis=3)
        beta_weights = None
    elif suff_stat_method == 'fwbw':
        next_val_suff_stats, fw_deltas = imputation_model.get_sufficient_statistics_next_val(char_data, max_delta=None)
        prev_val_suff_stats, bw_deltas = imputation_model.get_sufficient_statistics_last_val(char_data, max_delta=None)
        suff_stats = np.concatenate([prev_val_suff_stats, 
                                              np.expand_dims(next_val_suff_stats, axis=3)], axis=3)
        if beta_weight:            
            beta_weight_arr = np.concatenate([np.expand_dims(fw_deltas, axis=3), 
                                                  np.expand_dims(bw_deltas, axis=3)], axis=3)
            beta_weight_arr = 2 * beta_weight_arr / np.sum(beta_weight_arr, axis=3, keepdims=True)
            beta_weights = {}
            one_arr = np.ones((gamma_ts.shape[-1], 1))
            for t, i, j in tqdm(np.argwhere(np.logical_and(~np.isnan(fw_deltas), ~np.isnan(bw_deltas)))):
                beta_weights[(t,i,j)] = np.concatenate([one_arr, beta_weight_arr[t,i,j].reshape(-1, 1)], axis=0).squeeze()
        else:
            beta_weights = None
        
    elif suff_stat_method == 'None':
        suff_stats = None
        beta_weights = None
        
    if lmbda is not None:
        if tv_lmbda:
            imputed_chars = np.concatenate([np.expand_dims(x @ l.T, axis=0) for x,l in zip(gamma_ts, lmbda)], axis=0)
        else:
            imputed_chars = np.concatenate([np.expand_dims(x @ lmbda.T, axis=0) for x in gamma_ts], axis=0)
        if suff_stats is None:
            return imputed_chars
        else:
            return imputation_model.impute_beta_combined_regression(char_data, imputed_chars, sufficient_statistics=suff_stats, 
                           window_size=0, beta_weights=None)
    
    if window_size is not None:
        imputed_chars, betas = imputation_model.impute_beta_regression(char_data, gamma_ts, suff_stats,
                                                                      window_size=window_size, beta_weights=beta_weights)
    else:
        imputed_chars, betas = imputation_model.impute_fixed_beta_regression(char_data, gamma_ts, suff_stats,
                                                                        num_months_train=num_months_train,
                                                                        beta_weights=beta_weights)
    return imputed_chars

def get_imputation_metrics(imputed_chars, eval_char_data, monthly_update_mask, char_groupings, norm_func=None,
                          clip=True):
    by_char_metrics, by_char_m_metrics, by_char_q_metrics  = imputation_metrics.get_aggregate_imputation_metrics(imputed_chars,
                                                          eval_char_data, None, monthly_update_mask, char_groupings,
                                                          norm_func=norm_func, clip=clip)
    return by_char_metrics, by_char_q_metrics, by_char_m_metrics

def simple_imputation(gamma_ts, char_data, suff_stat_method, monthly_update_mask, char_groupings,
                                 eval_char_data=None, num_months_train=None, median_imputation=False,
                                 industry_median=False, industries=None):
    if eval_char_data is None:
        eval_char_data = char_data
    imputed_chars = imputation_model.simple_impute(char_data)
    if median_imputation:
        imputed_chars[:,:,:] = 0
    elif industry_median:
        imputed_chars = imputation_model.xs_industry_median_impute(char_panel=char_data, industry_codes=industries)
        
    return imputed_chars


def print_metrics_table(metrics_vals, names, norm_func=None):
    if norm_func is None:
        norm_func = np.sqrt
    print("  ", 'aggregate', "quarterly", 'monthly')
    metrics = [[round(norm_func(np.nanmean(x)), 5) for x in y] for y in metrics_vals]
    for i, m in enumerate(metrics):
        print(names[i], '&', ' & '.join(["{:.2f}".format(x) for x in m]) + ' \\\\')
        
        
        
def print_metrics_table_by_size(metrics_vals, names):
    print("  ", 'aggregate', "quarterly", 'monthly')

    for i in range(10):
        print("\\midrule")
        metrics = [[round(np.sqrt(np.nanmean(x)), 5) for x in y] for y in [x[i] for x in metrics_vals]]
        for j,m in enumerate(metrics):
            if j == 0:
                start_str = "\\multirow{3}{*}{" + str(i+1) + "} & "
            else:
                start_str = '& '
            print(start_str + names[j] + ' & ' + ' & '.join(["{:.2f}".format(x) for x in m]) + ' \\\\')
        print()
        
        
def get_present_flags(raw_char_panel):
    
    T, N, C = raw_char_panel.shape
    flag_panel = np.zeros_like(raw_char_panel, dtype=np.int8)
    
    first_occurances = np.argmax(~np.isnan(raw_char_panel), axis=0)
    not_in_sample = np.all(np.isnan(raw_char_panel), axis=0)
    last_occurances = T - 1 - np.argmax(~np.isnan(raw_char_panel[::-1]), axis=0)
    
    for t in tqdm(range(raw_char_panel.shape[0])):
        
        present_mask = ~np.isnan(raw_char_panel[t])
        previous_entry = t == first_occurances
        next_entry = t == last_occurances
        
        flag_panel[t, np.logical_and(present_mask, previous_entry)] = -1
        
        flag_panel[t, np.logical_and(present_mask, next_entry)] = -3
        
        both = np.logical_and(t > first_occurances, t < last_occurances)
        
        flag_panel[t, np.logical_and(present_mask, both)] = -2
        previous_entry[present_mask] = 1
    flag_panel[:,not_in_sample] = 0
    return flag_panel