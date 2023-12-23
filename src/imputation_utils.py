import numpy as np
import imputation_metrics
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
    T, N = size_chars.shape
    decile_data = np.zeros((T, N, 10))
    to_decide_deciles = np.logical_and(~np.isnan(size_chars), permno_mask)
    for t in range(T):
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


def plot_metrics_over_time(metrics, names, dates, save_name=None, extra_line=None, nans_ok=False):
    save_base = '../images-pdfs/section5/metrics_over_time-'
    

    date_vals = np.array(dates) // 10000 + ((np.array(dates) // 100) % 100) / 12
    
    start_idx = metrics[0][0][0].shape[0] - date_vals.shape[0]

    plot_names = ["aggregate", "quarterly_chars", "monthly_chars"]
    
    for i, plot_name in enumerate(plot_names):
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
        
        if save_name is not None:
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


def get_imputation_metrics(imputed_chars, eval_char_data, monthly_update_mask, char_groupings, norm_func=None,
                          clip=True):
    by_char_metrics, by_char_m_metrics, by_char_q_metrics  = imputation_metrics.get_aggregate_imputation_metrics(imputed_chars,
                                                          eval_char_data, None, monthly_update_mask, char_groupings,
                                                          norm_func=norm_func, clip=clip)
    return by_char_metrics, by_char_q_metrics, by_char_m_metrics


def get_present_flags(raw_char_panel):
    
    T, N, C = raw_char_panel.shape
    flag_panel = np.zeros_like(raw_char_panel, dtype=np.int8)
    
    first_occurances = np.argmax(~np.isnan(raw_char_panel), axis=0)
    not_in_sample = np.all(np.isnan(raw_char_panel), axis=0)
    last_occurances = T - 1 - np.argmax(~np.isnan(raw_char_panel[::-1]), axis=0)
    
    for t in range(raw_char_panel.shape[0]):
        
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