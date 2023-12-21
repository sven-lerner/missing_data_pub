import numpy as np
from itertools import chain, combinations
from tqdm.notebook import tqdm
from collections import defaultdict
import imputation_utils
import matplotlib.pyplot as plt
import pandas as pd

SAVE_BASE = '../images-pdfs'
import time

mycolors = ['#152eff', '#e67300', '#0e374c', '#6d904f', '#8b8b8b', '#30a2da', '#e5ae38', '#fc4f30', '#6d904f', '#8b8b8b', '#0e374c']
c = mycolors[:4]

def get_vw_portfolio_returns(returns, mask, formation_lag, sizes):
    start = time.time()
    T, N = returns.shape
    portfolio_returns = np.zeros((T-formation_lag, 1))
    counts = np.zeros((T-formation_lag))
    start = time.time()
    
    for t in range(T - formation_lag):
        try:
            in_portfolio = np.logical_and(mask[t], ~np.isnan(sizes[t+formation_lag]))
            start = time.time()
            t_mask = np.logical_and(mask[t], ~np.isnan(returns[t+formation_lag]))
            start = time.time()
            
            counts[t] = np.sum(t_mask)
            start = time.time()

            portfolio_returns[t,0] = np.average(returns[t+formation_lag, t_mask], 
                                      weights=np.nan_to_num(sizes[t+formation_lag, t_mask]))
            start = time.time()
        except Exception as e:
            print(e)
            print("failed timestep", t)
            portfolio_returns[t,0] = np.nan
            counts[t] = np.nan
        
        
    return portfolio_returns, counts

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_combinations_map(chars, returns, char_names, initial_mask, 
                              char_sequence, sizes, portfolio_form_lag, shares, permutations=True):
    return_map = {}
    
    if permutations:
        pset = powerset(char_sequence)
    else:
        pset = [char_sequence[:i] for i in range(len(char_sequence) + 1)]
    
    return_present = ~np.isnan(returns)
    total_available = np.sum(return_present, axis=1)[portfolio_form_lag:,]
    total_size = np.nansum(sizes * return_present, axis=1)
    total_shares = np.nansum(shares * return_present, axis=1)
    
    for subset in tqdm(list(pset)):
        subset = tuple(sorted(subset))
        
        portfolio_mask = np.copy(initial_mask)
        for _c in subset:
            char_ind =np.argwhere(char_names == _c)[0][0] 
            char_mask = ~np.isnan(chars[:,:,char_ind])
            portfolio_mask = np.logical_and(char_mask, portfolio_mask)
        portfolio_returns, counts = get_vw_portfolio_returns(returns, portfolio_mask, portfolio_form_lag, sizes)
        percent_included = np.nanmean(counts / total_available)
        vw_percent_included = np.nanmean(np.nansum(sizes * portfolio_mask, axis=1) / total_size)
        perc_shares = np.nanmean(np.nansum(shares * portfolio_mask, axis=1) / total_shares)

        curr_res = (np.nanmean(portfolio_returns) / np.nanstd(portfolio_returns),
            percent_included,
            vw_percent_included, perc_shares, np.nanmean(portfolio_returns),
                   np.nanstd(portfolio_returns))
        
        return_map[subset] = curr_res
    char_mask = np.all(~np.isnan(chars), axis=2)
    portfolio_mask = np.logical_and(char_mask, initial_mask)
    portfolio_returns, counts = get_vw_portfolio_returns(returns, portfolio_mask, portfolio_form_lag, sizes)
    percent_included = np.nanmean(counts / total_available)
    vw_percent_included = np.nanmean(np.nansum(sizes * portfolio_mask, axis=1) / total_size)
    perc_shares = np.nanmean(np.nansum(shares * portfolio_mask, axis=1) / total_shares)
    curr_res = (np.nanmean(portfolio_returns) / np.nanstd(portfolio_returns),
            percent_included,
            vw_percent_included, 
              perc_shares, np.nanmean(portfolio_returns), np.nanstd(portfolio_returns))
    
    return_map["all"] = curr_res
    
    return return_map


def get_nyse_permnos_mask(dates_ap, permnos, percentile_rank_chars):
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

def get_buckets_nyse_cutoffs(permno_mask, present_char_panel, 
                                 full_char_panel, n_bins):
    T, N, C = full_char_panel.shape
    percentile_indicators = np.zeros((T, N, C, n_bins), dtype=bool)
    to_decide_deciles = permno_mask
    for t in tqdm(range(T)):
        for c in range(C):
            c_filter = np.logical_and(to_decide_deciles[t], ~np.isnan(present_char_panel[t, :, c]))
            valid_values_sorted = np.sort(present_char_panel[t, c_filter, c])
            assert(np.all(~np.isnan(valid_values_sorted)))
            valid_values_sorted = valid_values_sorted[~np.isnan(valid_values_sorted)]
            interval_size = int(valid_values_sorted.shape[0] / n_bins)
            cutoffs = [[valid_values_sorted[i * interval_size], 
                        valid_values_sorted[min((i+1) * interval_size, 
                           valid_values_sorted.shape[0] - 1)]] for i in range(n_bins)]
            cutoffs[-1][1] = np.inf
            cutoffs[0][0] = -np.inf
            prev_bucket = None
            for i in range(n_bins):
                in_bucket = np.logical_and(full_char_panel[t,:,c] > cutoffs[i][0], 
                                          full_char_panel[t,:,c] <= cutoffs[i][1])
                percentile_indicators[t,in_bucket,c,i] = True
    return percentile_indicators

def plot_page_multiple_maps(observed_chars, observed_chars_page_masks, imputed_chars, imputed_chars_page_masks, 
             char_sequence, return_panel, sizes, chars, title, shares, map_labels, prefix=''):
    
    sr_reses, per_reses, vw_per_reses, per_share_reses, xs_sm, mean_ret_reses, vol_reses =  [], [], [], [], [], [], []
    xs = None
    labels = []
    
    for mask, label in zip(observed_chars_page_masks, map_labels):
        observed_res = generate_combinations_map(observed_chars, return_panel,
                                                                    chars, mask, 
                                                                    char_sequence, sizes, 
                                                     4, shares, permutations=False)
        
        sr_res, per_res, vw_per_res, per_share_res, xs, mean_ret_res, vol_res = [], [], [], [], [], [], []
        
        for k, v in observed_res.items():
            sr_res.append(v[0])
            per_res.append(v[1])
            vw_per_res.append(v[2])
            per_share_res.append(v[3])
            mean_ret_res.append(v[4])
            vol_res.append(v[5])
            if k == "all":
                xs.append(len(char_sequence)+1)
            else:
                xs.append(len(k))
                
        sr_reses.append(sr_res)
        per_reses.append(per_res)
        mean_ret_reses.append(mean_ret_res)
        vol_reses.append(vol_res)
        labels.append(label + '-Observed')

        

    for mask, label in zip(imputed_chars_page_masks, map_labels):
        imputed_res = generate_combinations_map(imputed_chars, return_panel, 
                                                                       chars, mask, 
                                                                       char_sequence, sizes, 
                                                         4, shares, permutations=False)





        sr_res, per_res, vw_per_res, per_share_res, xs, mean_ret_res, vol_res = [], [], [], [], [], [], []
        for k, v in imputed_res.items():
    #         label = '-'.join([x[4] for x in seq])
            sr_res.append(v[0])
            per_res.append(v[1])
            vw_per_res.append(v[2])
            per_share_res.append(v[3])
            mean_ret_res.append(v[4])
            vol_res.append(v[5])
            if k == "all":
                xs.append(len(char_sequence)+1)
            else:
                xs.append(len(k))
        sr_reses.append(sr_res)
        per_reses.append(per_res)
        mean_ret_reses.append(mean_ret_res)
        vol_reses.append(vol_res)
        labels.append(label + '-Imputed')
    
    plot_names = ["Sharpe Ratio", "Percent Used", "Mean Return", "Volatility"]
    plt_data = [sr_reses, per_reses, mean_ret_reses, vol_reses]
    for data, name in zip(plt_data, plot_names):
        plt.figure(figsize=(10, 8))
        for i, data_series in enumerate(data):
            if 'Imputed' in labels[i]:
                plt.plot(xs, data_series, label=labels[i], linewidth=2.0)
            else:
                plt.plot(xs, data_series, label=labels[i], linewidth=4.0, marker='o')
            
#         plt.xticks(range(len(char_sequence)+2), ['-'] + char_sequence + ["All"], rotation=45, size=35)
#         plt.yticks(size=35)
#         plt.title(name, size=35)
#         if name == "Sharpe Ratio":
#             plt.legend(prop={'size': 30}, framealpha=0.5)
        plt.xticks(range(len(char_sequence)+2), ['-'] + char_sequence + ["All"], rotation=45, fontsize=30)
        plt.yticks(fontsize=30)
        plt.minorticks_off()
        plt.title(name, fontsize=40)
        if name == "Sharpe Ratio" and title == 'B2M':
            plt.legend(prop={'size': 30}, framealpha=0.5)

        plt.savefig(f'{SAVE_BASE}/section6/portfolio-sorts-{title}-{name}{prefix}.pdf'.replace(' ', '_'), 
                    bbox_inches='tight')
        plt.tight_layout()
        plt.show()
    
def impute_chars(gamma_ts, char_data, suff_stat_method, monthly_update_mask, char_groupings,
                          eval_char_data=None, num_months_train=None, window_size=None):
    if eval_char_data is None:
        eval_char_data = char_data
    if suff_stat_method == 'last_val':
        suff_stats, _ = imputation_model.get_sufficient_statistics_last_val(char_data, max_delta=None)
        suff_stats = np.expand_dims(suff_stats, axis=3)
    elif suff_stat_method == 'next_val':
        suff_stats = np.expand_dims(imputation_model.get_sufficient_statistics_next_val(char_data, max_delta=None)[0], axis=3)
    elif suff_stat_method == 'fwbw':
        next_val_suff_stats = imputation_model.get_sufficient_statistics_next_val(char_data, max_delta=None)[0]
        prev_val_suff_stats = imputation_model.get_sufficient_statistics_last_val(char_data, max_delta=None)[0]
        suff_stats = np.concatenate([np.expand_dims(prev_val_suff_stats, axis=3), 
                                              np.expand_dims(next_val_suff_stats, axis=3)], axis=3)
    elif suff_stat_method == 'None':
        suff_stats = None
    
    if window_size is not None:
        imputed_chars, betas = imputation_model.impute_beta_regression(char_data, gamma_ts, suff_stats,
                                                                      window_size=window_size)
    else:
        imputed_chars, betas = imputation_model.impute_fixed_beta_regression(char_data, gamma_ts, suff_stats,
                                                                        num_months_train=num_months_train)
    return imputed_chars

from matplotlib.lines import Line2D

import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost

def bar_plot_mean_sharpe_diffs_seq(h_port_returns, l_port_returns, title, chars_to_plot_seq, series_to_plot,
                                  group_names, h_port_mean_returns, l_port_mean_returns, chars):
    

    
    from matplotlib.lines import Line2D
    
    mean_bar_seq = []
    sharpe_bar_seq = []

    x_ticks = []
    gap = 1

    xs = []
    all_chars = []
    x_start = 0
    sub_group_starts = []

    
    fig1 = plt.figure(figsize=(10, 1))
    ax = SubplotHost(fig1, 111)
    fig1.add_subplot(ax)
    fontsize = 12
    
    for k, chars_to_plot in enumerate(chars_to_plot_seq):
        all_chars += chars_to_plot
        for i, j in enumerate(np.argwhere(np.isin(chars, chars_to_plot)).squeeze()):

            h_port_mean_returns = (np.mean(h_port_returns[j][0][0]),
                                       np.mean(h_port_returns[j][1][0]))

            l_port_mean_returns = (np.mean(l_port_returns[j][0][0]),
                                       np.mean(l_port_returns[j][1][0]))

            h_port_sharpes = (h_port_mean_returns[0] / np.std(h_port_returns[j][0][0]),
                                  h_port_mean_returns[1] / np.std(h_port_returns[j][1][0]))

            l_port_sharpes = (l_port_mean_returns[0] / np.std(l_port_returns[j][0][0]),
                                  l_port_mean_returns[1] / np.std(l_port_returns[j][1][0]))

            mean_bar_seq += [l_port_mean_returns[0],
                            l_port_mean_returns[1],
                            h_port_mean_returns[0],
                            h_port_mean_returns[1]]

            sharpe_bar_seq += [l_port_sharpes[0],
                            l_port_sharpes[1],
                            h_port_sharpes[0],
                            h_port_sharpes[1]]

            x_ticks.append(x_start + i*(4+gap) + 1.5)
            if i == 0:
                if k == 0:
                    sub_group_starts.append(x_ticks[-1])
                else:
                    sub_group_starts.append((x_ticks[-1] + x_ticks[-2]) / 2)

            xs += [x_start + x + i*(4 + gap) for x in range(4)]
        x_start = xs[-1] + gap*2
    sub_group_starts.append(x_ticks[-1])
    if series_to_plot == 'S':
        ax.bar(xs, sharpe_bar_seq, color=c*len(chars_to_plot))
        title = title + ' - Sharpe Ratio'
    if series_to_plot == 'M':
        ax.bar(xs, mean_bar_seq, color=c*len(chars_to_plot))
        title = title + ' - Mean Return'
#     ax.set_title(title)
    ax.set_xticks(x_ticks)
    plt.minorticks_off()
    ax.set_xticklabels(all_chars, fontsize=1)
    
    # Second X-axis
    ax2 = ax.twiny()
    offset = 0, -75 # Position of the second axis
    new_axisline = ax2.get_grid_helper().new_fixed_axis
    ax2.axis["bottom"] = new_axisline(loc="bottom", axes=ax2, offset=offset)
    ax2.axis["top"].set_visible(False)
    ax2.set_xticks(sub_group_starts)
    plt.minorticks_off()
    
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    plt.setp(ax2.axis["bottom"].major_ticks, ticksize=50)
    
    plt.setp(ax2.axis["bottom"].major_ticklabels, fontsize=fontsize)
    plt.setp(ax.axis["bottom"].major_ticklabels, fontsize=fontsize)
    plt.setp(ax.axis["left"].major_ticklabels, fontsize=fontsize)
    
    ax2.xaxis.set_minor_locator(ticker.FixedLocator([(sub_group_starts[i] + sub_group_starts[i+1])/2
                                                    for i in range(len(sub_group_starts) - 1)]))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(group_names))
    
    plt.setp(ax2.axis["bottom"].minor_ticklabels, fontsize=fontsize)

    
    custom_lines = [Line2D([0], [0], color=c[0], lw=4),
               Line2D([0], [0], color=c[1], lw=4),
                   Line2D([0], [0], color=c[2], lw=4),
               Line2D([0], [0], color=c[3], lw=4)]
    
    

    plt.setp(ax.axis["bottom"].major_ticklabels, rotation=90)
    ax.axis["bottom"].major_ticklabels.set_ha("right")

    if series_to_plot == 'S':
        plt.ylim(0, 0.6)
    ax.legend(custom_lines, ['Bottom Decile', 'Bottom Decile - Fully Observed', 
                          'Top Decile', 'Top Decile - Fully Observed'], framealpha=0.5,
              ncol=2,
              bbox_to_anchor=(0., 1.02, 1., .102), fontsize=fontsize)
    for x in ax.get_xticklabels() + ax.get_yticklabels():
        x.set_fontsize(fontsize)
    plt.savefig(f'{SAVE_BASE}/section6/hl-portfolios-{title}.pdf'.replace(' ', ''), 
                    bbox_inches='tight')
    plt.show()



def bar_plot_mean_sharpe_diffs(h_port_mean_returns, l_port_mean_returns, title, chars_to_plot, series_to_plot,
    h_port_returns, l_port_returns,
    chars):
    mean_bar_seq = []
    sharpe_bar_seq = []

    x_ticks = []
    gap = 1

    xs = []
    plt.figure(figsize=(10, 8))
    mean_diffs = {}
    for i, j in enumerate(np.argwhere(np.isin(chars, chars_to_plot)).squeeze()):

        h_port_mean_returns.append((np.mean(h_port_returns[j][0][0]),
                                   np.mean(h_port_returns[j][1][0]),
                                   np.mean(h_port_returns[j][2][0])))

        l_port_mean_returns.append((np.mean(l_port_returns[j][0][0]),
                                   np.mean(l_port_returns[j][1][0]),
                                   np.mean(l_port_returns[j][2][0])))

        h_port_sharpes.append((h_port_mean_returns[j][0] / np.std(h_port_returns[j][0][0]),
                              h_port_mean_returns[j][1] / np.std(h_port_returns[j][1][0]),
                              h_port_mean_returns[j][2] / np.std(h_port_returns[j][2][0])))

        l_port_sharpes.append((l_port_mean_returns[j][0] / np.std(l_port_returns[j][0][0]),
                              l_port_mean_returns[j][1] / np.std(l_port_returns[j][1][0]),
                              l_port_mean_returns[j][2] / np.std(l_port_returns[j][2][0])))

        mean_bar_seq += [-1*(l_port_mean_returns[-1][0] - l_port_mean_returns[-1][1]),
                         -1*(l_port_mean_returns[-1][0] - l_port_mean_returns[-1][2]),
                        -1*(h_port_mean_returns[-1][0] - h_port_mean_returns[-1][1]),
                        -1*(h_port_mean_returns[-1][0] - h_port_mean_returns[-1][2])]
        
        mean_diffs[chars_to_plot[i]] = (-1*(l_port_mean_returns[-1][0] - l_port_mean_returns[-1][1]),
                         -1*(l_port_mean_returns[-1][0] - l_port_mean_returns[-1][2]),
                        -1*(h_port_mean_returns[-1][0] - h_port_mean_returns[-1][1]),
                        -1*(h_port_mean_returns[-1][0] - h_port_mean_returns[-1][2]))

        sharpe_bar_seq += [-1*(l_port_sharpes[-1][0] - l_port_sharpes[-1][1]),
                        -1*(l_port_sharpes[-1][0] - l_port_sharpes[-1][2]),
                        -1*(h_port_sharpes[-1][0] - h_port_sharpes[-1][1]),
                        -1*(h_port_sharpes[-1][0] - h_port_sharpes[-1][2])]

        x_ticks.append(i*(4+gap) + 1.5)
        xs += [x + i*(4 + gap) for x in range(4)]
    if series_to_plot == 'S':
        plt.bar(xs, sharpe_bar_seq, color=c*len(chars_to_plot))
        title = title + ' - Sharpe Ratio'
    if series_to_plot == 'M':
        plt.bar(xs, mean_bar_seq, color=c*len(chars_to_plot))
        title = title + ' - Mean Return'
    plt.title(title)
    plt.xticks(x_ticks, chars_to_plot, rotation = 90)
    plt.minorticks_off()
    plt.savefig(f'{SAVE_BASE}/section6/masked-hl-portfolios-{title}.pdf'.replace(' ', ''), 
                    bbox_inches='tight')
    plt.show()
    return mean_diffs