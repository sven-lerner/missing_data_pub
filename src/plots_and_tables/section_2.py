import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore")

from plots_and_tables import plot_base
from abc import ABC
import numpy as np
from tqdm.notebook import tqdm 
import logit_models_and_masking
import pandas as pd
from sklearn import metrics
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


class SectionTwoPlotBase(plot_base.PaperPlot, ABC):
    section = 'section2'

class SectionTwoTableBase(plot_base.PaperTable, ABC):
    section = 'section2'


char_groupings  = [('A2ME', "Q"),
                   ('AC', 'Q'),
('AT', 'Q'),
('ATO', 'Q'),
('B2M', 'QM'),
('BETA_d', 'M'),
('BETA_m', 'M'),
('C2A', 'Q'),
('CF2B', 'Q'),
('CF2P', 'QM'),
('CTO', 'Q'),
('D2A', 'Q'),
('D2P', 'M'),
('DPI2A', 'Q'),
('E2P', 'QM'),
('FC2Y', 'QY'),
('IdioVol', 'M'),
('INV', 'Q'),
('LEV', 'Q'),
('ME', 'M'),
('TURN', 'M'),
('NI', 'Q'),
('NOA', 'Q'),
('OA', 'Q'),
('OL', 'Q'),
('OP', 'Q'),
('PCM', 'Q'),
('PM', 'Q'),
('PROF', 'QY'),
('Q', 'QM'),
('R2_1', 'M'),
('R12_2', 'M'),
('R12_7', 'M'),
('R36_13', 'M'),
('R60_13', 'M'),
('HIGH52', 'M'),
('RVAR', 'M'),
('RNA', 'Q'),
('ROA', 'Q'),
('ROE', 'Q'),
('S2P', 'QM'),
('SGA2S', 'Q'),
('SPREAD', 'M'),
('SUV', 'M'),
('VAR', 'M')]
char_map = {x[0]:x[1] for x in char_groupings}


def print_results(ret_data, include_chars, include_FE, include_last_val, include_missing_gap, params, t_stats,
                 train_auc, test_auc, tgt_chars, num_fit_chars=7, num_tgt_chars=37):
    col_names = list(sorted(tgt_chars)) + \
        ['FE included', 'Last Val Indicator', "Missing Gap"] + \
        ["agg train AOC", "agg  test AOC"]
    print_str = ''
    print_str_2 = ''
    if include_chars:
        print_str += ' & '.join([str(round(x, 2))+'***' for x in params[:num_fit_chars]]) + ' & '
        print_str_2 += ' & '.join(["{["+str(round(x, 2)) + "]}" for x in t_stats[:num_fit_chars]]) + ' & '
    else:
        print_str += ' & ' * num_fit_chars
        print_str_2 += ' & ' * num_fit_chars
    
    if include_FE:
        print_str += ' T & '
        print_str_2 += ' & '
    else:
        print_str += ' F & '
        print_str_2 += ' & '
    
    if include_last_val:
        last_val_idx = num_fit_chars + num_tgt_chars
        if not include_chars:
            last_val_idx -= num_fit_chars
        if not include_FE:
            last_val_idx -= num_tgt_chars
        print_str += f' {round(params[last_val_idx], 2)} & '
        print_str_2 += "{[" + f' {round(t_stats[last_val_idx], 2)}'+  ']} & '
    else:
        print_str += ' F & '
        print_str_2 += ' & '
        
    if include_missing_gap:
        missing_gap_idx = -1
        
        missing_gap_idx = num_tgt_chars + num_fit_chars + 1
        if not include_chars:
            missing_gap_idx -= num_fit_chars
        if not include_FE:
            missing_gap_idx -= num_tgt_chars
        if not include_last_val:
            missing_gap_idx -= 1
        else:
            missing_gap_idx = num_fit_chars
        print_str += f' {round(params[missing_gap_idx], 2)} & '
        print_str_2 += "{[" + f' {round(t_stats[missing_gap_idx], 2)}'+  ']} & '
    else:
        print_str += ' F & '
        print_str_2 += ' & '
    
    print_str += str(round(train_auc, 2)) + ' & ' + str(round(test_auc, 2)) + '\\\\'
    print_str_2 += ' & \\\\'
    print(print_str)
    ret_data.append(print_str.replace('\\\\', '').split('&'))
    print(print_str_2)
    ret_data.append(print_str_2.replace('\\\\', '').split('&'))

    

class MissingValuesOverTime(SectionTwoPlotBase):
    name = 'MissingValuesOverTime'
    description = ''

    def setup(self, percentile_rank_chars, regular_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates):
        cc = ['B2M', 'OP', 'INV', 'LEV', 'DPI2A']
        fig, axs = plt.subplots(3, 2, figsize=(15, 9))

        # plot1
        dates = [pd.to_datetime(str(int(x))) for x in dates]
        observed_chars = ~np.isnan(regular_chars)
        
        observed_stocks = np.logical_or(~np.isnan(return_panel), np.any(observed_chars, axis=2))
        
        # (a)
        axs[0,0].plot(dates, np.sum(observed_stocks, axis=1), label='All')
        
        for i, c in enumerate(chars):
            if c in cc:
                axs[0,0].plot(dates, np.sum(observed_chars[:,:,i], axis=1), label=c)
        axs[0,0].legend()
        axs[0,0].set_title("Number of Stocks")

        # (b)
        for i, c in enumerate(chars):
            if c in cc:
                axs[0,1].plot(dates, 100 *(1 - np.sum(observed_chars[:,:,i], axis=1) / np.sum(observed_stocks, axis=1)), label=c)
        axs[0,1].legend()
        axs[0,1].set_title("Missing Percentage")


        # (c)
        ME = np.nan_to_num(regular_chars[:,:,np.argwhere(chars == 'ME')[0][0]])
        mkt_cap = np.sum(ME, axis=1)
        
        for i, c in enumerate(chars):
            if c in cc:
                axs[1,0].plot(dates, 100 * (1 - np.sum(observed_chars[:,:,i] * np.nan_to_num(ME), axis=1) / mkt_cap), label=c)
        axs[1,0].legend()
        axs[1,0].set_title("Missing Market Cap.")


        q_chars = [x[0] for x in char_groupings if x[1] == 'Q']
        q_mask = np.isin(chars, q_chars)
        m_chars = [x[0] for x in char_groupings if x[1] == 'M']
        m_mask = np.isin(chars, m_chars)
        
        axs[1,1].plot(dates, 100 *(1 - np.sum(observed_chars[:,:,m_mask], axis=(1,2)) / (len(m_chars) * np.sum(observed_stocks, axis=1))), 
                 label='Monthly - EW')
        
        axs[1,1].plot(dates, 100 *(1 - np.sum(observed_chars[:,:,q_mask], axis=(1,2)) / (len(q_chars) * np.sum(observed_stocks, axis=1))), 
                 label='Quarterly - EW')
        
        axs[1,1].plot(dates, 100 *(1 - np.sum(observed_chars[:,:,m_mask] * np.expand_dims(ME, axis=2), axis=(1,2)) / (len(m_chars) * mkt_cap)), 
                 label='Monthly - VW')
        
        axs[1,1].plot(dates, 100 *(1 - np.sum(observed_chars[:,:,q_mask] * np.expand_dims(ME, axis=2), axis=(1,2)) / (len(q_chars) * mkt_cap)), 
                 label='Quarterly - VW')
        
        axs[1,1].set_title('Quarterly & Monthly')
        axs[1,1].legend()

        me_rank = percentile_rank_chars[:,:,np.argwhere(chars == 'ME')[0][0]]
        bounds = [[-0.5, -0.3], [-0.3, -0.1], [-0.1, 0.1], [0.1, 0.3], [0.3, 0.51]]
        
        for i, (lb, ub) in enumerate(bounds):
            mask = np.logical_and(me_rank >= lb, me_rank < ub)
            axs[2,0].plot(dates, 100 *(1 - np.sum(observed_chars * np.expand_dims(mask, axis=2), axis=(1, 2)) / (45 * np.sum(mask, axis=1))), 
                     label=f"ME{i+1}")
        
        axs[2,0].legend()
        axs[2,0].set_title('Size Quintiles')


        cutoffs = [0, 3, 15, 35]
        num_observed = np.sum(observed_chars, axis=2) * observed_stocks
        
        for c in cutoffs:
            label = '=0' if c == 0 else f"<={c}"
            axs[2,1].plot(dates, 100 * np.sum(num_observed >= 45 - c, axis=1) / np.sum(observed_stocks, axis=1), 
                     label=label)    
        
        axs[2,1].legend()
        axs[2,1].set_title('Multiple Characteristics')
        plt.tight_layout()


class MissingObservationByCharacteristic(SectionTwoPlotBase):
    name = 'MissingObservationByCharacteristic'
    description = ''

    def setup(self, percentile_rank_chars, regular_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates,
             ignore_fully_missing=False, mkt_weight=False, first_mean_permno=False, value_weight=False):
        if first_mean_permno:
            self.name += '_by_permno_first'
        else:
            self.name += '_by_date_first'

        
        
        from imputation_metrics import get_flags
        from data_loading import get_data_dataframe

        def get_missing_percs(df, weight=None):
            weights = df['weights'].T
            df = df.drop(columns='weights')
            df = df.T
            missing_percs = [100 * ((df == -1) * weights).sum(axis=1) / weights.sum(), # start
                         100 * ((df == -2) * weights).sum(axis=1) / weights.sum(), # middle
                         100 * ((df == -3) * weights).sum(axis=1) / weights.sum(), # end
                         100 * ((df == -4) * weights).sum(axis=1) / weights.sum(), # stock missing
                         100 * ((df == -5) * weights).sum(axis=1) / weights.sum(), # totally_missing missing
                        ]
            missing_percs = pd.concat(missing_percs, axis=1)
            missing_percs.columns = ['Start', 'Middle', 'End', 'Stock Miss.', 'Complete']
            return missing_percs

        
        
        flag_panel = get_flags(regular_chars, return_panel)
        
        flags = get_data_dataframe(flag_panel, return_panel, chars, dates, permnos, monthly_updates, None)
        std_data = get_data_dataframe(regular_chars, return_panel, chars, dates, permnos, monthly_updates, None)
        
        flags.date = pd.to_datetime(flags.date.astype(int).astype(str))
        std_data.date = pd.to_datetime(std_data.date.astype(int).astype(str))

        flags = flags.set_index(['permno', 'date'])[chars]
        std_data = std_data.set_index(['permno', 'date'])[chars]
        if value_weight:
            assert not first_mean_permno
            self.name += '_value_weight'
            flags['weights'] = std_data['ME'].fillna(0)
            flags['weights'] = flags['weights'].fillna(0)
        else:
            flags['weights'] = 1
            
        value_weights = regular_chars[:,:,np.argwhere(chars == 'ME')[0][0]]


        if first_mean_permno:
            missing_percs = flags.groupby(level=0).apply(get_missing_percs).groupby(level=1).mean()
        else:
            missing_percs = flags.groupby(level=1).apply(get_missing_percs).groupby(level=1).mean()

        if ignore_fully_missing:
            missing_percs = missing_percs.drop(columns='Complete')

        # if ignore_fully_missing:
        #     flags = flags.loc[(flags != -5).any(axis=1)]

        ordering = [
        'DPI2A',
        'R60_13',
        'HIGH52',
        'AC',
        'OA',
        'SGA2S',
        'FC2Y',
        'D2A',
        'R36_13',
        'BETA_d',
        'INV',
        'NI',
        'OP',
        'RNA',
        'ROA',
        'NOA',
        'LEV',
        'ATO',
        'CTO',
        'ROE',
        'OL',
        'C2A',
        'BETA_m',
        'PCM',
        'PM',
        'PROF',
        'Q',
        'A2ME',
        'AT',
        'CF2B',
        'B2M',
        'S2P',
        'R12_2',
        'E2P',
        'CF2P',
        'R12_7',
        'SUV',
        'TURN',
        'RVAR',
        'VAR',
        'IdioVol',
        'SPREAD',
        'D2P',
        'R2_1',
        'ME',]

        missing_percs = missing_percs.reindex(ordering)
        missing_percs.plot(kind='bar', stacked=True, color=['blue', 'orange', 'red', 'grey', 'black'])
 
        plt.title('Missing Observations by Characteristic')



import seaborn as sns

class MissingObservationBySizeQuintile(SectionTwoPlotBase):
    name = 'MissingValuesBySizeQuintile'
    description = ''

    def setup(self, percentile_rank_chars, regular_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates):
        
        observed_chars = ~np.isnan(regular_chars)
        
        observed_stocks = np.logical_or(~np.isnan(return_panel), np.any(observed_chars, axis=2))
        
        ax = plt.gca()
        
        prop_cycle = ['black', 'blue', 'orange', 'grey', 'red', 'green']
        
        me_rank = np.nanmean(percentile_rank_chars[:,:,np.argwhere(chars == 'ME')[0][0]], axis=0).reshape(1, -1)
        
        
        bounds = [[-0.5, -0.3], [-0.3, -0.1], [-0.1, 0.1], [0.1, 0.3], [0.3, 0.51]]
        
        vals = []
        for i, (lb, ub) in enumerate(bounds):
            mask = np.logical_and(me_rank >= lb, me_rank < ub)
            mask_over_time = np.logical_and(observed_stocks, mask)
            
            vals.append(100 * (1 - np.sum(observed_chars * np.expand_dims(mask, axis=2), axis=(0, 1, 2)) /
                                  (45 * np.sum(mask_over_time, axis=(0, 1)))
                              ))
        
        _=sns.regplot(x=[1,2,3,4,5], y=vals, scatter=False, order=3, ci=False, 
                              line_kws={'lw':3}, ax=ax, color=prop_cycle[0], label='All')
        
        
        cc = ['B2M', 'OP', 'INV', 'LEV', 'DPI2A']
        for j, c in enumerate(cc):
            vals = []
            for i, (lb, ub) in enumerate(bounds):
                c_idx = np.argwhere(chars == c)[0][0]
                mask = np.logical_and(me_rank >= lb, me_rank < ub)
                mask_over_time = np.logical_and(observed_stocks, mask)
                vals.append(100 *(1 - np.sum(observed_chars[:,:,c_idx] * mask, axis=(0, 1)) / (np.sum(mask_over_time, axis=(0, 1)))))
            
            _=sns.regplot(x=[1,2,3,4,5], y=vals, scatter=False, order=3, ci=False, 
                              line_kws={'lw':3}, ax=ax, color=prop_cycle[j+1], label=c)
           
        
        plt.legend()
        plt.title('Size Quintiles')
        
        plt.xticks([1,2,3,4,5])

class MissingObservationByCharacteristicQuintile(SectionTwoPlotBase):
    name = 'MissingValuesByCharQuintile'
    description = ''

    def setup(self, percentile_rank_chars, regular_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates):
        observed_chars = ~np.isnan(regular_chars)
        
        observed_stocks = np.logical_or(~np.isnan(return_panel), np.any(observed_chars, axis=2))
        ax = plt.gca()
        
        bounds = [[-0.5, -0.3], [-0.3, -0.1], [-0.1, 0.1], [0.1, 0.3], [0.3, 0.51]]
        prop_cycle = ['black', 'blue', 'orange', 'grey', 'red', 'green']
        
        cc = ['B2M', 'OP', 'INV', 'LEV', 'DPI2A']
        for j, c in enumerate(cc):
            vals = []
            for i, (lb, ub) in enumerate(bounds):
                c_idx = np.argwhere(chars == c)[0][0]
                c_rank = percentile_rank_chars[:,:,c_idx]
                avg_c_rank = np.nanmean(c_rank, axis=0)
                
                mask = np.logical_and(avg_c_rank >= lb, avg_c_rank < ub)
                mask_over_time = np.logical_and(observed_stocks, avg_c_rank.reshape(1, -1))
                
                vals.append(100 *(1 - np.sum(observed_chars[:,:,c_idx] * mask.reshape(1, -1), axis=(0, 1)) / (np.sum(mask_over_time, axis=(0, 1)))))
        
            _=sns.regplot(x=[1,2,3,4,5], y=vals, scatter=False, order=3, ci=False, 
                              line_kws={'lw':3}, ax=ax, color=prop_cycle[j+1], label=c)
        plt.title('Char Quintiles')
        
        plt.legend()
        
        plt.xticks([1,2,3,4,5])
        


class AutocorrOfChars(SectionTwoPlotBase):
    name = 'AutocorrOfChars'
    description = ''

    def setup(self, percentile_rank_chars, regular_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates):
        def autocorr(char_panel, char_groupings):
            fig = plt.figure(figsize=(15,10))
            ordering = None
        
            T, N, C = char_panel.shape
            char_auto_corrs = []
            for c in tqdm(range(C)):
                auto_corrs = []
                for n in range(N):
                    if np.sum(~np.isnan(char_panel[:,n,c])) >= 60:
                        s = pd.Series(data = char_panel[:,n,c], index=dates)
                        first_idx = s.first_valid_index()
                        last_idx = s.last_valid_index()
                        s = s.loc[first_idx:last_idx]
                        if char_groupings[c][1] == 'M':
                            auto_corrs.append((s.autocorr(lag=1), s.autocorr(lag=12)))
                        else:
                            auto_corrs.append((s.autocorr(lag=3), s.autocorr(lag=12)))
                char_auto_corrs.append(np.nanmean(auto_corrs, axis=0))
            
            if ordering is None:
                ordering = np.argsort(np.array(char_auto_corrs)[:,0])[::-1]
            
            plt.plot(np.arange(45), np.array(char_auto_corrs)[ordering,0], label=f'1 month')
            plt.plot(np.arange(45), np.array(char_auto_corrs)[ordering,1], label=f'12 month')
            
            plt.legend()
            # save_path = f'../images-pdfs/revision/'
            plt.xticks(np.arange(45), chars[ordering], rotation=90, fontsize=15)
            # plt.savefig(save_path + f'auto_corrs.pdf', bbox_inches='tight')
            plt

        autocorr(percentile_rank_chars, char_groupings)



class StdOfChars(SectionTwoPlotBase):
    name = 'StdOfChars'
    description = ''

    def setup(self, percentile_rank_chars, regular_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates):
        def std(char_panel, char_groupings):
            fig = plt.figure(figsize=(15,10))
            ordering = None
        
            T, N, C = char_panel.shape
            stds = []
            for c in tqdm(range(C)):
                auto_corrs = []
                for n in range(N):
                    if np.sum(~np.isnan(char_panel[:,n,c])) >= 60:
                        s = pd.Series(data = char_panel[:,n,c], index=dates)
                        first_idx = s.first_valid_index()
                        last_idx = s.last_valid_index()
                        s = s.loc[first_idx:last_idx]
                        auto_corrs.append(s.std())
                stds.append(np.nanmean(auto_corrs, axis=0))
            
            if ordering is None:
                ordering = np.argsort(np.array(stds))[::-1]
            
            plt.plot(np.arange(45), np.array(stds)[ordering], label=f'Std')
            
            plt.legend()
            plt.xticks(np.arange(45), chars[ordering], rotation=90, fontsize=15)

        std(percentile_rank_chars, char_groupings)

class HeatmatOfCorr(SectionTwoPlotBase):
    name = 'HeatmatOfCorr'
    description = ''

    def setup(self, percentile_rank_chars, regular_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates):
        def nancorr(c1, c2):
            mask = np.logical_and(~np.isnan(c1), ~np.isnan(c2))
            return np.corrcoef(c1[mask], c2[mask])[0, 1]
        def nancorr_mat(m1):
            T, C = m1.shape
            corr_mat = np.zeros((C, C))
            for i in range(C):
                for j in range(i, C):
                    corr_mat[i,j] = corr_mat[j,i] = nancorr(m1[:,i], m1[:,j])
            return corr_mat
        def get_pariwise_corrs(ranked_chars):
            pairwise_corrs = [nancorr_mat(ranked_chars[:,i,:]) for i in tqdm(range(ranked_chars.shape[1]))]
            return np.nanmean(pairwise_corrs, axis=0)
        
        from tqdm.notebook import tqdm
        pairwise_chars = get_pariwise_corrs(percentile_rank_chars[:,:])
        corrs = pd.DataFrame(pairwise_chars, index=chars, columns=chars)
        ordering = [
            'ATO', 'OL',
        'CTO',
        'PROF',
        'D2A',
        'NOA',
        'AC',
        'OA',
        'R2_1',
        'SUV',
        'SPREAD',
        'RVAR',
        'IdioVol',
        'VAR',
        'FC2Y',
        'SGA2S',
        'C2A',
        'BETA_d',
        'BETA_m',
        'TURN',
        'NI',
        'Q',
        'B2M',
        'A2ME',
        'LEV',
        'S2P',
        'AT',
        'ME',
        'E2P',
        'CF2P',
        'D2P',
        'DPI2A',
        'INV',
        'PCM',
        'R12_2',
        'R12_7',
        'HIGH52',
        'OP',
        'PM',
        'CF2B',
        'R36_13',
        'R60_13',
        'ROA',
        'RNA',
        'ROE',
        ]

        import seaborn as sns
        n = 45
        fig, ax = plt.subplots(figsize=(26, 26))
            
        cax = ax.matshow(corrs.loc[ordering, ordering], cmap='RdBu_r',vmax=0.65, vmin=-0.65)
        # #     cax = ax.matshow(corr, cmap=cmap)
        
        # # Major ticks
        ax.set_xticks(np.arange(0, n, 1))
        ax.set_yticks(np.arange(0, n, 1))
        
        # # Labels for major ticks
        ax.set_xticklabels(np.arange(1, n+1, 1))
        ax.set_yticklabels(np.arange(1, n+1, 1))
        
        # # Minor ticks
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.grid(which='major', color='w', linestyle='-', linewidth=0)
        
        
        plt.xticks(range(len(ordering)), ordering, rotation=70, fontsize=33, weight='normal');
        plt.yticks(range(len(ordering)), ordering, rotation=0,fontsize=33, weight='medium');
        
        ax.tick_params(bottom=False, top=True, left=True, right=True)
        ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True)
        
        #     plt.minorticks_off()
        cbar = fig.colorbar(cax, aspect=40, shrink=.5, pad=.072, location='right')
        cbar.ax.tick_params(labelsize=25)
        plt.grid()
        
        ax.set_aspect(1)
        plt.tight_layout()
        #     plt.subplots_adjust(right=1.15, top=2.5, bottom=-.75)
        plt.subplots_adjust(right=1.1)
        
        # sns.heatmap(corrs.loc[ordering, ordering], annot=False)


class HeatmatOfMissingPerc(SectionTwoPlotBase):
    name = 'HeatmatOfMissingPerc'
    description = ''

    def setup(self, percentile_rank_chars, regular_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates):
        observed_chars = ~np.isnan(regular_chars)
        
        observed_stocks = np.logical_or(~np.isnan(return_panel), np.any(observed_chars, axis=2))
        
        missing_t_perc = [
            100 *(1 - np.sum(observed_chars[:,:,i], axis=1) / np.sum(observed_stocks, axis=1))
        for i, c in enumerate(chars)
        ]
        dates = [pd.to_datetime(str(int(x))) for x in dates]
        
        missing_t_perc = pd.DataFrame(missing_t_perc, index=chars, columns=dates)
        
        char_lbl_q = ['A2ME', 'AC', 'AT', 'ATO', 'B2M', 'C2A', 'CF2B', 'CF2P', 'CTO', 'D2A', 'DPI2A', 'E2P', 'FC2Y', 'INV', 'LEV', 'NI', 
              'NOA', 'OA', 'OL', 'OP', 'PCM', 'PM', 'PROF', 'Q', 'RNA', 'ROA', 'ROE', 'S2P', 'SGA2S']
        
        char_lbl_m = [x for x in chars if x not in char_lbl_q]
        
        
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        
        cmp = cm.get_cmap('plasma_r', 100+4)
        newcolors = cmp(np.linspace(0, 1, 100+4))
        newcolors[0]
        yello = [1, 1, .8, 1]
        white = np.array([1,1,1,1])
        newcolors = newcolors[4:]
        
        l = 2
        newcolors[:l, :] = yello #white
        newcmp = ListedColormap(newcolors)
        len(newcolors), 100*l/len(newcolors)

        fig, ax = plt.subplots(2,1, figsize=(22,16))
        plt.grid(False)
        
        # x = missing_t_mean[char_lbl_q].resample('M').mean().T
        x = missing_t_perc.loc[char_lbl_q].sort_index()
        # _=sns.heatmap(missing_t_mean[char_lbl_q].resample('Y').mean().T, cmap="YlGnBu", vmax=1)
        
        cax=ax[0].matshow(x, cmap=newcmp, aspect='auto', vmax=100*.5, vmin=0)
        
        ny, nx = x.shape
        
        # y ticks
        _=ax[0].set_yticks(np.arange(0, ny, 1))
        _=ax[0].set_yticklabels(x.index, fontdict={'fontsize':24})
        
        # x ticks
        xticks = np.arange(3*12, nx, 3*40) # monthly
        # xticks = np.arange(12, nx, 40) # quarterly
        
        # xticks = xticks[:-1]
        xticklabels = [str(i.quarter)+'/'+str(i.year) for i in x.columns[xticks]]
        _=ax[0].set_xticks(xticks)
        _=ax[0].set_xticklabels(xticklabels, fontdict={'fontsize':24})
        _=ax[0].xaxis.set_ticks_position('bottom')
        _=ax[0].tick_params(width=3,  length=15, direction='inout')
        
        # # Minor ticks
        # _=ax.set_xticks([])#(np.arange(-.5, 10, 1), minor=True)
        # _=ax.set_yticks(np.arange(-.5, ny, 1), minor=True)
        
        _=ax[0].set_xticks([], minor=True)
        _=ax[0].set_yticks(np.arange(-.5, ny, 1), minor=True)
        _=ax[0].grid(which='minor', color='w', linestyle='-', linewidth=1.5)
        
        cbar = fig.colorbar(cax, aspect=30, shrink=.85, pad=0.02, format='%2.0f%%')
        # cbar.ax.tick_params(labelsize=17)

        x = missing_t_perc.loc[char_lbl_m].sort_index()
        # _=sns.heatmap(missing_t_mean[char_lbl_q].resample('Y').mean().T, cmap="YlGnBu", vmax=1)
        
        cax=ax[1].matshow(x, cmap=newcmp, aspect='auto', vmax=100*.5, vmin=0)
        
        ny, nx = x.shape
        
        # y ticks
        _=ax[1].set_yticks(np.arange(0, ny, 1))
        _=ax[1].set_yticklabels(x.index, fontdict={'fontsize':24})
        
        # x ticks
        xticks = np.arange(3*12, nx, 3*40) # monthly
        # xticks = np.arange(12, nx, 40) # quarterly
        
        # xticks = xticks[:-1]
        xticklabels = [str(i.quarter)+'/'+str(i.year) for i in x.columns[xticks]]
        _=ax[1].set_xticks(xticks)
        _=ax[1].set_xticklabels(xticklabels, fontdict={'fontsize':24})
        _=ax[1].xaxis.set_ticks_position('bottom')
        _=ax[1].tick_params(width=3,  length=15, direction='inout')
        
        # # Minor ticks
        # _=ax.set_xticks([])#(np.arange(-.5, 10, 1), minor=True)
        # _=ax.set_yticks(np.arange(-.5, ny, 1), minor=True)
        
        _=ax[1].set_xticks([], minor=True)
        _=ax[1].set_yticks(np.arange(-.5, ny, 1), minor=True)
        _=ax[1].grid(which='minor', color='w', linestyle='-', linewidth=1.5)
        
        cbar = fig.colorbar(cax, aspect=30, shrink=.85, pad=0.02, format='%2.0f%%')

        
        # plt.minorticks_off()
        dpi = 60
        plt.gcf().set_dpi(dpi)
        plt.tight_layout()
        # plt.savefig(DIR+'missing_t_heatmap_m.pdf')

class MissingLogitRegressions(SectionTwoTableBase):
    
    name = 'MissingLogitRegressions'
    index = False
    description = ''
    sigfigs=10
    
    def setup(self, percentile_rank_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates):
        result_data = []
        result_index = []
        result_cols = []

        tgt_chars = ['ME', 'R2_1', 'D2P', 'IdioVol', 'TURN', 'SPREAD', 'VAR']
        exl_chars = [ 'RVAR']
        regr_chars = np.logical_and(~np.isin(chars, tgt_chars),
                                ~np.isin(chars, exl_chars))
        tgt_char_mask = np.isin(chars, tgt_chars)
        exl_char_mask = np.isin(chars, exl_chars)
        np.sum(tgt_char_mask)

        tover2 = int(percentile_rank_chars.shape[0]/2)
        t_over_4 = int(tover2 / 2)
        char_present_filter = np.all(~np.isnan(percentile_rank_chars[:,:,tgt_char_mask]), axis=2)

        start_train_aocs, start_test_aocs = [], []
        start_agg_train_aocs, start_agg_test_aocs = [], []
        start_param_values, start_t_stats = [], []
        start_std_errs = []
        start_p_values = []

        filter_too_long_gaps = True
        char_present_filter = np.all(~np.isnan(percentile_rank_chars[:,:,tgt_char_mask]), axis=2)
        input_filter = np.zeros_like(percentile_rank_chars, dtype=bool)
        input_filter[:,:,regr_chars] = 1
        input_filter[~char_present_filter,:] = 0
        input_filter[:t_over_4,:] = 0
        input_filter[0,:,:] = 0


        not_start = np.any(~np.isnan(percentile_rank_chars[:t_over_4]), axis=0)
        for t in range(t_over_4, percentile_rank_chars.shape[0]):
            input_filter[t, not_start] = 0
            not_start = np.logical_or(not_start, ~np.isnan(percentile_rank_chars[t]))

        train_input_filter = input_filter[:tover2]
        test_input_filter = input_filter[tover2:]
            
        missing_gap = np.zeros_like(input_filter, dtype=float)
        missing_gap[:, :, :] = np.nan
        first_occ = np.argmax(np.any(~np.isnan(percentile_rank_chars), axis=2), axis=0)
        for t in range(t_over_4, percentile_rank_chars.shape[0]):
            for c in range(input_filter.shape[2]):
                missing_gap[t, input_filter[t, :, c], c] = t - first_occ[input_filter[t, :, c]]
                if filter_too_long_gaps:
                    input_filter[:t-10, input_filter[t, :, c], c] = 0

        configs = [(True, False, False), (True, False, True), (False, True, False), (False, True, True), 
                      (True, True, True)]
        
        for (include_chars, include_FE, include_missing_gap) in tqdm(configs):
            X, Y, idxs, feature_names = logit_models_and_masking.get_pooled_x_y_from_panel(percentile_rank_chars[:tover2], 
                                       train_input_filter[:tover2], 
                                        chars,
                                        tgt_char_mask, 
                                        exl_char_mask,
                                        factors=None, 
                                        include_chars=include_chars,
                                        include_factors=False,
                                        include_FE=include_FE,
                                        include_last_val=False,
                                        switch=False,
                                        include_missing_gap=include_missing_gap,
                                        missing_gaps=missing_gap[:tover2])

            logit_model = sm.Logit(Y, X) #Create model instance
            result_start = logit_model.fit(method = "newton", maxiter=50, disp=False,
                                        kwargs={"tol":1e-8}) #Fit model, 0.652114
            train_fpr, train_tpr, _ = metrics.roc_curve(Y, result_start.predict(X))
            start_agg_train_aocs.append(metrics.auc(train_fpr, train_tpr))

            test_input_filter[0] = 0

            X, Y, idxs, feature_names = logit_models_and_masking.get_pooled_x_y_from_panel(percentile_rank_chars[tover2:], 
                                        test_input_filter, 
                                        chars,
                                        tgt_char_mask, 
                                        exl_char_mask,
                                        factors=None, 
                                        include_chars=include_chars,
                                        include_factors=False,
                                        include_FE=include_FE,
                                        include_last_val=False,
                                        switch=False,
                                        include_missing_gap=include_missing_gap,
                                        missing_gaps=missing_gap[tover2:])


            test_fpr, test_tpr, _ = metrics.roc_curve(Y, result_start.predict(X))

            start_agg_test_aocs.append(metrics.auc(test_fpr, test_tpr))

            start_std_errs.append(result_start.bse)
            start_t_stats.append(result_start.tvalues)
            start_p_values.append(result_start.pvalues)

            start_param_values.append(result_start.params)


        col_names = list(sorted(tgt_chars)) + \
                ['FE included', 'Last Val Indicator', "Missing Gap"] + \
                ["agg train AOC", "agg  test AOC"]
        print(' & '.join(col_names) + '\\\\')

        
        for c, params, t_stats, train_auc, test_auc in zip(configs, start_param_values, start_t_stats,
                                                start_agg_train_aocs, start_agg_test_aocs):
            include_chars, include_FE, include_missing_gap = c
            include_last_val = False

            print_results(result_data, include_chars, include_FE, include_last_val, include_missing_gap, params, t_stats,
                        train_auc, test_auc, tgt_chars=tgt_chars, num_fit_chars=7, num_tgt_chars=37)

        # middle
        middle_train_aocs, middle_test_aocs = [], []
        middle_agg_train_aocs, middle_agg_test_aocs = [], []
        middle_param_values, middle_t_stats = [], []
        middle_std_errs = []
        middle_p_values = []

        char_present_filter = np.all(~np.isnan(percentile_rank_chars[:,:,tgt_char_mask]), axis=2)
        char_present_filter = np.all(~np.isnan(percentile_rank_chars[:,:,tgt_char_mask]), axis=2)
        input_filter = np.zeros_like(percentile_rank_chars, dtype=bool)
        input_filter[:,:,regr_chars] = 1
        input_filter[~char_present_filter,:] = 0
        input_filter[:t_over_4,:] = 0
        input_filter[0,:,:] = 0


        not_start = np.any(~np.isnan(percentile_rank_chars[:t_over_4]), axis=0)
        for t in range(t_over_4, percentile_rank_chars.shape[0]):
            input_filter[t, ~not_start] = 0
            not_start = np.logical_or(not_start, ~np.isnan(percentile_rank_chars[t]))

        curr_missing_gap = np.zeros(input_filter.shape[1:], dtype=int)
        missing_gap = np.zeros_like(input_filter, dtype=int)

        for t in range(0, tover2):
            if t > t_over_4:            
                for c in range(input_filter.shape[2]):
                    missing_gap[t, input_filter[t, :, c], c] = curr_missing_gap[input_filter[t, :, c], c]
                
            curr_missing_gap += 1
            curr_missing_gap[~np.isnan(percentile_rank_chars[t])] = 0
            
        for i, c in enumerate(chars):
            if char_map[c] != "M": 
                input_filter[:,:,i] = np.logical_and(input_filter[:,:,i], monthly_updates)
        configs = [(True, False, False, False),
                                                                                (False, True, False, False),
                                                                                (True, False, True, False),
                                                                                (False, True, True, False),
                                                                                (False, True, True, True),
                                                                                (True, True, True, True)]
        for (include_chars, include_FE, include_last_val, include_missing_gap) in tqdm(configs):
            X, Y, idxs, feature_names = logit_models_and_masking.get_pooled_x_y_from_panel(percentile_rank_chars[:tover2], 
                                        input_filter[:tover2], 
                                        chars,
                                        tgt_char_mask, 
                                        exl_char_mask,
                                        factors=None, 
                                        include_chars=include_chars,
                                        include_factors=False,
                                        include_FE=include_FE,
                                        include_last_val=include_last_val,
                                        switch=False,
                                        include_missing_gap=include_missing_gap,
                                        missing_gaps=missing_gap[:tover2])

            logit_model = sm.Logit(Y, X) #Create model instance
            result_middle = logit_model.fit(method = "newton", maxiter=50, disp=False,
                                        kwargs={"tol":1e-8}) #Fit model
            train_fpr, train_tpr, _ = metrics.roc_curve(Y, result_middle.predict(X))
            

            middle_agg_train_aocs.append(metrics.auc(train_fpr, train_tpr))

            input_filter[tover2] = 0
            X, Y, idxs, feature_names = logit_models_and_masking.get_pooled_x_y_from_panel(percentile_rank_chars[tover2:], 
                                        input_filter[tover2:], 
                                        chars,
                                        tgt_char_mask, 
                                        exl_char_mask,
                                        factors=None, 
                                        include_chars=include_chars,
                                        include_factors=False,
                                        include_FE=include_FE,
                                        include_last_val=include_last_val,
                                        switch=False,
                                        include_missing_gap=include_missing_gap,
                                        missing_gaps=missing_gap[tover2:])


            test_fpr, test_tpr, _ = metrics.roc_curve(Y, result_middle.predict(X))
            
            middle_agg_test_aocs.append(metrics.auc(test_fpr, test_tpr))

            middle_std_errs.append(result_middle.bse)
            middle_t_stats.append(result_middle.tvalues)
            middle_p_values.append(result_middle.pvalues)

            middle_param_values.append(result_middle.params)
        
        col_names = list(sorted(tgt_chars)) + \
                ['FE included', 'Last Val Indicator', "Missing Gap"] + \
                ["agg train AOC", "agg  test AOC"]
        print(' & '.join(col_names) + '\\\\')

        
        for c, params, t_stats, train_auc, test_auc in zip(configs, middle_param_values, middle_t_stats,
                                                middle_agg_train_aocs, middle_agg_test_aocs):
            
            include_chars, include_FE, include_last_val, include_missing_gap = c
            print_results(result_data, include_chars, include_FE, include_last_val, include_missing_gap, params, t_stats,
                        train_auc, test_auc, tgt_chars=tgt_chars, num_fit_chars=7, num_tgt_chars=37)


        # end
        end_train_aocs, end_test_aocs = [], []
        end_agg_train_aocs, end_agg_test_aocs = [], []
        end_param_values, end_t_stats = [], []
        end_std_errs = []
        end_p_values = []

        input_filter = np.zeros_like(percentile_rank_chars, dtype=bool)
        input_filter[:,:,regr_chars] = 1
        input_filter[~char_present_filter,:] = 0
        input_filter[:t_over_4,:] = 0
        input_filter[0,:,:] = 0

        for t in tqdm(range(t_over_4, percentile_rank_chars.shape[0])):
            last_gap = np.sum(np.isnan(percentile_rank_chars[t:-1]) != np.isnan(percentile_rank_chars[t+1:]), axis=0) <= 1
            input_filter[t, ~last_gap] = 0

        configs = [(True, False), (False, True), (True, True)]
        for include_chars, include_FE in configs:
            X, Y, idxs, feature_names = logit_models_and_masking.get_pooled_x_y_from_panel(percentile_rank_chars[:tover2], 
                                        input_filter[:tover2], 
                                        chars,
                                        tgt_char_mask, 
                                        exl_char_mask,
                                        factors=None, 
                                        include_chars=include_chars,
                                        include_factors=False,
                                        include_FE=include_FE,
                                        include_last_val=False,
                                        switch=False,
                                        include_missing_gap=False,
                                        missing_gaps=None)


            logit_model = sm.Logit(Y, X, disp=False) #Create model instance
            result_end = logit_model.fit(method = "newton", maxiter=50, disp=False) #Fit model
            train_fpr, train_tpr, _ = metrics.roc_curve(Y, result_end.predict(X))
            end_agg_train_aocs.append(metrics.auc(train_fpr, train_tpr))

            input_filter[tover2] = 0
            X, Y, idxs, feature_names = logit_models_and_masking.get_pooled_x_y_from_panel(percentile_rank_chars[tover2:], 
                                        input_filter[tover2:], 
                                        chars, 
                                        tgt_char_mask, 
                                        exl_char_mask,
                                        factors=None, 
                                        include_chars=include_chars,
                                        include_factors=False,
                                        include_FE=include_FE,
                                        include_last_val=False,
                                        switch=False,
                                        include_missing_gap=False,
                                        missing_gaps=None)


            test_fpr, test_tpr, _ = metrics.roc_curve(Y, result_end.predict(X))
            # print(metrics.auc(test_fpr, test_tpr))
            end_agg_test_aocs.append(metrics.auc(test_fpr, test_tpr))

            end_std_errs.append(result_end.bse)
            end_t_stats.append(result_end.tvalues)
            end_p_values.append(result_end.pvalues)

            end_param_values.append(result_end.params)


        col_names = list(sorted(tgt_chars)) + \
                ['FE included', 'Last Val Indicator', "Missing Gap"] + \
                ["agg train AOC", "agg  test AOC"]
        print(' & '.join(col_names) + '\\\\')

        
        for c, params, t_stats, train_auc, test_auc in zip(configs, end_param_values, end_t_stats,
                                                end_agg_train_aocs, end_agg_test_aocs):
            include_chars, include_FE = c
            include_last_val, include_missing_gap = False, False
            print_results(result_data, include_chars, include_FE, include_last_val, include_missing_gap, params, t_stats,
                        train_auc, test_auc, tgt_chars=tgt_chars, num_fit_chars=7, num_tgt_chars=37)


            
        result_index = np.arange(len(result_data))
        result_cols = col_names

        self.data_df = pd.DataFrame(data=result_data, index=result_index, columns=result_cols)
        

class MssingBlockLengths(SectionTwoTableBase):
    
    name = 'MssingBlockLengths'
    description = ''
    sigfigs=10
    
    def setup(self, percentile_rank_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates):
        missing_length = np.ones(percentile_rank_chars.shape[1:]) * -1
        missing_gaps = [[] for _ in chars]
        prev_obs = ~np.isnan(percentile_rank_chars[0])
        for t in tqdm(range(1, percentile_rank_chars.shape[0])):
            present_at_t = ~np.isnan(percentile_rank_chars[t])
            to_add = np.logical_and(present_at_t, missing_length > 0)
            for n,c in np.argwhere(to_add):
                missing_gaps[c].append(missing_length[n,c])
            missing_length[present_at_t] = 0
            missing_length[np.logical_and(prev_obs, ~present_at_t)] += 1
            prev_obs = np.logical_or(prev_obs, present_at_t)

        data = [(len(x), np.mean(x), np.median(x)) for x in missing_gaps]
        self.data_df = pd.DataFrame(data=data, index=chars, columns=['number of gaps', 'mean length', 'median length'])
        
        
class MssingByQuintile(SectionTwoTableBase):
    
    name = 'MssingByQuintile'
    description = ''
    sigfigs=10
    
    def setup(self, percentile_rank_chars, return_panel, dates, permnos, chars, char_groupings, monthly_updates):
        columns = ['ALL', 'ME Quintile 1', 'ME Quintile 2', 'ME Quintile 3', 'ME Quintile 4', 'ME Quintile 5', 
                   'Char Quintile 1', 'Char Quintile 2', 'Char Quintile 3', 'Char Quintile 4', 'Char Quintile 5'] 
        present = ~np.isnan(return_panel)
        missing = np.logical_and(np.isnan(percentile_rank_chars), np.expand_dims(present, axis=2))
        me_ind = np.argwhere(chars == 'ME')[0][0]
        size_data = percentile_rank_chars[:,:,me_ind]
        ret_data = [[] for _ in chars]
        for i, c in enumerate(chars):
            ret_data[i].append(np.sum(missing[:,:,i]) / np.sum(present))
        
        for i in range(5):
            size_mask = np.logical_and(size_data >= -0.5 + i * 0.2, size_data < -0.5 + (i+1) * 0.2)
            for j, c in enumerate(chars):
                ret_data[j].append(np.sum(missing[size_mask][:,j]) / np.sum(present[size_mask]))
            
        for i in range(5):
            for j, c in enumerate(chars):
                char_mask = np.logical_and(size_data >= -0.5 + i * 0.2, size_data < -0.5 + (i+1) * 0.2)
                ret_data[j].append(np.sum(missing[char_mask][:,j]) / np.sum(present[char_mask]))
                
        self.data_df = pd.DataFrame(data=ret_data, index=chars, columns=columns)