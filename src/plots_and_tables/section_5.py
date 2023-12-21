import os

from plots_and_tables import plot_base
import missing_data_utils
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm 
import imputation_utils,logit_models_and_masking, imputation_metrics, imputation_model_simplified
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
from matplotlib import patches
from multiprocessing import Pool
from joblib import Parallel, delayed

class SectionFivePlotBase(plot_base.PaperPlot, ABC):
    section = 'section5'

class SectionFiveTableBase(plot_base.PaperTable, ABC):
    section = 'section5'

class ExampleImputations(SectionFivePlotBase):
    
    name = 'ExampleImputations'
    description = ''
    
    def setup(self, percentile_rank_chars, return_panel, dates, permnos, chars, char_groupings):
        self.char_groupings = char_groupings
        self.return_panel = return_panel
        self.permnos = permnos
        self.chars = chars
        self.percentile_rank_chars = percentile_rank_chars
        self.date_vals = np.array(dates) // 10000 + ((np.array(dates) // 100) % 100) / 12
        tickers_to_permnos = pd.read_csv('ticker_to_permno.csv')
        tickers_to_permnos = tickers_to_permnos[tickers_to_permnos['keep'] == 'X'][['Company', 'Symbol', 'Permno']]
        self.T = percentile_rank_chars.shape[0]
        mask_windows_2yr = [(1985, 1987),
                        (1990, 1992),
                        (1995, 1997),
                        (2000, 2003),
                        (2005, 2007),
                        (2010, 2013),
                        (2015, 2017)]

        self.mask_windows_1yr = [(1985, 1986),
                        (1990, 1991),
                        (1995, 1996),
                        (2000, 2001),
                        (2005, 2006),
                        (2010, 2011),
                        (2015, 2016),
                        (2017, 2018)]

        one_yr_mask = np.copy(self.date_vals == 0)
        two_yr_mask = np.copy(self.date_vals == 0)

        for start, end in self.mask_windows_1yr:
            one_yr_mask = np.logical_or(one_yr_mask, np.logical_and(self.date_vals >= start, self.date_vals <= end))

        for start, end in mask_windows_2yr:
            two_yr_mask = np.logical_or(two_yr_mask, np.logical_and(self.date_vals >= start, self.date_vals <= end))

        tickers_to_permnos = tickers_to_permnos[tickers_to_permnos['Symbol'].isin(['MSFT', 'HAS'])]
        self.tickers_to_permnos = tickers_to_permnos
        mask_permnos = tickers_to_permnos['Permno'].to_numpy()
        permno_mask = np.isin(permnos, mask_permnos)

        self.tgt_chars = ["AT", "ME", "Q", 'VAR']
        self.char_mask = np.isin(chars, self.tgt_chars)
        self.one_yr_masked_data = np.copy(percentile_rank_chars)
        self.two_yr_masked_data = np.copy(percentile_rank_chars)
        for t in np.argwhere(one_yr_mask):
            for p in np.argwhere(permno_mask):
                for c in np.argwhere(self.char_mask):
                    self.one_yr_masked_data[t,p,c] = np.nan

        for t in np.argwhere(two_yr_mask):
            for p in np.argwhere(permno_mask):
                for c in np.argwhere(self.char_mask):
                    self.two_yr_masked_data[t,p,c] = np.nan

        
    def run(self):
        T = self.T
        for masked_data, data_plot_name, start in [
            (self.one_yr_masked_data, 'one-year-mask', 0),
        ]:
            end = start + masked_data.shape[0]
            
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=masked_data, 
                return_panel=self.return_panel, min_chars=1, K=20, 
                num_months_train=T,
                reg=0.01,
                time_varying_lambdas=False,
                window_size=548, 
                n_iter=1,
                eval_data=None,
                allow_mean=False)

            imputed = np.concatenate([np.expand_dims(x @ lmbda.T, axis=0) for x in gamma_ts])
            resids = masked_data - imputed
            global_bw = imputation_model_simplified.impute_chars(char_data=masked_data,
                           imputed_chars=imputed, residuals=resids, 
                           suff_stat_method='last_val', constant_beta=True, beta_weight=False)

            global_fwbw = imputation_model_simplified.impute_chars(char_data=masked_data,
                           imputed_chars=imputed, residuals=resids, 
                           suff_stat_method='fwbw', constant_beta=True, beta_weight=False)

            global_xs = imputed

            for company_ticker, tgt_permno in self.tickers_to_permnos[['Symbol', 'Permno']].to_numpy():
                save_path = f'../images-pdfs/section5/'
                tgt_permno_mask = self.permnos == tgt_permno

                
                char_bound_edits = {
                    'AT': [-0.05, 0.55],
                    'Q': [-0.05, 0.55],
                    'VAR': [-0.55, 0.15],
                }
                for char in self.tgt_chars:
                    
                    if char in char_bound_edits:
                        lb, ub = char_bound_edits[char]
                    else:
                        lb, ub = [-0.55, 0.55]
                    
                    tgt_char_mask = self.chars == char
                    char_ind = np.argwhere(tgt_char_mask)[0][0]
                    fig = plt.figure(figsize=(10,5))
                    ax = plt.gca()
                    for start_mask, end_mask in self.mask_windows_1yr:
                        if np.any(self.date_vals == start_mask):
                            start_date_ind = np.argwhere(self.date_vals == start_mask)[0][0]
                            end_date_ind = np.argwhere(self.date_vals == end_mask)[0][0]
                            if start_date_ind >= start:
                                rectangle = patches.Rectangle((self.date_vals[start_date_ind], -1), 
                                                              self.date_vals[end_date_ind] - self.date_vals[start_date_ind], 
                                                              2, facecolor="grey", alpha=0.25)
                                ax.add_patch(rectangle)

                            elif end_date_ind >= start:
                                rectangle = patches.Rectangle((self.date_vals[start], -1), 
                                                              self.date_vals[end_date_ind] - self.date_vals[start], 
                                                              2, facecolor="grey", alpha=0.25)
                                ax.add_patch(rectangle)
                    

                    plt.plot(self.date_vals[start:end], self.percentile_rank_chars[start:end,tgt_permno_mask,char_ind], label='masked')
                    plt.plot(self.date_vals[start:end], masked_data[start:end,tgt_permno_mask,char_ind], label='observed')
                    plt.plot(self.date_vals[start:end], global_bw[start:end,tgt_permno_mask,char_ind], label='imputed-B-XS')
                    plt.plot(self.date_vals[start:end], 0*global_bw[start:end,tgt_permno_mask,char_ind], label='imputed-median')
                    plt.plot(self.date_vals[start:end], global_fwbw[start:end,tgt_permno_mask,char_ind], label='imputed-BF-XS')
#                     plt.title(char)
                    plt.legend(fontsize=20, loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=3)
                    plt.gcf().subplots_adjust(bottom=0.25)
                    plt.ylim(lb, ub)
                    plt.savefig(save_path + f'{company_ticker}-{data_plot_name}-{char}.pdf', bbox_inches='tight')
                    plt.show()
                    fig = plt.figure(figsize=(10,5))
                    ax = plt.gca()
                    for start_mask, end_mask in self.mask_windows_1yr:
                        if np.any(self.date_vals == start_mask):
                            start_date_ind = np.argwhere(self.date_vals == start_mask)[0][0]
                            end_date_ind = np.argwhere(self.date_vals == end_mask)[0][0]
                            if start_date_ind >= start:
                                rectangle = patches.Rectangle((self.date_vals[start_date_ind], -1), 
                                                              self.date_vals[end_date_ind] - self.date_vals[start_date_ind], 
                                                              2, facecolor="grey", alpha=0.25)
                                ax.add_patch(rectangle)

                            elif end_date_ind >= start:
                                rectangle = patches.Rectangle((self.date_vals[start], -1), 
                                                              self.date_vals[end_date_ind] - self.date_vals[start], 
                                                              2, facecolor="grey", alpha=0.25)
                                ax.add_patch(rectangle)

                    plt.plot(self.date_vals[start:end], 
                             self.percentile_rank_chars[start:end,tgt_permno_mask,char_ind], label='masked')
                    plt.plot(self.date_vals[start:end], masked_data[start:end,tgt_permno_mask,char_ind], label='observed')
                    plt.plot(self.date_vals[start:end], global_bw[start:end,tgt_permno_mask,char_ind], label='imputed-B-XS')
                    plt.plot(self.date_vals[start:end], global_xs[start:end,tgt_permno_mask,char_ind], label='imputed-XS')
#                     plt.title(char)
                    plt.legend(fontsize=20, loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=3)
                    plt.gcf().subplots_adjust(bottom=0.25)
                    plt.ylim(lb, ub)
                    plt.savefig(save_path + f'{company_ticker}-{data_plot_name}-{char}-xs_only.pdf', bbox_inches='tight')
                    plt.show()

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


class AggregateXSImputationErrors(SectionFiveTableBase):
    name = 'XSImputationErrors'
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    
    def setup(self, percentile_rank_chars, chars, monthly_updates, char_groupings):
        fp_mask = np.all(~np.isnan(percentile_rank_chars), axis=2)
        file_names = ['global_fwbw', 'global_fw', 'global_bw', 'global_xs',
                     'global_ts', 'local_bw', 'local_xs', 'prev_val', 'local_ts',
                     'xs_median', 'indus_median']
        plot_names = ["global BF-XS", "global F-XS", "global B-XS", "global XS",
                      "global B",
                        "local B-XS", "local XS",  "prev", "local B",
                      "XS-median", "ind-median"]
        data = []
        columns = []
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_logit': "logit_eval_data",
            '_out_of_sample_block': 'prob_block_eval_data',
        }
        fit_maps = {
            '_out_of_sample_MAR': "MAR_fit_data",
            '_out_of_sample_logit': "logit_fit_data",
            '_out_of_sample_block': 'prob_block_fit_data',
        }
        tags = ['_in_sample', '_out_of_sample_MAR', 
                '_out_of_sample_block', 
                '_out_of_sample_logit']
        import os
        for tag in tags:
            if tag != '_in_sample':
                assert os.path.exists('../data/imputation_cache/' + eval_maps[tag] + '.npz'), \
                    '../data/imputation_cache/' + eval_maps[tag] + '.npz' + ' does not exist'
            for fname in file_names:
                assert os.path.exists('../data/imputation_cache/' + fname + tag + '.npz'), \
                    fname + tag + '.npz' + ' does not exist'
        
        for tag in tags:
            if tag == '_in_sample':
                eval_chars = update_chars
            else:
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
            eval_chars[fp_mask,:] = np.nan
            
            def get_rmse(pred, gt):
                diff = np.square(pred - gt)
                q_mask = np.isin(chars, [x for x in chars if char_groupings[x] != 'M'])
                m_mask = np.isin(chars, [x for x in chars if char_groupings[x] == 'M'])
                m_rmse = np.sqrt(np.nanmean(np.nanmean(np.nanmean(diff[:,:,m_mask], axis=2), axis=0), axis=0))
                q_rmse = np.sqrt(np.nanmean(np.nanmean(np.nanmean(diff[:,:,q_mask], axis=2), axis=0), axis=0))
                agg_rmse = np.sqrt(np.nanmean(np.nanmean(np.nanmean(diff, axis=2), axis=0), axis=0))
                return agg_rmse, q_rmse, m_rmse
            
            metrics = [get_rmse(imputation_utils.load_imputation(fname + tag), eval_chars) for fname in file_names]

            data.append([[round(x, 5) for x in y] for y in metrics])
            columns += [x + tag for x in ['aggregate', "quarterly", 'monthly']]
        self.data_df = pd.DataFrame(data=np.concatenate(data, axis=1), index=plot_names, columns=columns)

        
class AggregateImputationErrorsFullDataset(SectionFiveTableBase):
    name = 'AggregateImputationErrorsFullDataset'
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    
    def setup(self, percentile_rank_chars, chars, monthly_updates):
        
        subsets = [
            ['global_fwbw', 'global_bw', 'global_xs'],
            ['global_bw', 'global_xs'],
            ['global_xs'],
            ["global_ts", 'xs_median'], 
            ['local_bw', 'local_xs'],
            ['local_xs'],
            ["local_ts", 'xs_median'], 
            ["prev_val", 'xs_median'], 
            ['xs_median'], 
        ]
        
        plot_names = [
            "global BF-XS, B-XS, XS",
                      "global B-XS, XS",
                      "global XS",
                      "global B, median",
                      "local B-XS, XS",
                      "local XS",
                      "local B, median",
                      "prev val, median",
                      "median",
        ]
        
        
        data = []
        columns = []
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_logit': "logit_eval_data",
            '_out_of_sample_block': 'prob_block_eval_data',
        }
        fit_maps = {
            '_out_of_sample_MAR': "MAR_fit_data",
            '_out_of_sample_logit': "logit_fit_data",
            '_out_of_sample_block': 'prob_block_fit_data',
        }
        tags = ['_in_sample', '_out_of_sample_MAR', 
                '_out_of_sample_block', 
                '_out_of_sample_logit']
        import os
        
        
        def run_tag(tag):
            if tag == '_in_sample':
                eval_chars = np.copy(update_chars)
                lt_10_mask = np.sum(~np.isnan(percentile_rank_chars), axis=2) < 1
            else:
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                lt_10_mask = np.sum(~np.isnan(imputation_utils.load_imputation(fit_maps[tag])), axis=2) < 1
            
            eval_chars[lt_10_mask,:] = np.nan
            def func_call(group, eval_data, groups):
                data = None
                overall_mask = ~np.isnan(eval_data)
                for fname in group:
                    if data is None:
                        data = imputation_utils.load_imputation(fname + tag)
                    else:
                        mask = np.isnan(data)
                        data[mask] = imputation_utils.load_imputation(fname + tag)[mask]
                        
                return imputation_utils.get_imputation_metrics(data, 
                                      eval_char_data=eval_data, monthly_update_mask=None, 
                                      char_groupings=groups, norm_func=None)

            metrics = [func_call(group, eval_chars, char_groupings) for group in subsets]
            ret_data = [[round(self.norm_func(np.nanmean(x)), 5) for x in y] for y in metrics]
            ret_cols = [x + tag for x in ['aggregate', "quarterly", 'monthly']]
            return ret_data, ret_cols
        
        results = list(Parallel(n_jobs=4)(delayed(run_tag)(tag) for tag in tags))
        data = [x[0] for x in results]
        columns = sum([x[1] for x in results], [])
        self.data_df = pd.DataFrame(data=np.concatenate(data, axis=1), index=plot_names, columns=columns)

        
class AggregateImputationErrors(SectionFiveTableBase):
    name = 'ImputationErrors'
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    
    def setup(self, percentile_rank_chars, chars, monthly_updates):
        file_names = [
            'global_fwbw', 
                      'global_fw',
                      'global_bw',
                      'global_xs',
                      'global_ts', 
                      'local_bw',
                      'local_xs', 
                      'prev_val',
                      'local_ts',
                      'xs_median', 
                      'indus_median',
                     ]
        plot_names = [
            "global BF-XS",
            "global F-XS",
            "global B-XS",
            "global XS",
            "global B",
            "local B-XS",
            "local XS",  
            "prev", 
            "local B",
            "XS-median",
            "ind-median",
        ]
        
        
        data = []
        columns = []
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_logit': "logit_eval_data",
            '_out_of_sample_block': 'prob_block_eval_data',
        }
        fit_maps = {
            '_out_of_sample_MAR': "MAR_fit_data",
            '_out_of_sample_logit': "logit_fit_data",
            '_out_of_sample_block': 'prob_block_fit_data',
        }
        tags = ['_in_sample', 
                '_out_of_sample_MAR', 
                '_out_of_sample_block', 
                '_out_of_sample_logit'
               ]
        import os
        
        def run_tag(tag):
            if tag == '_in_sample':
                eval_chars = np.copy(update_chars)
                lt_10_mask = np.sum(np.isnan(percentile_rank_chars), axis=2) > 35
            else:
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                lt_10_mask = np.sum(np.isnan(imputation_utils.load_imputation(fit_maps[tag])), axis=2) > 35


            def func_call(fname, eval_data, groups):
                return imputation_utils.get_imputation_metrics(imputation_utils.load_imputation(fname + tag), 
                                      eval_char_data=eval_data, monthly_update_mask=None, 
                                      char_groupings=groups, norm_func=None)    
            metrics = [func_call(i, eval_chars, char_groupings) for i in file_names]

            ret_data = [[round(self.norm_func(np.nanmean(x)), 5) for x in y] for y in metrics]
            ret_cols = [x + tag for x in ['aggregate', "quarterly", 'monthly']]
            return ret_data, ret_cols
            
        results = list(Parallel(n_jobs=4)(delayed(run_tag)(tag) for tag in tags))
        data = [x[0] for x in results]
        columns = sum([x[1] for x in results], [])
        self.data_df = pd.DataFrame(data=np.concatenate(data, axis=1), index=plot_names, columns=columns)


        
class AggregateImputationR2FullDataset(SectionFiveTableBase):
    name = 'AggregateImputationR2FullDataset'
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    
    def setup(self, percentile_rank_chars, chars, monthly_updates):
        subsets = [
            ['global_fwbw', 'global_bw', 'global_xs'],
            ['global_bw', 'global_xs'],
            ['global_xs'],
            ["global_ts", 'xs_median'], 
            ['local_bw', 'local_xs'],
            ['local_xs'],
            ["local_ts", 'xs_median'], 
            ["prev_val", 'xs_median'], 
            ['xs_median'], 
        ]
        
        plot_names = [
            "global BF-XS, B-XS, XS",
                      "global B-XS, XS",
                      "global XS",
                      "global B, median",
                      "local B-XS, XS",
                      "local XS",
                      "local B, median",
                      "prev val, median",
                      "median"
        ]
        

        
        data = []
        columns = []
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_logit': "logit_eval_data",
            '_out_of_sample_block': 'prob_block_eval_data',
        }
        fit_maps = {
            '_out_of_sample_MAR': "MAR_fit_data",
            '_out_of_sample_logit': "logit_fit_data",
            '_out_of_sample_block': 'prob_block_fit_data',
        }
        tags = ['_in_sample', '_out_of_sample_MAR', 
                '_out_of_sample_block', 
                '_out_of_sample_logit']

        
        monthly_chars = [x for x in chars if char_map[x] == 'M']
        monthly_char_mask = np.isin(chars, monthly_chars)
        quarterly_char_mask = ~monthly_char_mask
        
        def run_tag(tag):
            if tag == '_in_sample':
                eval_chars =  np.copy(update_chars)
                lt_10_mask = np.sum(np.isnan(percentile_rank_chars), axis=2) > 35
            else:
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                lt_10_mask = np.sum(np.isnan(imputation_utils.load_imputation(fit_maps[tag])), axis=2) > 35
            eval_chars[lt_10_mask] = np.nan
            def get_r2(tgt, fnames, monthly_char_mask, quarterly_char_mask):
                data = None
                for fname in fnames:
                    if data is None:
                        data = imputation_utils.load_imputation(fname + tag)
                        override_mask = np.isnan(imputation_utils.load_imputation('global_bw' + tag))
                        data[override_mask] = np.nan
                    else:
                        mask = np.isnan(data)
                        data[mask] = imputation_utils.load_imputation(fname + tag)[mask]
                        
                        
                imputed = data
                tgt = np.copy(tgt)
                tgt[np.isnan(imputed)] = np.nan
                
                overal_r2 = np.nanmean(1 - np.nansum(np.square(tgt - imputed), axis=(1,2)) /
                            np.nansum(np.square(tgt), axis=(1,2)), axis=0)

                monthly_r2 = np.nanmean(1 - np.nansum(np.square(tgt[:,:,monthly_char_mask] - 
                                                                  imputed[:,:,monthly_char_mask]), axis=(1,2)) /
                            np.nansum(np.square(tgt[:,:,monthly_char_mask]), axis=(1,2)), axis=0)
                
                quarterly_r2 = np.nanmean(1 - np.nansum(np.square(tgt[:,:,quarterly_char_mask] - 
                                                                  imputed[:,:,quarterly_char_mask]), axis=(1,2)) /
                            np.nansum(np.square(tgt[:,:,quarterly_char_mask]), axis=(1,2)), axis=0)
                
                return overal_r2, quarterly_r2, monthly_r2
                
            
            r2s = [get_r2(eval_chars, i, monthly_char_mask, quarterly_char_mask) for i in subsets]
            ret_data = [[round(x, 5) for x in y] for y in r2s]
            ret_cols = [x + tag for x in ['aggregate', "quarterly", 'monthly']]
            return ret_data, ret_cols
        
        results = list(Parallel(n_jobs=4)(delayed(run_tag)(tag) for tag in tags))
        data = [x[0] for x in results]
        columns = sum([x[1] for x in results], [])
        self.data_df = pd.DataFrame(data=np.concatenate(data, axis=1), index=plot_names, columns=columns)

        
class AggregateImputationR2(SectionFiveTableBase):
    name = 'AggregateImputationR2'
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    
    def setup(self, percentile_rank_chars, chars, monthly_updates):
        file_names = [
            'global_fwbw', 
                      'global_fw',
                      'global_bw',
                      'global_xs',
                      'global_ts', 
                      'local_bw',
                       "global_bxs_resid_ts",
                      'local_xs', 
                      'prev_val',
                      'local_ts',
                      'xs_median', 
                      'indus_median'
                     ]
        plot_names = [
            "global BF-XS",
            "global F-XS",
            "global B-XS",
            "global XS",
            "global B",
            "local B-XS",
            "XS + AR 1 Model",
            "local XS",  
            "prev", 
            "local B",
            "XS-median",
            "ind-median"
        ]
        data = []
        columns = []
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_logit': "logit_eval_data",
            '_out_of_sample_block': 'prob_block_eval_data',
        }
        fit_maps = {
            '_out_of_sample_MAR': "MAR_fit_data",
            '_out_of_sample_logit': "logit_fit_data",
            '_out_of_sample_block': 'prob_block_fit_data',
        }
        tags = ['_in_sample', '_out_of_sample_MAR', 
                '_out_of_sample_block', 
                '_out_of_sample_logit']
        import os
        for tag in tags:
            if tag != '_in_sample':
                assert os.path.exists('../data/current/imputation_cache/' + eval_maps[tag] + '.npz'), \
                    '../data/current/imputation_cache/' + eval_maps[tag] + '.npz' + ' does not exist'
            for fname in file_names:
                assert os.path.exists('../data/current/imputation_cache/' + fname + tag + '.npz'), \
                    fname + tag + '.npz' + ' does not exist'
        
        monthly_chars = [x for x in chars if char_map[x] == 'M']
        monthly_char_mask = np.isin(chars, monthly_chars)
        quarterly_char_mask = ~monthly_char_mask
        
        def run_tag(tag):
            if tag == '_in_sample':
                eval_chars = np.copy(update_chars)
                lt_10_mask = np.sum(np.isnan(percentile_rank_chars), axis=2) > 35
            else:
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                lt_10_mask = np.sum(np.isnan(imputation_utils.load_imputation(fit_maps[tag])), axis=2) > 35
            eval_chars[lt_10_mask] = np.nan
            def get_r2(tgt, fname, monthly_char_mask, quarterly_char_mask):
                imputed = imputation_utils.load_imputation(fname + tag)
                tgt = np.copy(tgt)
                tgt[np.isnan(imputed)] = np.nan
                
                overal_r2 = np.nanmean(1 - np.nansum(np.square(tgt - imputed), axis=(1,2)) /
                            np.nansum(np.square(tgt), axis=(1,2)), axis=0)

                monthly_r2 = np.nanmean(1 - np.nansum(np.square(tgt[:,:,monthly_char_mask] - 
                                                                  imputed[:,:,monthly_char_mask]), axis=(1,2)) /
                            np.nansum(np.square(tgt[:,:,monthly_char_mask]), axis=(1,2)), axis=0)
                
                quarterly_r2 = np.nanmean(1 - np.nansum(np.square(tgt[:,:,quarterly_char_mask] - 
                                                                  imputed[:,:,quarterly_char_mask]), axis=(1,2)) /
                            np.nansum(np.square(tgt[:,:,quarterly_char_mask]), axis=(1,2)), axis=0)
                
                return overal_r2, quarterly_r2, monthly_r2
                
            
            r2s = [get_r2(eval_chars, i, monthly_char_mask, quarterly_char_mask) for i in file_names]
            ret_data = [[round(x, 5) for x in y] for y in r2s]
            ret_cols = [x + tag for x in ['aggregate', "quarterly", 'monthly']]
            return ret_data, ret_cols
        
        results = list(Parallel(n_jobs=4)(delayed(run_tag)(tag) for tag in tags))
        data = [x[0] for x in results]
        columns = sum([x[1] for x in results], [])
        self.data_df = pd.DataFrame(data=np.concatenate(data, axis=1), index=plot_names, columns=columns)


class ImputationR2Plots(SectionFiveTableBase):
    name = 'ImputationR2Plots'
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    
    def setup(self, percentile_rank_chars, chars, monthly_updates, dates):
        
        
        
        data = []
        columns = []
        date_vals = np.array(dates) // 10000 + ((np.array(dates) // 100) % 100) / 12
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        mean_vols = np.nanmean(np.nanstd(update_chars, axis=0), axis=0)
        
        monthly_chars = [x for x in chars if char_map[x] == 'M']
        monthly_char_mask = np.isin(chars, monthly_chars)
        quarterly_char_mask = ~monthly_char_mask
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_logit': "logit_eval_data",
            '_out_of_sample_block': 'prob_block_eval_data',
        }
        fit_maps = {
            '_out_of_sample_MAR': "MAR_fit_data",
            '_out_of_sample_logit': "logit_fit_data",
            '_out_of_sample_block': 'prob_block_fit_data',
        }
        
        
        
        
        subsets = [
            ['global_fwbw', 'global_bw', 'global_xs'],
            ['global_bw', 'global_xs'],
            ['global_xs'],
            ["global_ts", 'xs_median'], 
            ['local_bw', 'local_xs'],
            ['local_xs'],
            ["local_ts", 'xs_median'], 
            ["prev_val", 'xs_median'], 
            ['xs_median'], 
            ['indus_median'], 
        ]
        
        plot_names = ["global BF-XS", "global B-XS", "global XS",
                      "global B",
                        "local B-XS", "local XS",  "local B", "prev", 
                      "XS-median", "ind-median"]
        assert len(plot_names) == len(subsets)
        
        table_1_plotnames = ["global BF-XS", "local B-XS", "local XS",  "local B",  "XS-median", "ind-median"]
        

        
        table_2_plotnames = ["global BF-XS", "global B-XS", "global XS", "global B",
                             "local B-XS", "local XS", "local B", 'XS-median']
        
        assert np.all(np.isin(table_1_plotnames, plot_names))
        assert np.all(np.isin(table_2_plotnames, plot_names))
        
        
        tags = ['_in_sample', '_out_of_sample_MAR', 
                '_out_of_sample_block', 
                '_out_of_sample_logit']
        def run_tag(tag):
            if tag == '_in_sample':
                eval_chars = update_chars
            else:
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                
            
            def get_r2(tgt, fnames, monthly_char_mask, quarterly_char_mask, tag):
                data = None

                for fname in fnames:
                    if data is None:
                        data = imputation_utils.load_imputation(fname + tag)
                    else:
                        mask = np.isnan(data)
                        data[mask] = imputation_utils.load_imputation(fname + tag)[mask]
                        
                imputed = data
                tgt = np.copy(tgt)
                tgt[np.isnan(imputed)] = np.nan
                r2s = []
                for i in range(imputed.shape[2]):
                    r2s.append(1 - np.nansum(np.square(tgt[:,:,i] - imputed[:,:,i])) / 
                               np.nansum(np.square(tgt[:,:,i])))

                return r2s
            
            metrics = [get_r2(tgt=eval_chars, fnames=fname, 
                              monthly_char_mask=monthly_char_mask, 
                              quarterly_char_mask=quarterly_char_mask,
                                  tag=tag) for fname in subsets]
            def plot_metrics_by_mean_vol(mean_vols, input_metrics, names, chars, save_name=None):
                
                char_names = []
                metrics_by_type = [[] for _ in input_metrics] 

                for i in np.argsort(mean_vols):
                    metrics = [round(y[i], 5) for y in input_metrics]
                    char_names.append(chars[i])
                    for j, m in enumerate(metrics):
                        metrics_by_type[j].append(m)
                plt.tight_layout() 
                fig = plt.figure(figsize=(20,10))
                fig.patch.set_facecolor('white')
                mycolors = ['#152eff', '#e67300', '#0e374c', '#6d904f', '#8b8b8b', '#30a2da', '#e5ae38', '#fc4f30', '#6d904f', '#8b8b8b', '#0e374c']
                for j, (c, line_name, metrics_series) in enumerate(zip(mycolors, names,
                                             metrics_by_type)):
                    
                    plt.plot(np.arange(45), metrics_series, label=line_name, c=c,
                            linestyle=imputation_utils.line_styles[j])
                plt.plot(np.arange(45), np.array(mean_vols)[np.argsort(mean_vols)], label="mean volatility of char", c='black')
                plt.xticks(np.arange(45), chars[np.argsort(mean_vols)], rotation='vertical')
                plt.ylabel("R2")
                plt.legend(prop={'size': 20}, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, framealpha=1)
                plt.minorticks_off()

                if save_name is not None:
                    save_base = '../images-pdfs/section5/metrics_by_char_vol_sort-'
                    save_path = save_base + save_name + '.pdf'
                    plt.savefig(save_path, bbox_inches='tight', format='pdf')
                plt.show()
            
            table_1_metrics = [metrics[plot_names.index(x)] for x in table_1_plotnames]
            table_2_metrics = [metrics[plot_names.index(x)] for x in table_2_plotnames]

            plot_metrics_by_mean_vol(mean_vols, table_1_metrics, table_1_plotnames, chars=chars, 
                                                      save_name='R2-table_1' + tag)
            plot_metrics_by_mean_vol(mean_vols, table_2_metrics, table_2_plotnames, chars=chars,
                                                      save_name='R2-table_2' + tag)

        [run_tag(tag) for tag in [
            '_in_sample', '_out_of_sample_MAR', 
                                  '_out_of_sample_block', '_out_of_sample_logit']]
    def run(self):
        pass        


class ImputationErrorPlots(SectionFiveTableBase):
    name = 'ImputationErrors'
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    
    def setup(self, percentile_rank_chars, chars, monthly_updates, dates):

        data = []
        columns = []
        date_vals = np.array(dates) // 10000 + ((np.array(dates) // 100) % 100) / 12
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        mean_vols = np.nanmean(np.nanstd(update_chars, axis=0), axis=0)
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_logit': "logit_eval_data",
            '_out_of_sample_block': 'prob_block_eval_data',
        }
        fit_maps = {
            '_out_of_sample_MAR': "MAR_fit_data",
            '_out_of_sample_logit': "logit_fit_data",
            '_out_of_sample_block': 'prob_block_fit_data',
        }

        subsets = [
            ['global_fwbw', 'global_bw', 'global_xs'],
            ['global_bw', 'global_xs'],
            ['global_xs'],
            ["global_ts", 'xs_median'], 
            ['local_bw', 'local_xs'],
            ['local_xs'],
            ["local_ts", 'xs_median'], 
            ["prev_val", 'xs_median'], 
            ['xs_median'], 
            ['indus_median'], 
        ]
        
        plot_names = ["global BF-XS", "global B-XS", "global XS",
                      "global B",
                        "local B-XS", "local XS",  "local B", "prev", 
                      "XS-median", "ind-median"]
        
        table_1_plotnames = ["global BF-XS", "local B-XS", "local XS",  "local B",  "XS-median", "ind-median"]
        

        
        table_2_plotnames = ["global BF-XS", "global B-XS", "global XS", "global B",
                             "local B-XS", "local XS", "local B", 'XS-median']
        
        assert np.all(np.isin(table_1_plotnames, plot_names))
        assert np.all(np.isin(table_2_plotnames, plot_names))
        
        tags = ['_in_sample', '_out_of_sample_MAR', 
                '_out_of_sample_block', 
                '_out_of_sample_logit']
        
        def run_tag(tag):
            if tag == '_in_sample':
                eval_chars = update_chars
            else:
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                
            
            def get_metrics(fnames, tag):
                data = None

                for fname in fnames:
                    if data is None:
                        data = imputation_utils.load_imputation(fname + tag)
                    else:
                        mask = np.isnan(data)
                        data[mask] = imputation_utils.load_imputation(fname + tag)[mask]
                        
                imputed = data
                
                return imputation_utils.get_imputation_metrics(imputed, 
                                  eval_char_data=eval_chars, monthly_update_mask=None, 
                                  char_groupings=char_groupings, norm_func=None)
            
            metrics = [get_metrics(fnames, tag) for fnames in subsets]
            
            table_1_metrics = list(itertools.compress(metrics, np.isin(plot_names, table_1_plotnames)))
            table_2_metrics = list(itertools.compress(metrics, np.isin(plot_names, table_2_plotnames)))

            imputation_utils.plot_metrics_by_mean_vol(mean_vols, table_1_metrics, table_1_plotnames, chars=chars, 
                                                      save_name='table_1' + tag)
            imputation_utils.plot_metrics_by_mean_vol(mean_vols, table_2_metrics, table_2_plotnames, chars=chars,
                                                      save_name='table_2' + tag)
            
            imputation_utils.plot_metrics_over_time(table_1_metrics, table_1_plotnames, date_vals, 
                                                    save_name=f'{tag}-table_1_reg', nans_ok='logit' in tag)
            imputation_utils.plot_metrics_over_time(table_2_metrics, table_2_plotnames, date_vals,
                                        save_name=f'{tag}-table_2_reg', nans_ok='logit' in tag)
        list(Parallel(n_jobs=4)(delayed(run_tag)(tag) for tag in 
                               ['_in_sample', '_out_of_sample_MAR', '_out_of_sample_block', '_out_of_sample_logit']))
            
    def run(self):
        pass

class ImputationErrorsByMissingType(SectionFiveTableBase):
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    
    @property
    @abstractmethod
    def flag_val(self) -> int:
        "number of sigfigs to put in table"
        pass
    
    def setup(self, percentile_rank_chars, chars, monthly_updates):
        file_names = [
            'global_fwbw', 
                      'global_fw',
                      'global_bw',
                      'global_xs',
                     'global_ts', 
                      'local_bw',
                      'local_xs', 
                      'prev_val',
                      'local_ts',
                     'xs_median', 
                      'indus_median'
                     ]
        plot_names = [
            "global BF-XS",
            "global F-XS",
            "global B-XS",
            "global XS",
            "global B",
            "local B-XS",
            "local XS",  
            "prev", 
            "local B",
            "XS-median",
            "ind-median"
        ]

        data = []
        columns = []
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_block': "prob_block_eval_data",
            '_out_of_sample_logit': "logit_eval_data",
        }
        
        flag_maps = {
            '_out_of_sample_MAR': "MAR_flag_panel",
            '_out_of_sample_block': "prob_block_flag_panel",
            '_out_of_sample_logit': "logit_flag_panel",
        }
        fit_maps = {
            '_out_of_sample_MAR': "MAR_fit_data",
            '_out_of_sample_logit': "logit_fit_data",
            '_out_of_sample_block': 'prob_block_fit_data',
        }
        
        
        tags = ['_in_sample', '_out_of_sample_MAR', '_out_of_sample_block', '_out_of_sample_logit']

            
        for tag in tags:
            if tag == '_in_sample':
                flags = imputation_utils.get_present_flags(percentile_rank_chars)
                eval_chars = np.copy(update_chars)
                
            else:
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                flags = imputation_utils.load_imputation(flag_maps[tag])
            
            if tag == '_in_sample':
                eval_chars = update_chars
                lt_10_mask = np.sum(np.isnan(percentile_rank_chars), axis=2) > 35
            else:
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                lt_10_mask = np.sum(np.isnan(imputation_utils.load_imputation(fit_maps[tag])), axis=2) > 35
            eval_chars[lt_10_mask] = np.nan
            eval_chars[flags != self.flag_val] = np.nan

            
            def func_call(fname, eval_data, groups):
                    return imputation_utils.get_imputation_metrics(imputation_utils.load_imputation(fname + tag), 
                                      eval_char_data=eval_data, monthly_update_mask=None, 
                                      char_groupings=groups, norm_func=None)    
            metrics = list(Parallel(n_jobs=4)(delayed(func_call)(i, eval_chars, char_groupings) for i in file_names))
            
            data.append([[round(self.norm_func(np.nanmean(x)), 5) for x in y] for y in metrics])
            columns += [x + tag for x in ['aggregate', "quarterly", 'monthly']]
        self.data_df = pd.DataFrame(data=np.concatenate(data, axis=1), index=plot_names, columns=columns)
        

class ImputationErrorsBySizeDecile(SectionFiveTableBase):
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    name = 'ImputationErrorsBySizeDecile'
    

    def setup(self, percentile_rank_chars, chars, monthly_updates):
        file_names = ['local_bw', 'local_xs', 'local_ts']
        plot_names = ['local B-XS', 'local XS', 'local B']
        data = []
        columns = []
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_block': "prob_block_eval_data",
            '_out_of_sample_logit': "logit_eval_data"
        }
        
        flag_maps = {
            '_out_of_sample_MAR': "MAR_flag_panel",
            '_out_of_sample_block': "prob_block_flag_panel",
            '_out_of_sample_logit': "logit_flag_panel"
        }
        missing_maps = {}
        size_ind = np.argwhere(chars=='ME')[0][0]
        sizes = percentile_rank_chars[:,:,size_ind]
        tags = ['_in_sample', '_out_of_sample_MAR', '_out_of_sample_block', '_out_of_sample_logit']
        for dec in range(10):
            fliter = np.logical_and(sizes >= (-0.5 + dec * 0.1), sizes <=( -0.5 + (dec+1) * 0.1))
            for tag in tags:
                if tag == '_in_sample':
                    eval_chars = np.copy(update_chars)

                else:
                    eval_chars = imputation_utils.load_imputation(eval_maps[tag])

                eval_chars[~fliter] = np.nan
                
                def func_call(fname, eval_data, groups):
                    return imputation_utils.get_imputation_metrics(imputation_utils.load_imputation(fname + tag), 
                                      eval_char_data=eval_data, monthly_update_mask=None, 
                                      char_groupings=groups, norm_func=None)    
                metrics = list(Parallel(n_jobs=4)(delayed(func_call)(i, eval_chars, char_groupings) for i in file_names))

                res = [[round(self.norm_func(np.nanmean(x)), 5) for x in y] for y in metrics]
                missing_maps[(dec + 1, tag)] = res
        data = []
        indexs = []
        for decile in range(1, 11):
            for i, plot_name in enumerate(plot_names):
                data.append(sum([missing_maps[(decile, x)][i] for x in tags], []))
                indexs.append((decile, plot_name))
        columns = sum([[x + tag for x in ['all', 'quarterly', 'monthly']] for tag in tags], [])
        self.data_df = pd.DataFrame(data=data, index=indexs, columns=columns)
        
class ImputationErrorsByCharQuintile(SectionFiveTableBase):
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    name = 'ImputationErrorsByCharQuintile'
    

    def setup(self, percentile_rank_chars, chars, monthly_updates):
        file_names = ['global_fwbw', 'global_fw', 'global_bw', 'global_xs',
                     'global_ts', 'local_bw', 'local_xs', 'prev_val', 'local_ts',
                     'xs_median', 'indus_median']
        plot_names = ["global BF-XS", "global F-XS", "global B-XS", "global XS",
                      "global B",
                        "local B-XS", "local XS",  "prev", "local B",
                      "XS-median", "ind-median"]

        data = []
        columns = []
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_block': "prob_block_eval_data",
            '_out_of_sample_logit': "logit_eval_data"
        }
        
        flag_maps = {
            '_out_of_sample_MAR': "MAR_flag_panel",
            '_out_of_sample_block': "prob_block_flag_panel",
            '_out_of_sample_logit': "logit_flag_panel"
        }
        missing_maps = {}
        
        tags = ['_in_sample', '_out_of_sample_MAR', '_out_of_sample_block', '_out_of_sample_logit']
        
        for dec in range(5):
            for tag in tags:
                if tag == '_in_sample':
                    eval_chars = np.copy(update_chars)

                else:
                    eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                
                filter_ = np.logical_and(eval_chars >= (-0.5 + dec * 0.2), eval_chars <=( -0.5 + (dec+1) * 0.2))
                eval_chars[~filter_] = np.nan

                def func_call(fname, eval_data, groups):
                    return imputation_utils.get_imputation_metrics(imputation_utils.load_imputation(fname + tag), 
                                      eval_char_data=eval_data, monthly_update_mask=None, 
                                      char_groupings=groups, norm_func=None)    
                metrics = list(Parallel(n_jobs=4)(delayed(func_call)(i, eval_chars, char_groupings) for i in file_names))

                res = [[round(self.norm_func(np.nanmean(x)), 5) for x in y] for y in metrics]
                missing_maps[(dec + 1, tag)] = res
        data = []
        indexs = []
        for decile in range(1, 6):
            for i, plot_name in enumerate(plot_names):
                x = [missing_maps[(decile, x)][i] for x in tags]
                data.append(sum(x, []))
                indexs.append((decile, plot_name))
                
        columns = sum([[x + tag for x in ['all', 'quarterly', 'monthly']] for tag in tags], [])
        self.data_df = pd.DataFrame(data=data, index=indexs, columns=columns)
        
        
class ImputationErrorsByCharQuintileFullDS(SectionFiveTableBase):
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    name = 'ImputationErrorsByCharQuintileFullDS'
    

    def setup(self, percentile_rank_chars, chars, monthly_updates):
        subsets = [
            ['global_fwbw', 'global_bw', 'global_xs'],
            ['global_bw', 'global_xs'],
            ['global_xs'],
            ["global_ts", 'xs_median'], 
            ['local_bw', 'local_xs'],
            ['local_xs'],
            ["local_ts", 'xs_median'], 
            ["prev_val", 'xs_median'], 
            ['xs_median'], 
        ]
        
        plot_names = ["global BF-XS, B-XS, XS",
                      "global B-XS, XS",
                      "global XS",
                      "global B, median",
                      "local B-XS, XS",
                      "local XS",
                      "local B, median",
                      "prev val, median",
                      "median"]

        data = []
        columns = []
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_block': "prob_block_eval_data",
            '_out_of_sample_logit': "logit_eval_data"
        }
        
        flag_maps = {
            '_out_of_sample_MAR': "MAR_flag_panel",
            '_out_of_sample_block': "prob_block_flag_panel",
            '_out_of_sample_logit': "logit_flag_panel"
        }
        missing_maps = {}
        
        tags = ['_in_sample', '_out_of_sample_MAR', '_out_of_sample_block', '_out_of_sample_logit']
        
        for dec in range(5):
            for tag in tags:
                if tag == '_in_sample':
                    eval_chars = np.copy(update_chars)

                else:
                    eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                
                filter_ = np.logical_and(eval_chars >= (-0.5 + dec * 0.2), eval_chars <=( -0.5 + (dec+1) * 0.2))
                eval_chars[~filter_] = np.nan
                
                
                
                def func_call(fnames, eval_data, groups):
                    data = None

                    for fname in fnames:
                        if data is None:
                            data = imputation_utils.load_imputation(fname + tag)
                        else:
                            mask = np.isnan(data)
                            data[mask] = imputation_utils.load_imputation(fname + tag)[mask]

                    return imputation_utils.get_imputation_metrics(data, 
                                      eval_char_data=eval_data, monthly_update_mask=None, 
                                      char_groupings=groups, norm_func=None)    
                metrics = list(Parallel(n_jobs=4)(delayed(func_call)(i, eval_chars, char_groupings) for i in subsets))

                res = [[round(self.norm_func(np.nanmean(x)), 5) for x in y] for y in metrics]
                missing_maps[(dec + 1, tag)] = res
        data = []
        indexs = []
        for decile in range(1, 6):
            for i, plot_name in enumerate(plot_names):
                x = [missing_maps[(decile, x)][i] for x in tags]
                data.append(sum(x, []))
                indexs.append((decile, plot_name))
                
        columns = sum([[x + tag for x in ['all', 'quarterly', 'monthly']] for tag in tags], [])
        self.data_df = pd.DataFrame(data=data, index=indexs, columns=columns)

class ImputationErrorsByMissingTypeStart(ImputationErrorsByMissingType):
    name = 'ImputationErrorsByMissingTypeStart' 
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    flag_val = -1
    
class ImputationErrorsByMissingTypeMiddle(ImputationErrorsByMissingType):
    name = 'ImputationErrorsByMissingTypeMiddle' 
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    flag_val = -2
    
class ImputationErrorsByMissingTypeEnd(ImputationErrorsByMissingType):
    name = 'ImputationErrorsByMissingTypeEnd' 
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    flag_val = -3

class InfoUsedForImputationBW(SectionFivePlotBase):
    name = 'InfoUsedForImputationBW-bw_beta_weights'
    description = ''
    
    def setup(self, percentile_rank_chars, return_panel, chars, monthly_updates, norm_regressors=False):

        gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=percentile_rank_chars, 
                return_panel=return_panel, min_chars=1, K=10, 
                num_months_train=return_panel.shape[0],
                reg=0.01,
                time_varying_lambdas=False,
                window_size=548, 
                n_iter=1,
                eval_data=None,
                allow_mean=False)

        imputed = np.concatenate([np.expand_dims(x @ lmbda.T, axis=0) for x in gamma_ts])
        
        resids = percentile_rank_chars - imputed        

        suff_stats,_ = imputation_model_simplified.get_sufficient_statistics_last_val(
                    percentile_rank_chars, max_delta=None, residuals=resids
                )
        
        suff_stats, _ = imputation_model_simplified.get_sufficient_statistics_last_val(percentile_rank_chars,
                                                                                       max_delta=None,
                                                                           residuals=resids)
        
        _, bw_betas = imputation_model_simplified.impute_beta_combined_regression(
            percentile_rank_chars, imputed, sufficient_statistics=suff_stats, 
            beta_weights=None, constant_beta=True, get_betas=True
        )
        
        import statsmodels.api as sm
        
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan

        #calculate autocorrelations
        mean_acfs = []
        for i in tqdm(range(45)):
            acfs = []
            for j in range(update_chars[:,:,i].shape[1]):
                in_sample = ~np.isnan(percentile_rank_chars[:,j,i])
                if np.sum(in_sample) > 2:
                    acfs.append(sm.tsa.acf(percentile_rank_chars[in_sample,j,i])[1])
            mean_acfs.append(np.nanmean(acfs))
            assert ~np.isnan(np.nanmean(acfs)), acfs
        
        plt.tight_layout() 
        fig = plt.figure(figsize=(20,10))
        fig.patch.set_facecolor('white')
        
        if norm_regressors:
            resid_std = np.nanstd(resids, axis=(0,1)).reshape(1, -1, 1)
            imp_std = np.nanstd(imputed, axis=(0,1)).reshape(1, -1, 1)
            char_std = np.nanstd(percentile_rank_chars, axis=(0,1)).reshape(1, -1, 1)
            bw_betas *= np.concatenate([imp_std, char_std, resid_std], axis=2)
            
        

        plt.plot(np.arange(45), np.array(mean_acfs)[np.argsort(mean_acfs)], label="Autocorrelation")
        
        bw_betas[0,:,2] += bw_betas[0,:,1]
        norm = np.linalg.norm(bw_betas[0,:,:], axis=1, ord=1)
        plt.plot(np.arange(45), (np.abs(bw_betas[0,:,0]) /norm)[np.argsort(mean_acfs)], label="XS weight")
        
        plt.plot(np.arange(45),(np.linalg.norm(bw_betas[0,:,1:], axis=1, ord=1)/norm)[np.argsort(mean_acfs)], label="TS weight")
        
        plt.xticks(np.arange(45), chars[np.argsort(mean_acfs)], rotation='vertical')
        plt.minorticks_off()
        plt.ylabel("weight")
        plt.legend(prop={'size': 20}, framealpha=0.5)
                      
        

class BW_BetasTable(SectionFiveTableBase):
    description = ''
    sigfigs = 2
    name = 'BW_BetasTable'
    description = ''
    
    def setup(self, percentile_rank_chars, return_panel, chars):
        
        gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                    char_panel=percentile_rank_chars, 
                    return_panel= return_panel, min_chars=10, K=20, 
                    num_months_train=percentile_rank_chars.shape[0],
                    reg=0.01,
                    time_varying_lambdas=False,
                    window_size=548, 
                    n_iter=3,
                    eval_data=None,
                    allow_mean=False)
        
        imputed = np.concatenate([np.expand_dims(x @ lmbda.T, axis=0) for x in gamma_ts])
        resids = percentile_rank_chars - imputed
        
        suff_stats, _ = imputation_model_simplified.get_sufficient_statistics_last_val(percentile_rank_chars,
                                                                                       max_delta=None,
                                                                           residuals=resids)
        
        _, bw_betas = imputation_model_simplified.impute_beta_combined_regression(
            percentile_rank_chars, imputed, sufficient_statistics=suff_stats, 
            beta_weights=None, constant_beta=True, get_betas=True
        )
        
        self.data_df = pd.DataFrame(data = bw_betas[0], index=chars, columns=['CC', 'Prev Val', 'Prev Resid'])
        
        
        
class FactorStructure(SectionFivePlotBase):
    name = 'FactorStructure'
    description = ''
    
    def setup(self, percentile_rank_chars, return_panel, chars):
        
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
        def plot_factor(lmbda, index):
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


            plt.bar(bar_locations, bars, color=colors)
            plt.xticks(ticks=tick_locations, labels=ticks)
            plt.xticks(rotation=90)
            fig.patch.set_facecolor('white')

            for i in np.where((np.max(bar_maxs) - bar_maxs) / np.max(bar_maxs) < 0.5 )[0]:
                ax.get_xticklabels()[i].set_color("green")

            for i in np.where((np.max(bar_maxs) - bar_maxs) / np.max(bar_maxs) < 0.1 )[0]:
                ax.get_xticklabels()[i].set_color("red")

            custom_lines = [Line2D([0], [0], color=group_colors[i], lw=4) for i in range(len(groups))]

            ax.legend(custom_lines, groups)
            plt.show()
        gamma_ts, lmbda, return_mask, missing_counts, beta_lambda, _, _ = imputation_model.impute_panel_xp_lp(
                percentile_rank_chars, return_panel,
            min_chars=10, K=6, num_months_train=240, 
                gamma_0=0, gamma_1=1, gamma_2=0,
                fully_present=False, intercept=False,
                alternative_ft_construction=False,
                hardcoded_mask=None, t_lag_maps=None)
        for i in range(6):
            plot_factor(lmbda, i)

    def run(self):
        pass
    
    
class DataMakeupByType(SectionFiveTableBase):
    description = ''
    sigfigs = 2
    norm_func = np.sqrt
    name = 'DataMakeupByType'
    

    def setup(self, percentile_rank_chars, chars, monthly_updates):
        T = percentile_rank_chars.shape[0]
        eval_maps = {
            '_out_of_sample_MAR': "MAR_eval_data",
            '_out_of_sample_block': "prob_block_eval_data",
            '_out_of_sample_logit': "logit_eval_data"
        }
        fit_maps = {
            '_out_of_sample_MAR': "MAR_fit_data",
            '_out_of_sample_block': "prob_block_fit_data",
            '_out_of_sample_logit': "logit_fit_data"
        }
        

        
        columns = ['start', 'missing', 'end']
        results = []
        for tag in ['in_sample', '_out_of_sample_MAR', '_out_of_sample_block', '_out_of_sample_logit']:
            if tag == 'in_sample':
                fit_chars = percentile_rank_chars
                eval_chars = np.isnan(percentile_rank_chars) * np.any(~np.isnan(percentile_rank_chars),\
                                                                      axis=-1, keepdims=True)
            else:
                fit_chars = imputation_utils.load_imputation(fit_maps[tag])
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
                
            fit_mask = ~np.isnan(fit_chars)
            eval_mask = ~np.isnan(eval_chars)
            
            start_count, middle_count, end_count, total_count = 0, 0, 0, 0
            always_missing = np.all(~fit_mask, axis=0)
            first_occ = np.argmax(fit_mask, axis=0)
            last_occ = T - np.argmax(fit_mask[::-1], axis=0) - 1
            first_occ[always_missing] = T + 1
            last_occ[always_missing] = -1
            
            for t in tqdm(range(T)):
                start_count += np.sum(eval_mask[t] * first_occ > t)
                middle_count += np.sum(eval_mask[t] * (first_occ < t) * (last_occ > t))
                end_count += np.sum(eval_mask[t] * (first_occ < t) * (last_occ < t))
                total_count += np.sum(eval_mask[t])
                
            results.append([start_count / total_count, 
                            middle_count / total_count, 
                            end_count / total_count])
            
        self.data_df = pd.DataFrame(results, index=['in sample', 'MAR', 'block', 'logit'], columns=columns)
            
            