from plots_and_tables import plot_base
import missing_data_utils
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.notebook import tqdm 
import imputation_utils, imputation_model_simplified
import pandas as pd
import scipy as scp

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

class SectionFourPlotBase(plot_base.PaperPlot, ABC):
    section = 'section4'

class SectionFourTableBase(plot_base.PaperTable, ABC):
    section = 'section4'

class XS_Factor_EV(SectionFourPlotBase):
    name = 'figure_2_avg_cov_ev'
    description = "averge EVS of characteristic covariance matrix over time"
    
    
    def setup(self, return_panel, percentile_rank_chars):
    
        cov_mats = [missing_data_utils.get_cov_mat(percentile_rank_chars[t])
            for t in tqdm(range(percentile_rank_chars.shape[0]))]
        cov_evals = np.array([sorted(np.linalg.eigh(c)[0])[::-1] for c in cov_mats if np.sum(np.isnan(c)) == 0])
        normed_cov_evals = cov_evals / np.sum(cov_evals, axis=1, keepdims=True)
        from matplotlib.ticker import MaxNLocator
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(np.arange(1, 41), np.mean(cov_evals[:,0:40], axis=0) / np.sum(np.mean(cov_evals[:,:], axis=0)))

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0.25,20.25)
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Average magnitude")
        # ax.set_yscale('log')
        plt.xticks(np.arange(1, 41, 1.0))
        plt.gca().tick_params(axis='both', which='major', labelsize=15)


class Optimal_Num_FactorsMSE(SectionFourPlotBase):
    name = "Optimal_Num_FactorsMSE"
    description = ''
    
    def setup(self, percentile_rank_chars, return_panel, chars, tag, factor_nums):
        self.name += '-' + tag
        
        flag_panel = imputation_utils.load_imputation(tag + "_flag_panel")
        eval_data = imputation_utils.load_imputation(tag + "_eval_data")
        fit_data = imputation_utils.load_imputation(tag + "_fit_data")
        
        T = percentile_rank_chars.shape[0]
        
        monthly_chars = [x for x in chars if char_map[x] == 'M']
        monthly_char_mask = np.isin(chars, monthly_chars)
        quarterly_char_mask = ~monthly_char_mask

        block_metrics_by_num_factors = []
        for k in factor_nums:    
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=fit_data, 
                return_panel= return_panel, min_chars=10, K=k, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01,
                time_varying_lambdas=False,
                window_size=548, 
                n_iter=1,
                eval_data=None,
                allow_mean=False)

            imputation = imputation_utils.impute_chars(gamma_ts, fit_data, 
                                                 "None", None, char_groupings, num_months_train=T,
                                                                eval_char_data=eval_data,
                                                                   window_size=None, lmbda=lmbda, tv_lmbda=False)
            metrics = imputation_utils.get_imputation_metrics(imputation, 
                                      eval_char_data=eval_data, monthly_update_mask=None, 
                                      char_groupings=char_groupings, norm_func=None)
            
            rmses = [np.sqrt(np.nanmean(x)) for x in metrics]
            
            block_metrics_by_num_factors.append(rmses)
        
        agg_mmse = [x[0] for x in block_metrics_by_num_factors]
        monthly_mmse = [x[2] for x in block_metrics_by_num_factors]
        quarterly_mmse = [x[1] for x in block_metrics_by_num_factors]
        plt.figure(figsize=(7.5, 5))
        plt.gca().tick_params(axis='both', which='major', labelsize=25)

        plt.plot(factor_nums, agg_mmse[:45], label='aggregate')
        plt.plot(factor_nums, monthly_mmse[:45], label='monthly characteristics')
        plt.plot(factor_nums, quarterly_mmse[:45], label='quarterly characteristics')
        plt.gca().tick_params(axis='both', which='major', labelsize=15)
        plt.legend()

        
class Optimal_Num_Factors(SectionFourPlotBase):
    name = "number_of_factors_determination_xs"
    description = ''
    
    def setup(self, percentile_rank_chars, return_panel, chars, tag, factor_nums, reg=0.01, eval_weight=True,
             recalc_data=True, shrink_lmbda=False):
        self.name += '-' + tag + f'{reg}'.replace('.', '_') + '-' + f"{eval_weight}"
        if shrink_lmbda:
            self.name += '-shrunken_lmbda'
        
        flag_panel = imputation_utils.load_imputation(tag + "_flag_panel")
        eval_data = imputation_utils.load_imputation(tag + "_eval_data")
        fit_data = imputation_utils.load_imputation(tag + "_fit_data")
        
        T = percentile_rank_chars.shape[0]
        
        monthly_chars = [x for x in chars if char_map[x] == 'M']
        monthly_char_mask = np.isin(chars, monthly_chars)
        quarterly_char_mask = ~monthly_char_mask


   
        if recalc_data:
            def get_r2_for_k(k, fit_data, return_panel, T, eval_data, char_groupings,
                            monthly_char_mask, quarterly_char_mask):
                gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                    char_panel=fit_data, 
                    return_panel= return_panel, min_chars=10, K=k, 
                    num_months_train=T,
                    reg=reg,
                    time_varying_lambdas=False,
                    window_size=548, 
                    n_iter=1,
                    eval_data=None,
                    allow_mean=False,
                    eval_weight_lmbda=eval_weight,
                    shrink_lmbda=shrink_lmbda)

                imputation = imputation_utils.impute_chars(gamma_ts, fit_data, 
                                                     "None", None, char_groupings, num_months_train=T,
                                                                    eval_char_data=eval_data,
                                                                       window_size=None, lmbda=lmbda, tv_lmbda=False)
                imputation = np.clip(imputation, -0.5, 0.5)
                def get_r2(tgt, imputed, monthly_char_mask, quarterly_char_mask):
                    tgt = np.copy(tgt)
                    tgt[np.isnan(imputed)] = np.nan

                    overal_r2 = np.nanmean(
                        1 - np.nansum(np.square(tgt - imputed), axis=(1,0)) /
                            np.nansum(np.square(tgt), axis=(1,0)), 
                        axis=0
                    )

                    monthly_r2 = np.nanmean(
                        1 - np.nansum(np.square(tgt[:,:,monthly_char_mask] - imputed[:,:,monthly_char_mask]), axis=(1,0)) /
                            np.nansum(np.square(tgt[:,:,monthly_char_mask]), axis=(1,0)), 
                        axis=0
                    )

                    quarterly_r2 = np.nanmean(
                        1 - np.nansum(np.square(tgt[:,:,quarterly_char_mask] - imputed[:,:,quarterly_char_mask]), axis=(1,0)) /
                            np.nansum(np.square(tgt[:,:,quarterly_char_mask]), axis=(1,0)), 
                        axis=0
                    )

                    return overal_r2, quarterly_r2, monthly_r2

                r2s = get_r2(eval_data, imputation, monthly_char_mask, quarterly_char_mask)
                return r2s

            block_metrics_by_num_factors = [get_r2_for_k(k=k, fit_data=fit_data,
                                                                                         return_panel=return_panel, 
                                                                                         T=fit_data.shape[0],
                                                                                        eval_data=eval_data,
                                                                                        char_groupings=char_groupings,
                            monthly_char_mask=monthly_char_mask, quarterly_char_mask=quarterly_char_mask)
                                                  for k in factor_nums]

            block_metrics_by_num_factors = [(0, 0, 0)] + block_metrics_by_num_factors
            self.data_df = pd.DataFrame(data=block_metrics_by_num_factors, index=[0] + list(factor_nums),
                                       columns = ['agg r2', 'q r2', 'm r2'])
            save_base = '../images-pdfs/section4/'
            save_path = save_base + self.name + '.csv'
            self.data_df.to_csv(save_path)
        else:
            save_base = '../images-pdfs/section4/'
            save_path = save_base + self.name + '.csv'
            self.data_df = pd.read_csv(save_path, index_col=0)
            block_metrics_by_num_factors = self.data_df.loc[[0] + list(factor_nums)].to_numpy()
            
        agg_mmse = [x[0] for x in block_metrics_by_num_factors]
        monthly_mmse = [x[2] for x in block_metrics_by_num_factors]
        quarterly_mmse = [x[1] for x in block_metrics_by_num_factors]
        
        plt.figure(figsize=(7.5, 5))
        plt.gca().tick_params(axis='both', which='major', labelsize=25)

        plt.plot(list(factor_nums), np.diff(agg_mmse), label='aggregate')
        plt.plot(list(factor_nums), np.diff(monthly_mmse), label='monthly characteristics')
        plt.plot(list(factor_nums), np.diff(quarterly_mmse), label='quarterly characteristics')
        plt.gca().tick_params(axis='both', which='major', labelsize=15)
        plt.legend()
        save_base = '../images-pdfs/section4/metrics_by_char_vol_sort-'
        save_path = save_base + self.name + '-incremental' + '.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
        
        
        plt.figure(figsize=(7.5, 5))
        plt.gca().tick_params(axis='both', which='major', labelsize=25)

        plt.plot([0] + list(factor_nums), agg_mmse, label='aggregate')
        plt.plot([0] + list(factor_nums), monthly_mmse, label='monthly characteristics')
        plt.plot([0] + list(factor_nums), quarterly_mmse, label='quarterly characteristics')
        plt.gca().tick_params(axis='both', which='major', labelsize=15)
        plt.legend()
        

class Optimal_Reg(SectionFourPlotBase):
    name = "optimal_reg_determination_xs"
    description = ''
    
    def setup(self, percentile_rank_chars, return_panel, chars, tag, num_factors, 
              regs,
             recalc_data=True):
        self.name += '-' + tag + f"k={num_factors}" + f'{"_".join([str(x) for x in regs])}'.replace('.', ',') + '-'
        
        flag_panel = imputation_utils.load_imputation(tag + "_flag_panel")
        eval_data = imputation_utils.load_imputation(tag + "_eval_data")
        fit_data = imputation_utils.load_imputation(tag + "_fit_data")
        
        T = percentile_rank_chars.shape[0]
        
        monthly_chars = [x for x in chars if char_map[x] == 'M']
        monthly_char_mask = np.isin(chars, monthly_chars)
        quarterly_char_mask = ~monthly_char_mask


   
        if recalc_data:
            def get_r2_for_reg(reg, fit_data, return_panel, T, eval_data, char_groupings,
                            monthly_char_mask, quarterly_char_mask):
                gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                    char_panel=fit_data, 
                    return_panel= return_panel, min_chars=10, K=num_factors, 
                    num_months_train=T,
                    reg=reg,
                    time_varying_lambdas=False,
                    window_size=548, 
                    n_iter=1,
                    eval_data=None,
                    allow_mean=False,
                    eval_weight_lmbda=True)

                imputation = imputation_utils.impute_chars(gamma_ts, fit_data, 
                                                     "None", None, char_groupings, num_months_train=T,
                                                                    eval_char_data=eval_data,
                                                                       window_size=None, lmbda=lmbda, tv_lmbda=False)
                imputation = np.clip(imputation, -0.5, 0.5)
                def get_r2(tgt, imputed, monthly_char_mask, quarterly_char_mask):
                    tgt = np.copy(tgt)
                    tgt[np.isnan(imputed)] = np.nan

                    overal_r2 = np.nanmean(
                        1 - np.nansum(np.square(tgt - imputed), axis=(1,0)) /
                            np.nansum(np.square(tgt), axis=(1,0)), 
                        axis=0
                    )

                    monthly_r2 = np.nanmean(
                        1 - np.nansum(np.square(tgt[:,:,monthly_char_mask] - imputed[:,:,monthly_char_mask]), axis=(1,0)) /
                            np.nansum(np.square(tgt[:,:,monthly_char_mask]), axis=(1,0)), 
                        axis=0
                    )

                    quarterly_r2 = np.nanmean(
                        1 - np.nansum(np.square(tgt[:,:,quarterly_char_mask] - imputed[:,:,quarterly_char_mask]), axis=(1,0)) /
                            np.nansum(np.square(tgt[:,:,quarterly_char_mask]), axis=(1,0)), 
                        axis=0
                    )

                    return overal_r2, quarterly_r2, monthly_r2

                r2s = get_r2(eval_data, imputation, monthly_char_mask, quarterly_char_mask)
                return r2s

            block_metrics_by_num_factors = [get_r2_for_reg(reg=reg, 
                                                         fit_data=fit_data,
                                                                                         return_panel=return_panel, 
                                                                                         T=fit_data.shape[0],
                                                                                        eval_data=eval_data,
                                                                                        char_groupings=char_groupings,
                            monthly_char_mask=monthly_char_mask, quarterly_char_mask=quarterly_char_mask)
                                                  for reg in regs]

            self.data_df = pd.DataFrame(data=block_metrics_by_num_factors, index=list(regs),
                                       columns = ['agg r2', 'q r2', 'm r2'])
            save_base = '../images-pdfs/section4/'
            save_path = save_base + self.name + '.csv'
            self.data_df.to_csv(save_path)
        else:
            save_base = '../images-pdfs/section4/'
            save_path = save_base + self.name + '.csv'
            self.data_df = pd.read_csv(save_path)
            block_metrics_by_num_factors = self.data_df.to_numpy()
            
        agg_mmse = [x[0] for x in block_metrics_by_num_factors]
        monthly_mmse = [x[2] for x in block_metrics_by_num_factors]
        quarterly_mmse = [x[1] for x in block_metrics_by_num_factors]
        
        
        plt.figure(figsize=(7.5, 5))
        plt.gca().tick_params(axis='both', which='major', labelsize=25)

        plt.plot(regs, agg_mmse, label='aggregate')
        plt.plot(regs, monthly_mmse, label='monthly characteristics')
        plt.plot(regs, quarterly_mmse, label='quarterly characteristics')
        plt.gca().tick_params(axis='both', which='major', labelsize=15)
        plt.xscale('log')
        plt.legend()
        
class GenCorr(SectionFourPlotBase):
    name = 'generalized_corr'
    description = ""
    
    def setup(self, percentile_rank_chars, return_panel, dates):
        date_vals = np.array(dates) // 10000 + ((np.array(dates) // 100) % 100) / 12
        
        def generalized_corr(lambda_1, lambda_2):
            N, r = lambda_1.shape
            rho = np.trace(np.linalg.solve(lambda_1.T @ lambda_1, lambda_1.T @ lambda_2) @ 
                                 np.linalg.solve(lambda_2.T @ lambda_2, lambda_2.T @ lambda_1))
            return rho, r
        gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=percentile_rank_chars, 
                return_panel= return_panel, min_chars=10, K=20, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01,
                time_varying_lambdas=False,
                window_size=548, 
                n_iter=1,
                eval_data=None,
                allow_mean=False)
        
        gamma_ts, lmbdas = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=percentile_rank_chars, 
                return_panel= return_panel, min_chars=10, K=20, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01,
                time_varying_lambdas=True,
                window_size=548, 
                n_iter=1,
                eval_data=None,
                allow_mean=False)
        self.generalized_corrs = [generalized_corr(lmbda, l) for l in lmbdas]
        plt.plot(date_vals, [x[0] for x in self.generalized_corrs], label="Constant vs Time Varying Loadings")
        # plt.plot(date_vals[119:], [x[1] for x in generalized_corrs[19:]], label="Number of Factors")
        plt.ylabel("Generalized Correlation")
        plt.ylim(0, 21)

base_colors = ['r', 'b', 'g', 'y', 'c', 'orange', 'purple']
from matplotlib.lines import Line2D
def plot_factor(lmbda, index, save_name=None, char_groupings=None, chars=None):
    assert char_groupings is not None
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
    plt.minorticks_off()
    fig.patch.set_facecolor('white')
    
    for i in np.where((np.max(bar_maxs) - bar_maxs) / np.max(bar_maxs) < 0.5 )[0]:
        ax.get_xticklabels()[i].set_color("green")
    
    for i in np.where((np.max(bar_maxs) - bar_maxs) / np.max(bar_maxs) < 0.1 )[0]:
        ax.get_xticklabels()[i].set_color("red")

    custom_lines = [Line2D([0], [0], color=group_colors[i], lw=4) for i in range(len(groups))]

    ax.legend(custom_lines, groups)
    
    
    if save_name:
        save_path = f'../images-pdfs/section4/factor_viz-{save_name}-{index}.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()

class FactorViz(SectionFourPlotBase):
    name = "FactorViz"
    description = ""
    
    def setup(self, percentile_rank_chars, return_panel, char_map, chars):
        char_groupings = {
            "Past Returns" : ['R2_1', 'R12_2', 'R12_7', 'R36_13', 'R60_13', 'HIGH52'],
            "Investment": ['INV', 'NOA', 'DPI2A', 'NI'],
            "Profitability": ['PROF', 'ATO', 'CTO', 'FC2Y', 'OP', 'PM', 'RNA', 'ROA', 'ROE', 'SGA2S', 
                             'D2A'],
            "Intangibles": ['AC', 'OA', 'OL', 'PCM'],
            "Value": ['A2ME', 'B2M',  'C2A', 'CF2B', 'CF2P', 'D2P', 'E2P', 'Q',  'S2P'],
            "Trading Frictions": ['BETA_d', 'BETA_m', 'SPREAD'],
            "Other": ['AT', 'LEV', 'IdioVol', 'ME', 'TURN', 'RVAR', 'SUV', 'VAR']
        }
        
        gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=percentile_rank_chars, 
                return_panel= return_panel, min_chars=10, K=10, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01,
                time_varying_lambdas=False,
                window_size=548, 
                n_iter=1,
                eval_data=None,
                allow_mean=False)
        
        for i in range(10):
            plot_factor(lmbda, i, save_name='econ-cat', char_groupings=char_groupings, chars=chars)
        
        freq = {v for k,v in char_map.items()}
        char_groupings = {
            f: [x for x in char_map if char_map[x] == f] for f in freq
        }
        
        for i in range(6):
            plot_factor(lmbda, i, save_name='freq-cat', char_groupings=char_groupings, chars=chars)

    def run(self):
        pass