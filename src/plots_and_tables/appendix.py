import os 
from plots_and_tables import plot_base, revision_simulation, revision_utils
import missing_data_utils
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import imputation_utils, imputation_metrics, imputation_model_simplified
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import scipy as scp
from scipy.optimize import LinearConstraint
from models import FactorModel, IpcaModelV2

class AppendixPlotBase(plot_base.PaperPlot, ABC):
    section = 'appendix'

class AppendixTableBase(plot_base.PaperTable, ABC):
    section = 'appendix'

class Simulation():
    
    def run(self, missing_type):
        
        columns = ["EM", "EM", "XS", "XS", "XS>K", "XS>K"]
        
        numiter= 5
        for K in [5, 10, 15]:
            for L in [50, 100, 500]:
                np.random.seed(0)
                standard_results = revision_simulation.run_scheme(numiter=numiter, N=1000,K=K,L=L,
                                              perc=0.3, missingtype=missing_type,num_factors_impute=25,
                               min_chars=10,snr=0.5,factor_scales=None, impute_reg=1e-4,
                                             lmbda_orth=False,
                                             estimate_reg=True)


                means = np.mean(standard_results, axis=0)
                labels = [f"{x} - {y:.3f}" for x, y in zip(columns, means)]
                _ = plt.boxplot(np.array(standard_results)[:,::2], labels=columns[::2], showmeans=True)
                _ = plt.title(f"MCAR data, MSE, K={K} L={L}")
                _ = plt.xticks(rotation='vertical')
                save_path = f'../images-pdfs/appendix/MAR_simulation_CCMSE_residreg_L={L}_K={K}.pdf'
                plt.savefig(save_path)
                plt.show()

                labels = [f"{x} - {y:.3f}" for x, y in zip(columns, means)]
                _ = plt.boxplot(np.array(standard_results)[:,1::2], labels=columns[1::2], showmeans=True)
                _ = plt.title(f"MCAR data, CC MSE, K={K} L={L}")
                _ = plt.xticks(rotation='vertical')
                save_path = f'../images-pdfs/appendix/MAR_simulation_CCMSE_residreg_L={L}_K={K}.pdf'
                plt.savefig(save_path, bbox_inches='tight')
                plt.show()

        
class NonlinearIPCA(AppendixTableBase):
    description = ''
    name = 'NonlinearIPCA'
    sigfigs = 3

    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars, rts, expansion_dim):
        
        return_lag = 6

        def ntile_basis_expansion(char_vect, n):
            ret_arr = np.zeros((char_vect.shape[0], n), dtype=float)
            ret_arr[:,:] = np.nan
            ret_arr[~np.isnan(char_vect),:] = 0
            bins = np.linspace(-0.5, 0.5, n+1)
            bin_vals = np.digitize(char_vect, bins, right=False)
            for i in range(n):
                ret_arr[bin_vals == i+1, i] = char_vect[bin_vals == i+1]
        #         ret_arr[bin_vals == i+1, i] -= np.mean(ret_arr[bin_vals == i+1, i])
            return ret_arr

        def get_ntile_basis_panel(chars, n):
            T, N, C = chars.shape
            ret_panel = np.zeros((T, N, C * n))
            for t in range(T):
                for c in range(C):
                    ret_panel[t,:,c*n:(c+1)*n] = ntile_basis_expansion(chars[t,:,c], n)
            return ret_panel

        def get_sharpes(decile_chars, imputed_decile_chars, chars, return_panel, rts, nfactors):
            
            
            NUM_MONTHS_TRAIN = 30
            

            results = []
            num_chars = percentile_rank_chars.shape[-1]
            
            mask_fo = np.logical_and(np.all(~np.isnan(decile_chars), axis=2), ~np.isnan(return_panel))
            mask_imputed = np.logical_and(np.all(~np.isnan(imputed_decile_chars), axis=2), ~np.isnan(return_panel))
            
            net_mask = np.logical_and(np.sum(mask_fo, axis=(1)) > 0, np.sum(mask_imputed, axis=(1)) > 0)
            decile_chars = decile_chars[net_mask]
            imputed_decile_chars = imputed_decile_chars[net_mask]
            return_panel = return_panel[net_mask]
            rts = rts[net_mask]
            
            mask_fo = np.logical_and(np.all(~np.isnan(decile_chars), axis=2), ~np.isnan(return_panel))
            mask_imputed = np.logical_and(np.all(~np.isnan(imputed_decile_chars), axis=2), ~np.isnan(return_panel))
            
            for nfactor in nfactors:

                ipca_fo = IpcaModelV2.IpcaModel("fully_observed", 
                                   num_factors=nfactor, num_chars=num_chars, 
                                                iter_tol=10e-6, maxiter=10)

                ipca_imputed = IpcaModelV2.IpcaModel("imputed", 
                                   num_factors=nfactor, num_chars=num_chars, 
                                                     iter_tol=10e-6, maxiter=10)

                
                

                ipca_fo.fit(return_panel, char_panel=decile_chars, 
                            masks=mask_fo, num_months_train=NUM_MONTHS_TRAIN)
                ipca_imputed.fit(return_panel, char_panel=imputed_decile_chars, 
                            masks=mask_imputed, num_months_train=NUM_MONTHS_TRAIN)



                from models.ipca_util import calculate_sharpe_ratio, calculate_efficient_portofolio

                fp_np_factors = np.copy(ipca_fo.in_factors)
                imputed_np_factors = np.copy(ipca_imputed.in_factors)
                fp_oos_mv_portfolio_returns = []
                imputed_oos_mv_portfolio_returns = []

                for t in range(len(ipca_fo.out_factors)):
                    
                    np_risk_free_rates = rts[t:NUM_MONTHS_TRAIN+t]

                    weights = calculate_efficient_portofolio(fp_np_factors[:,t:], np_risk_free_rates.squeeze())
                    ft = ipca_fo.out_factors[t]
                    oos_return = weights.T.dot(ft)
                    fp_oos_mv_portfolio_returns.append(oos_return)
                    fp_np_factors = np.concatenate([fp_np_factors, ft], axis=1)

                    weights = calculate_efficient_portofolio(imputed_np_factors[:,t:], np_risk_free_rates.squeeze())
                    ft = ipca_imputed.out_factors[t]
                    oos_return = weights.T.dot(ft)
                    imputed_oos_mv_portfolio_returns.append(oos_return)

                    imputed_np_factors = np.concatenate([imputed_np_factors, ft], axis=1)

                fp_is_weights = calculate_efficient_portofolio(ipca_fo.in_factors, rts[:NUM_MONTHS_TRAIN].squeeze())
                imputed_is_weights = calculate_efficient_portofolio(ipca_imputed.in_factors, rts[:NUM_MONTHS_TRAIN].squeeze())

                fp_is_returns = fp_is_weights.T.dot(ipca_fo.in_factors)
                imputed_is_returns = imputed_is_weights.T.dot(ipca_imputed.in_factors)

                fp_sharpes = (calculate_sharpe_ratio(np.array(fp_is_returns) - rts[:NUM_MONTHS_TRAIN]),
                             calculate_sharpe_ratio(np.array(fp_oos_mv_portfolio_returns) - rts[NUM_MONTHS_TRAIN:]))

                imputed_sharpes = (calculate_sharpe_ratio(np.array(imputed_is_returns) - rts[:NUM_MONTHS_TRAIN]),
                             calculate_sharpe_ratio(np.array(imputed_oos_mv_portfolio_returns) - rts[NUM_MONTHS_TRAIN:]))
                results.append((fp_sharpes, imputed_sharpes))
            return results 
        
        
        
        return_panel = return_panel[return_lag:]
        rts = rts[return_lag:]
        
        return_panel = return_panel - rts.reshape(-1, 1)
        rts = rts * 0 
        
        imputed_chars = imputation_utils.load_imputation('local_bw_in_sample')
        imputed_chars[~np.isnan(percentile_rank_chars)] = percentile_rank_chars[~np.isnan(percentile_rank_chars)]
        mask = np.isnan(imputed_chars)
        imputed_chars[mask] = imputation_utils.load_imputation('local_xs_in_sample')[mask]
        imputed_chars = np.clip(imputed_chars, -0.5, 0.5)
        
        percentile_rank_chars = get_ntile_basis_panel(percentile_rank_chars, expansion_dim)
        imputed_chars = get_ntile_basis_panel(imputed_chars, expansion_dim)
        
        percentile_rank_chars = percentile_rank_chars[:-return_lag]
        imputed_chars = imputed_chars[:-return_lag]

        
        
        sharpes = get_sharpes(percentile_rank_chars, imputed_chars, 
                              chars, return_panel, rts, nfactors=[3, 4, 5, 6, 7, 8, 9])

        self.data_df = pd.DataFrame(np.array(sharpes).reshape(7, 4), 
                                 index=[f'k={x}' for x in [3, 4, 5, 6, 7, 8, 9]], 
                                 columns = ['IS Observed','OOS Observed', 'IS Imputed', 'OOS Imputed']
                                )
        

        x_locs = [i*3 + j for i in range(7) for j in range(2)]
        is_heights = sum([[sharpes[i][0][0], sharpes[i][1][0]] for i in range(7)], [])
        oos_heights = sum([[sharpes[i][0][1], sharpes[i][1][1]] for i in range(7)], [])
        colors = sum([['orange', 'blue'] for i in range(7)], [])
        
        
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,9))
        orange_patch = mpatches.Patch(color='orange', label='Fully Observed')
        blue_patch = mpatches.Patch(color='blue', label='Imputed')
        plt.legend(handles=[orange_patch, blue_patch], fontsize=24)
        plt.bar(x_locs, is_heights, color=colors)
        plt.xticks([i*3 + 1 for i in range(7)], labels=[f"K={i+3}" for i in range(7)])
        plt.minorticks_off()
        plt.gca().tick_params(axis='both', which='major', labelsize=20)
        plt.savefig(f'../images-pdfs/appendix/decile_ipca_sharpes_in_sample.pdf'.replace(' ', ''), 
                            bbox_inches='tight')
        plt.show()

        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,9))
        orange_patch = mpatches.Patch(color='orange', label='Fully Observed')
        blue_patch = mpatches.Patch(color='blue', label='Imputed')
        plt.legend(handles=[orange_patch, blue_patch], fontsize=24)
        plt.bar(x_locs, oos_heights, color=colors)
        plt.xticks([i*3 + 1 for i in range(7)], labels=[f"K={i+3}" for i in range(7)])
        plt.minorticks_off()
        plt.gca().tick_params(axis='both', which='major', labelsize=20)
        plt.savefig(f'../images-pdfs/appendix/decile_ipca_sharpes_outof_sample.pdf'.replace(' ', ''), 
                            bbox_inches='tight')
        plt.show()
          
          
              
              
        
class ComparisonOfModelConfigs(AppendixTableBase):
    description = ''
    name = 'ComparisonOfModelConfigs'
    sigfigs = 3

    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars,
        regular_chars, permnos, dates, monthly_updates, char_maps):
        
      
        def impute_chars_with_all_configs(tag, char_maps,
                          eval_char_data=None, num_months_train=None, window_size=None):
    
            metrics = []
            names = []
            char_lags = {'Q': 3, 'M': 1, 'QM': 1, 'Y': 3, 'QY': 1}
            char_lag_maps = {k: char_lags[v] for k,v in char_maps.items()}
            
            eval_maps = {
                            '_out_of_sample_MAR': "MAR_eval_data",
                            '_out_of_sample_block': "prob_block_eval_data",
                            '_out_of_sample_logit': "logit_eval_data",
                        }
            fit_maps = {
                            '_out_of_sample_MAR': "MAR_fit_data",
                            '_out_of_sample_block': "prob_block_fit_data",
                            '_out_of_sample_logit': "logit_fit_data",
                        }
            
            char_data = imputation_utils.load_imputation(fit_maps[tag])
            
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                    char_panel=char_data, 
                    return_panel= return_panel, min_chars=10, K=20, 
                    num_months_train=percentile_rank_chars.shape[0],
                    reg=0.01,
                    time_varying_lambdas=False,
                    window_size=1, 
                    n_iter=1,
                    eval_data=None,
                    allow_mean=False)

            xs_imputation = imputation_model_simplified.get_all_xs_vals(char_data, reg=0.01, 
                                         Lmbda=lmbda, time_varying_lmbda=False)
            
            residuals = char_data - xs_imputation

            eval_chars = imputation_utils.load_imputation(eval_maps[tag])

            suff_stats,_ = imputation_model_simplified.get_sufficient_statistics_last_val(
                char_data, max_delta=None, residuals=residuals
            )


            imputed_chars = imputation_model_simplified.impute_beta_combined_regression(
                char_data, xs_imputation, sufficient_statistics=suff_stats, 
                beta_weights=None, constant_beta=False
            )
            
            metrics.append([[round(np.sqrt(np.nanmean(x)), 5) for x in 
                             imputation_utils.get_imputation_metrics(imputed_chars, 
                                                      eval_char_data=eval_chars, monthly_update_mask=None, 
                                                      char_groupings=char_groupings, norm_func=None)]])
            names.append('xhat-t, x_t-1, e_t-1')
            
            imputed_chars = imputation_model_simplified.impute_beta_combined_regression(
                char_data, xs_imputation, sufficient_statistics=suff_stats[:,:,:,0:1], 
                beta_weights=None, constant_beta=False
            )
            metrics.append([[round(np.sqrt(np.nanmean(x)), 5) for x in 
                             imputation_utils.get_imputation_metrics(imputed_chars, 
                                                      eval_char_data=eval_chars, monthly_update_mask=None, 
                                                      char_groupings=char_groupings, norm_func=None)]])
            names.append('xhat-t, x_t-1')
            
            imputed_chars = imputation_model_simplified.impute_beta_combined_regression(
                char_data, xs_imputation, sufficient_statistics=suff_stats[:,:,:,1:2], 
                beta_weights=None, constant_beta=False
            )
            metrics.append([[round(np.sqrt(np.nanmean(x)), 5) for x in 
                             imputation_utils.get_imputation_metrics(imputed_chars, 
                                                      eval_char_data=eval_chars, monthly_update_mask=None, 
                                                      char_groupings=char_groupings, norm_func=None)]])
            names.append('xhat-t, e_t-1')
            
            
            oos_gamma_ts = imputation_model_simplified.get_all_xs_vals(char_data, reg=0.01, 
                                         Lmbda=lmbda, time_varying_lmbda=False, get_factors=True)
            
            imputed_chars = imputation_model_simplified.impute_beta_combined_regression(
                char_data, None, sufficient_statistics=suff_stats, 
                beta_weights=None, constant_beta=False, gamma_ts = oos_gamma_ts, use_factors=True
            )
            metrics.append([[round(np.sqrt(np.nanmean(x)), 5) for x in 
                             imputation_utils.get_imputation_metrics(imputed_chars, 
                                                      eval_char_data=eval_chars, monthly_update_mask=None, 
                                                      char_groupings=char_groupings, norm_func=None)]])
            names.append('f-t, x_t-1, e_t-1')
            
            imputed_chars = imputation_model_simplified.impute_beta_combined_regression(
                char_data, None, sufficient_statistics=suff_stats[:,:,:,0:1] ,
                beta_weights=None, constant_beta=False, gamma_ts = oos_gamma_ts, use_factors=True
            )
            metrics.append([[round(np.sqrt(np.nanmean(x)), 5) for x in 
                             imputation_utils.get_imputation_metrics(imputed_chars, 
                                                      eval_char_data=eval_chars, monthly_update_mask=None, 
                                                      char_groupings=char_groupings, norm_func=None)]])
            names.append('f-t, x_t-1')
            
                
            return metrics, names
        
        
        self.results = [impute_chars_with_all_configs(tag, char_maps,
                          eval_char_data=None, num_months_train=None, window_size=None) 
         for tag in ['_out_of_sample_MAR', '_out_of_sample_block', '_out_of_sample_logit']
        ]
        
        names = self.results[0][1]
        data = np.concatenate([x[0] for x in self.results], axis=1).reshape(5, 9)
        
        self.data_df = pd.DataFrame(data=data, index=names)
        
        
        
class SparseFactors(AppendixPlotBase):
    description = ''
    name = 'SparseFactors'

    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars,
        regular_chars, permnos, dates):
        
        import imputation_model_simplified
        gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
            char_panel=percentile_rank_chars[-10:], 
            return_panel=return_panel[-10:], min_chars=10, K=10, 
            num_months_train=10, #percentile_rank_chars.shape[0],
            reg=0.01,
            time_varying_lambdas=False,
            window_size=1, 
            n_iter=1, 
            eval_data=None,
            allow_mean=False)
        
        t_mask = np.all(~np.isnan(gamma_ts[-1]), axis=1)
        
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

        group_mask = [np.isin(chars, x) for x in char_groupings.values()]
        lmbda_tilde, gamma_tildes = revision_utils.sparsify_lmbda(lmbda, gamma_ts[-1:][:,t_mask], 
                                           group_masks=group_mask, maxiter=3, tol=1e-3,
                                          reg=100)
        
        
        for i in range(10):
            revision_utils.plot_factor(lmbda, i, chars, tag='full_factors')
            revision_utils.plot_factor(lmbda_tilde, i, chars, tag='sparse_factors')
            
            
class SparseFactorsTable(AppendixTableBase):
    description = ''
    name = 'SparseFactorsTable'

    def setup(self, percentile_rank_chars, return_panel, char_freqs, chars,
        regular_chars, permnos, dates):
        
        self.metrics = []
        for tag in ['_in_sample', '_out_of_sample_MAR', '_out_of_sample_block', 
                   '_out_of_sample_logit']:
            if tag == '_in_sample':
                fit_chars = percentile_rank_chars
                eval_chars = percentile_rank_chars
            else:
                eval_maps = {
                    '_out_of_sample_MAR': "MAR_eval_data",
                    '_out_of_sample_block': "prob_block_eval_data",
                    '_out_of_sample_logit': "logit_eval_data",
                }
                fit_maps = {
                                '_out_of_sample_MAR': "MAR_fit_data",
                                '_out_of_sample_block': "prob_block_fit_data",
                                '_out_of_sample_logit': "logit_fit_data",
                            }
            
                fit_chars = imputation_utils.load_imputation(fit_maps[tag])
                eval_chars = imputation_utils.load_imputation(eval_maps[tag])
        
            import imputation_model_simplified
            gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                char_panel=fit_chars, 
                return_panel=return_panel, min_chars=10, K=10, 
                num_months_train=percentile_rank_chars.shape[0],
                reg=0.01,
                time_varying_lambdas=False,
                window_size=1, 
                n_iter=1, 
                eval_data=None,
                allow_mean=False)

            t_mask = np.all(~np.isnan(gamma_ts), axis=2)

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
            t_start = 500
            group_mask = [np.isin(chars, x) for x in char_groupings.values()]
            results = revision_utils.sparsify_lmbda(lmbda, [gamma_ts[t, t_mask[t]] for t in range(t_start, gamma_ts.shape[0])], 
                                               group_masks=group_mask, maxiter=3, tol=1e-3,
                                              reg=100)

            new_gts = np.copy(gamma_ts[t_start:])
            new_lmbda = results[0]

            imputed = imputation_model_simplified.get_all_xs_vals(fit_chars, reg=0.01, 
                                             Lmbda=new_lmbda, time_varying_lmbda=False)

            metrics = imputation_utils.get_imputation_metrics(imputed, 
                                          eval_char_data=eval_chars, monthly_update_mask=None, 
                                          char_groupings=char_freqs, norm_func=None)


            self.metrics.append((tag, [np.round(np.sqrt(np.nanmean(x)), 5) for x in metrics]))
        

class ComparisonWithAlternativeMethods(AppendixTableBase):
    
    description=''
    name = 'ComparisonWithAlternativeMethods'
    sigfigs=3
    
    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars,
        regular_chars, permnos, dates, rts, char_map):
        
        regr_chars = ['VAR', 'IdioVol', 'SPREAD', 'D2P', 'R2_1', 'ME']
        self.metrics = []
        
        monthly_chars = [x for x in chars if char_map[x] == 'M']
        monthly_char_mask = np.isin(chars, monthly_chars)
        quarterly_char_mask = ~monthly_char_mask

        eval_maps = {
                'MAR': "MAR_eval_data",
                'BLOCK': "prob_block_eval_data",
                'logit': "logit_eval_data",
            }
        columns = []
        
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
        
        self.r2s = []
        
        for t1, t2 in [('IN SAMPLE', '_in_sample'),
            ('MAR', '_out_of_sample_MAR'), ('BLOCK', '_out_of_sample_block'), ('logit', '_out_of_sample_logit')]:
            tag_metrics = []
            tag_r2s = []
            
            if t2 != '_in_sample':
                eval_chars = imputation_utils.load_imputation(eval_maps[t1])
            else:
                eval_chars = percentile_rank_chars
            
            imp = imputation_utils.load_imputation("local_bw" + t2)
            mask = np.isnan(imp)
            eval_chars[mask] = np.nan
            m = imputation_utils.get_imputation_metrics(imp, 
                                                  eval_char_data=eval_chars, monthly_update_mask=None, 
                                                  char_groupings=char_groupings, norm_func=None)   
            tag_metrics.append([round(np.sqrt(np.nanmean(x)), 5) for x in m])
            tag_r2s.append([round(x, 5) for x in get_r2(eval_chars, imp, monthly_char_mask, quarterly_char_mask)])
            
            imp = imputation_utils.load_imputation("local_xs" + t2)
            m = imputation_utils.get_imputation_metrics(imp, 
                                                  eval_char_data=eval_chars, monthly_update_mask=None, 
                                                  char_groupings=char_groupings, norm_func=None)   
            tag_metrics.append([round(np.sqrt(np.nanmean(x)), 5) for x in m])
            tag_r2s.append([round(x, 5) for x in get_r2(eval_chars, imp, monthly_char_mask, quarterly_char_mask)])

            imp = revision_utils.impute_chars_freyweb(chars, regr_chars=regr_chars, 
                                                        missing_data_type=t1,
                                                     percentile_rank_chars=percentile_rank_chars)
            
            m = imputation_utils.get_imputation_metrics(imp, 
                                                  eval_char_data=eval_chars, monthly_update_mask=None, 
                                                  char_groupings=char_groupings, norm_func=None)   
            tag_metrics.append([round(np.sqrt(np.nanmean(x)), 5) for x in m])
            tag_r2s.append([round(x, 5) for x in get_r2(eval_chars, imp, monthly_char_mask, quarterly_char_mask)])
            
            imp, _ = revision_utils.impute_andrew_chen(t1, min_chars=1, maxiter=10,
                                                     percentile_rank_chars=percentile_rank_chars)
            
            m = imputation_utils.get_imputation_metrics(imp, 
                                                  eval_char_data=eval_chars, monthly_update_mask=None, 
                                                  char_groupings=char_groupings, norm_func=None,
                                                       clip=False)   
            tag_metrics.append([round(np.sqrt(np.nanmean(x)), 5) for x in m])
            tag_r2s.append([round(x, 5) for x in get_r2(eval_chars, imp, monthly_char_mask, quarterly_char_mask)])
            
            self.metrics.append(tag_metrics)
            self.r2s.append(tag_r2s)
            columns += [x + '-' + t1 for x in ['aggregate', "quarterly", 'monthly']]
            
            
            
            
            
        labels = ['local B-XS', 'local XS', 'F\&W', 'EM']
        self.data_df = pd.DataFrame(
            data=[sum([x[i] for x in self.metrics], []) for i in range(4)], index=labels, columns=columns
        )
        
        self.r2_data_df = pd.DataFrame(
            data=[sum([x[i] for x in self.r2s], []) for i in range(4)], index=labels, columns=columns
        )

        
from joblib import Parallel, delayed
class RegularizationOfBetaRegression(AppendixTableBase):
    description = ''
    name = 'RegularizationOfBetaRegression'
    sigfigs = 3

    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars,
        regular_chars, permnos, dates, monthly_updates, local=False):
          
    
        def eval_configs(tag, regs):
            eval_maps = {
                '_out_of_sample_MAR': "MAR_eval_data",
                '_out_of_sample_block': "prob_block_eval_data",
                '_out_of_sample_logit': "logit_eval_data",
            }
            fit_maps = {
                            '_out_of_sample_MAR': "MAR_fit_data",
                            '_out_of_sample_block': "prob_block_fit_data",
                            '_out_of_sample_logit': "logit_fit_data",
                        }
            
            fit_chars = imputation_utils.load_imputation(fit_maps[tag])
            eval_chars = imputation_utils.load_imputation(eval_maps[tag])
            base_path = '../data/current/imputation_cache/'
            rerun_gammas = False
            if local:
                result_file_name = base_path + "local_gammas" + tag + '.npz'
            else:
                result_file_name = base_path + "global_gammas" + tag + '.npz'
                
            res = np.load(result_file_name)
            gamma_ts = res['gamma_ts']
            lmbda = res['lmbda']
            
            if local:
                xs_imputations = np.concatenate([np.expand_dims(g @ l.T, axis=0) for g, l in zip(gamma_ts, lmbda)], axis=0)
            else:
#                 xs_imputations = np.concatenate([np.expand_dims(g @ lmbda.T, axis=0) for g in gamma_ts], axis=0)
                xs_imputations = get_all_xs_vals(chars=fit_chars, reg=0.01, Lmbda=lmbda)
           
            residuals = fit_chars - xs_imputations

            suff_stats,_ = imputation_model_simplified.get_sufficient_statistics_last_val(
                                fit_chars, max_delta=None, residuals=residuals
                            )
            names = []
            metrics = []
            for switch_off_on_suff_stats in [False]:
                for reg in regs:

                    global_reg_bxs = imputation_model_simplified.impute_beta_combined_regression(
                        fit_chars, xs_imputations, sufficient_statistics=suff_stats, 
                        beta_weights=None, constant_beta=not local, gamma_ts = None, use_factors=False,
                        reg=reg, switch_off_on_suff_stats=switch_off_on_suff_stats
                    )

                    metrics.append([[round(np.sqrt(np.nanmean(x)), 5) for x in 
                                             imputation_utils.get_imputation_metrics(global_reg_bxs, 
                                                                      eval_char_data=eval_chars, monthly_update_mask=None, 
                                                                      char_groupings=char_groupings, norm_func=None)]])
                    names.append(f'{reg}-{switch_off_on_suff_stats}')
            return metrics, names

#         self.results = results = list(Parallel(n_jobs=4)(delayed(eval_configs)(tag, [1e-7, 1e-6, 1e-5])
#                                                          for tag in ['_out_of_sample_MAR', 
# #                                                                      '_out_of_sample_block',
#                                                                      '_out_of_sample_logit'
#                                                                     ]))
        self.results = results = [eval_configs(tag, [0])
                                                         for tag in [
                                                             '_out_of_sample_MAR', 
                                                                     '_out_of_sample_block',
                                                                     '_out_of_sample_logit'
                                                                    ]]
        names = self.results[0][1]
        data = np.concatenate([x[0] for x in self.results], axis=1).reshape(4, 9)
        
        self.data_df = pd.DataFrame(data=data, index=names)

        

                      
                 