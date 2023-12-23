from unicodedata import name
from plots_and_tables import plot_base
import missing_data_utils
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import imputation_utils, imputation_model_simplified
from collections import defaultdict
import numpy as np
from itertools import chain, combinations
from plots_and_tables import revision_utils
from models import FactorModel, IpcaModelV2
import scipy as scp
from models import FactorModel

class RevisionPlotBase(plot_base.PaperPlot, ABC):
    section = 'revision'

class RevisionTableBase(plot_base.PaperTable, ABC):
    section = 'revision'


class RevisionBase(RevisionPlotBase, ABC):
    
    def run(self, loud=True):
        pass
        


class NormalizationPersistence(RevisionBase):
    description = ''
    name = 'NormalizationPersistence'

    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars,
        regular_chars, permnos, dates):
#         percentile_rank_chars = percentile_rank_chars[-100:]
#         return_panel = return_panel[-100:]
#         regular_chars = regular_chars[-100:]
#         dates = dates[-100:]
        
        
        normed_chars = np.copy(regular_chars)
        T, N, C = normed_chars.shape

        for c in range(C):
            for t in range(T):
                lower, upper = np.nanquantile(normed_chars[t,:,c], 0.05), np.nanquantile(normed_chars[t,:,c], 0.95)
                if np.isnan(lower):
                    lower = np.nanmin(normed_chars[t,:,c][~np.isinf(normed_chars[t,:,c])])
                if np.isnan(upper):
                    upper = np.nanmax(normed_chars[t,:,c][~np.isinf(normed_chars[t,:,c])])
                normed_chars[t,normed_chars[t,:,c] < lower,c] = lower
                normed_chars[t,normed_chars[t,:,c] > upper,c] = upper

                assert np.all(~np.isinf(normed_chars[t,:,c])), (lower, upper)

            mu, sigma = np.nanmean(normed_chars[:,:,c]), np.nanstd(normed_chars[:,:,c])
            assert ~np.isnan(mu) and ~np.isnan(sigma), (chars[c], mu, sigma, np.sum(~np.isnan(normed_chars[:,:,c])))
            assert ~np.isinf(mu) and ~np.isinf(sigma), (chars[c], mu, sigma, np.sum(~np.isnan(normed_chars[:,:,c])))

            normed_chars = (normed_chars - mu) / sigma
            
        def plot_autocors(char_panels, labels, dates, chars=None):
            fig = plt.figure(figsize=(15,10))
            ordering = None
            
            
            for char_panel, label in zip(char_panels, labels):
                T, N, C = char_panel.shape
                char_auto_corrs = []
                for c in range(C):
                    auto_corrs = []
                    for n in range(N):
                        if np.sum(~np.isnan(char_panel[:,n,c])) >= 60:
                            s = pd.Series(data = char_panel[:,n,c], index=dates)
                            first_idx = s.first_valid_index()
                            last_idx = s.last_valid_index()
                            s = s.loc[first_idx:last_idx]
                            auto_corrs.append((s.autocorr(lag=3), s.autocorr(lag=12)))
                    char_auto_corrs.append(np.nanmean(auto_corrs, axis=0))
                
                if ordering is None:
                    ordering = np.argsort(np.array(char_auto_corrs)[:,0])[::-1]
                
                plt.plot(np.arange(45), np.array(char_auto_corrs)[ordering,0], label=f'quarterly-{label}')
                plt.plot(np.arange(45), np.array(char_auto_corrs)[ordering,1], label=f'yearly-{label}')
            plt.legend()
            save_path = f'../images-pdfs/revision/'
            plt.xticks(np.arange(45), chars[ordering], rotation=90, fontsize=15)
            plt.savefig(save_path + f'auto_corrs.pdf', bbox_inches='tight')
            plt.show()
        
        plot_autocors([normed_chars, percentile_rank_chars], ["globally normed", 'percentile ranked'], dates, chars=chars)
        
        
        
class NonlinearIPCA(RevisionTableBase):
    description = ''
    name = 'NonlinearIPCA'
    sigfigs = 3

    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars, rts, expansion_dim):
        from models import FactorModel, PCAFactorModel, IpcaModelV2
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
            
            
            NUM_MONTHS_TRAIN = 280
            

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
        plt.savefig(f'../images-pdfs/revision/decile_ipca_sharpes_in_sample.pdf'.replace(' ', ''), 
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
        plt.savefig(f'../images-pdfs/revision/decile_ipca_sharpes_outof_sample.pdf'.replace(' ', ''), 
                            bbox_inches='tight')
        plt.show()
          
          
              
              
class TimeSeriesMoreLags(RevisionTableBase):
    description = ''
    name = 'TimeSeriesMoreLags'
    sigfigs = 3

    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars,
        regular_chars, permnos, dates, monthly_updates, char_map):
        
        def get_sufficient_statistics_lagged_last_vals(characteristics_panel, residuals, chars, nlags=1, 
                                               char_lag_maps=None, max_delta=None):
            T,N,L = characteristics_panel.shape
            sufficient_statistics,_ = imputation_model_simplified.get_sufficient_statistics_last_val(
                characteristics_panel, max_delta=None, residuals=residuals
            )

            lagged_sufficient_statistics = np.zeros((T,N,L,2*nlags), dtype=float)
            lagged_sufficient_statistics[:,:,:] = np.nan
            for i, c in enumerate(chars):
                lag_length = char_lag_maps[c]
                for lag in range(nlags):
                    lagged_sufficient_statistics[lag*lag_length:,:,i,2*lag:2*(lag+1)] = sufficient_statistics[:T-lag*lag_length,:,i]
            return lagged_sufficient_statistics
      
        def impute_chars_with_lags(xs_imputation, residuals, char_data, lags, monthly_update_mask, chars, char_maps,
                          eval_char_data=None, num_months_train=None, window_size=None):
    
            char_lags = {'Q': 3, 'M': 1, 'QM': 1, 'Y': 3, 'QY': 1}
            char_lag_maps = {k: char_lags[v] for k,v in char_maps.items()}

            suff_stats = get_sufficient_statistics_lagged_last_vals(char_data, residuals, chars, nlags=lags, 
                                                       char_lag_maps=char_lag_maps, max_delta=None)

            beta_weights = None


            imputed_chars = imputation_model_simplified.impute_beta_combined_regression(
                char_data, xs_imputation, sufficient_statistics=suff_stats, 
                beta_weights=None, constant_beta=False
            )
                
            return imputed_chars
        import imputation_model_simplified
        import importlib
        importlib.reload(imputation_model_simplified)
        results = []
        nfactors=10
        mask = None
        for lags in [3, 2, 1]:
            iter_results = []
            for i, (t1, t2) in enumerate([('MAR', '_out_of_sample_MAR'), 
                                          ('BLOCK', '_out_of_sample_block'), 
                                          ('logit', '_out_of_sample_logit')
                                         ]):

                eval_maps = {
                            '_out_of_sample_MAR': "MAR_eval_data",
                            '_out_of_sample_block': "prob_block_eval_data",
                            '_out_of_sample_logit': "logit_eval_data",
                            '_out_of_sample_logit3': "logit3_eval_data"
                        }

                eval_chars = imputation_utils.load_imputation(eval_maps[t2])


                fit_maps = {
                    'MAR': "MAR_fit_data",
                    'BLOCK': "prob_block_fit_data",
                    'logit': "logit_fit_data"
                }

                masked_lagged_chars = imputation_utils.load_imputation(fit_maps[t1])



                gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
                    char_panel=masked_lagged_chars, 
                    return_panel= return_panel, min_chars=10, K=20, 
                    num_months_train=percentile_rank_chars.shape[0],
                    reg=0.01,
                    time_varying_lambdas=False,
                    window_size=1, 
                    n_iter=1,
                    eval_data=None,
                    allow_mean=False)

                xs_imputed_data = imputation_model_simplified.get_all_xs_vals(masked_lagged_chars, reg=0.01, 
                                         Lmbda=lmbda, time_varying_lmbda=False)
                
                residuals = masked_lagged_chars - xs_imputed_data
                xs_imputed_data = impute_chars_with_lags(
                    xs_imputed_data, residuals, masked_lagged_chars,
                    lags, monthly_updates, chars, char_map,
                    eval_char_data=None, num_months_train=None, window_size=None
                )
                
                if mask is None:
                    mask = np.isnan(xs_imputed_data)
                
                xs_imputed_data[mask] = np.nan

                metrics = imputation_utils.get_imputation_metrics(xs_imputed_data, 
                                                      eval_char_data=eval_chars, monthly_update_mask=None, 
                                                      char_groupings=char_groupings, norm_func=None)   
                iter_results.append([round(np.sqrt(np.nanmean(x)), 5) for x in metrics])
            results.append(iter_results)
        
        new_results = [[], [], []]
        for result in results:
            for i in range(3):
                new_results[i] += result[i]
        colnames = [x + y for x in ['MAR-', 'BLOCK-', 'LOGIT'] for y in ['Agg', 'Q', 'M'] ]
        index = ['1 lag', '2 lags', '3 lags']
        self.data_df = pd.DataFrame(data=np.array(new_results), columns=colnames, index=index)
        
        
class ComparisonOfModelConfigs(RevisionTableBase):
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
        
        
        
class SparseFactors(RevisionBase):
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
            
            
class SparseFactorsTable(RevisionBase):
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
        
            
            
class ReturnFactorGeneralizedCorr(RevisionTableBase):
    
    description=''
    name = 'ReturnFactorGeneralizedCorr'
    sigfigs=3
    
    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars,
        regular_chars, permnos, dates, rts):
        

        def generalized_corr(lambda_1, lambda_2):
            N, r = lambda_1.shape
            rho = np.trace(np.linalg.solve(lambda_1.T @ lambda_1, lambda_1.T @ lambda_2) @ 
                                 np.linalg.solve(lambda_2.T @ lambda_2, lambda_2.T @ lambda_1))
            return rho, r, scp.linalg.subspace_angles(lambda_1, lambda_2)

        def visualize_gamma_corr(percentile_rank_chars, gamma_ts, Ks, returns, lmbda, rts):

            exess_returns = return_panel - np.array(rts).reshape([-1, 1])

            train_end=300
            start=0
            val_end = 350
            return_lag = 6
            factors = []
            all_metrics = []

            imputed = np.concatenate([np.expand_dims(x @ lmbda.T, axis=0) for x in gamma_ts], axis=0)
            imputed[~np.isnan(percentile_rank_chars)] = percentile_rank_chars[~np.isnan(percentile_rank_chars)]

            mask = np.logical_and(np.all(~np.isnan(imputed[:-return_lag, :, :]), axis=2),
                                ~np.isnan(return_panel[return_lag:]))


            factor_model = FactorModel.FactorRegressionModel()
            factor_model.include_intercept = False

            factor_model.fit(
                train_return=exess_returns[start+return_lag:train_end+return_lag], 
                train_individualFeature=gamma_ts[start:train_end, :, :], 
                train_mask=mask[start:train_end], 
                val_return=return_panel[train_end+return_lag:val_end+return_lag], 
                val_individualFeature=gamma_ts[train_end:val_end, :, :], 
                val_mask=mask[train_end:val_end],
                test_return=return_panel[val_end+return_lag:], 
                test_individualFeature=gamma_ts[val_end:-return_lag, :, :], 
                test_mask=mask[val_end:],
                rfts=np.array(rts[start+return_lag:]), missing_bounds=None, recalc_data=True, reg=0)

            gamma_t_factor_returns = factor_model.get_factors() 
            evs, evects = np.linalg.eigh(np.cov(gamma_t_factor_returns.T))
        #     gamma_loadings = evects[:,::-1][:,:K]

            w_gamma = np.linalg.solve(np.cov(gamma_t_factor_returns[:val_end].T), 
                                      np.mean(gamma_t_factor_returns[:val_end], axis=0))
            ret_gamma = gamma_t_factor_returns @ w_gamma


            results = []

            factor_model = FactorModel.FactorRegressionModel()
            factor_model.include_intercept = False

            factor_model.fit(
                train_return=exess_returns[start+return_lag:train_end+return_lag], 
                train_individualFeature=imputed[start:train_end, :, :], 
                train_mask=mask[start:train_end], 
                val_return=return_panel[train_end+return_lag:val_end+return_lag], 
                val_individualFeature=imputed[train_end:val_end, :, :], 
                val_mask=mask[train_end:val_end],
                test_return=return_panel[val_end+return_lag:], 
                test_individualFeature=imputed[val_end:-return_lag, :, :], 
                test_mask=mask[val_end:],
                rfts=np.array(rts[start+return_lag:]), missing_bounds=None, recalc_data=True, reg=0)
            CMP_factors = factor_model.get_factors()
            evs, evects = np.linalg.eigh(np.cov(CMP_factors))
            ordering = np.argsort(evs)[::-1]
            for k in Ks:
                CMP_PCA_factors = evects[:,ordering[:k]]

                w_PCA = np.linalg.solve(np.cov(CMP_factors[:val_end].T), np.mean(CMP_factors[:val_end], axis=0))
                ret_PCA = CMP_factors @ w_PCA
                results.append(generalized_corr(CMP_PCA_factors, gamma_t_factor_returns))

            return results

        import imputation_model_simplified
        gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
            char_panel=percentile_rank_chars, 
            return_panel= return_panel, min_chars=10, K=45, 
            num_months_train=percentile_rank_chars.shape[0],
            reg=0.01,
            time_varying_lambdas=False,
            window_size=1, 
            n_iter=1,
            eval_data=None,
            allow_mean=False)


        results = []
        for k in range(2, 46):
            k_results = visualize_gamma_corr(percentile_rank_chars, gamma_ts[:,:,:k], Ks=range(2, 46), 
                                                      returns=return_panel, lmbda=lmbda[:,:k], rts=rts)
            results.append(k_results)
        factor_nums = np.arange(2, 46)
        self.data_df = pd.DataFrame(data=[[round(x[0], 3) for x in y] for y in results], index=factor_nums, columns=factor_nums)
        
        plot_data = np.diag([[round(x[0], 3) for x in y] for y in results])
        plot_data = plot_data / factor_nums
        fig = plt.figure(figsize=(15,10))
        plt.bar(factor_nums, plot_data)
        ax = plt.gca()
        ax.set_xticks(factor_nums)
        ax.set_xlabel('Number of factors')
        ax.set_ylabel('Generalized correlation')
        tick_names = [f"{x}" for x in factor_nums]
        ax.set_xticklabels(tick_names, rotation=65)
        save_path = f'../images-pdfs/revision/'
        plt.savefig(save_path + f'generalized_corr_barplot.pdf', bbox_inches='tight')
        plt.show()
        
        
class ReturnFactorSharpes(RevisionTableBase):
    
    description=''
    name = 'ReturnFactorSharpes'
    sigfigs=3
    
    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars,
        regular_chars, permnos, dates, rts):
        return_lag = 6
        import imputation_model_simplified
        
        exess_returns = return_panel - np.array(rts).reshape([-1, 1])
        train_end=300
        start=0
        val_end = 350
        return_lag = 6
        factors = []
        all_metrics = []
        
        gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
            char_panel=percentile_rank_chars, 
            return_panel= return_panel, min_chars=10, K=45, 
            num_months_train=percentile_rank_chars.shape[0],
            reg=0.01,
            time_varying_lambdas=False,
            window_size=1, 
            n_iter=1,
            eval_data=None,
            allow_mean=False)
        
    
        imputed = np.concatenate([np.expand_dims(x @ lmbda.T, axis=0) for x in gamma_ts], axis=0)
        imputed[~np.isnan(percentile_rank_chars)] = percentile_rank_chars[~np.isnan(percentile_rank_chars)]

        
        normalizer = np.linalg.norm(lmbda, axis=0, keepdims=True)
        gamma_ts, lmbda = gamma_ts * normalizer, lmbda / normalizer
        
        mask = np.logical_and(np.all(~np.isnan(imputed[:-return_lag, :, :]), axis=2),
                            ~np.isnan(return_panel[return_lag:]))
        
        factor_model = FactorModel.FactorRegressionModel()
        factor_model.include_intercept = False
        
        factor_model.fit(
                train_return=exess_returns[start+return_lag:train_end+return_lag], 
                train_individualFeature=imputed[start:train_end, :, :], 
                train_mask=mask[start:train_end], 
                val_return=return_panel[train_end+return_lag:val_end+return_lag], 
                val_individualFeature=imputed[train_end:val_end, :, :], 
                val_mask=mask[train_end:val_end],
                test_return=return_panel[val_end+return_lag:], 
                test_individualFeature=imputed[val_end:-return_lag, :, :], 
                test_mask=mask[val_end:],
                rfts=np.array(rts[start+return_lag:]), missing_bounds=None, recalc_data=True, reg=0)
        
        CMP_factors = factor_model.get_factors()
        cmp_evs, cmp_evects = np.linalg.eigh(np.cov(CMP_factors.T))
        cmp_evs, cmp_evects = cmp_evs[::-1], cmp_evects[:,::-1]
        
        sharpes = []
        for k in range(2, 46):
            CMP_k_factors = CMP_factors @ cmp_evects[:,:k]
            
            factor_model = FactorModel.FactorRegressionModel()
            factor_model.include_intercept = False
            factor_model.fit(
                train_return=exess_returns[start+return_lag:train_end+return_lag], 
                train_individualFeature=gamma_ts[start:train_end, :, :k], 
                train_mask=mask[start:train_end], 
                val_return=return_panel[train_end+return_lag:val_end+return_lag], 
                val_individualFeature=gamma_ts[train_end:val_end, :, :k], 
                val_mask=mask[train_end:val_end],
                test_return=return_panel[val_end+return_lag:], 
                test_individualFeature=gamma_ts[val_end:-return_lag, :, :k], 
                test_mask=mask[val_end:],
                rfts=np.array(rts[start+return_lag:]), missing_bounds=None, recalc_data=True, reg=0)

            gamma_t_factor_returns = factor_model.get_factors() 
            evs, evects = np.linalg.eigh(np.cov(gamma_t_factor_returns.T))
                
            sigma, mu = np.cov(gamma_t_factor_returns[:val_end].T), np.mean(gamma_t_factor_returns[:val_end], axis=0)
#             w_gamma = np.linalg.solve(sigma.T @ sigma + 1e-5 * np.eye(k), sigma.T @ mu)
            w_gamma = np.linalg.solve(sigma, mu)
            ret_gamma = gamma_t_factor_returns @ w_gamma
            
            sigma, mu = np.cov(CMP_k_factors[:val_end,:k].T), np.mean(CMP_k_factors[:val_end], axis=0)
#             w_cmp = np.linalg.solve(sigma.T @ sigma + 1e-5 * np.eye(k), sigma.T @ mu)
            w_cmp = np.linalg.solve(sigma, mu)
            ret_cmp = CMP_k_factors @ w_cmp
            
            sharpes.append(
                (np.mean(ret_gamma[:val_end]) / np.std(ret_gamma[:val_end]), 
                 np.mean(ret_gamma[val_end:]) / np.std(ret_gamma[val_end:]), 
                 np.mean(ret_cmp[:val_end]) / np.std(ret_cmp[:val_end]),
                 np.mean(ret_cmp[val_end:]) / np.std(ret_cmp[val_end:]),
                ))
            
     
        fig = plt.figure(figsize=(15,10))
        plt.plot(np.arange(2, 46), [x[0] for x in sharpes], label='characteristic factors IS sharpe')
        plt.plot(np.arange(2, 46), [x[1] for x in sharpes], label='characteristic factors OOS sharpe')
        plt.plot(np.arange(2, 46), [x[2] for x in sharpes], label='return factors IS sharpe')
        plt.plot(np.arange(2, 46), [x[3] for x in sharpes], label='return factors OOS sharpe')
        ax = plt.gca()
        ax.set_xticks(np.arange(2, 46))
        ax.set_xlabel('Number of factors')
        ax.set_ylabel('Sharpe ratio')
        tick_names = [f"{x}" for x in np.arange(2, 46)]
        ax.set_xticklabels(tick_names, rotation=65)
        plt.legend()
        save_path = f'../images-pdfs/revision/'
        plt.savefig(save_path + f'return_factor_sharpes.pdf', bbox_inches='tight')
        plt.show()
        
        

        
from joblib import Parallel, delayed
class RegularizationOfBetaRegression(RevisionTableBase):
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

        

                      
                 