from unicodedata import name
from plots_and_tables import plot_base
from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
import imputation_utils
import numpy as np
from plots_and_tables import section_6_utils
from models import FactorModel, IpcaModelV2

mycolors = ['#152eff', '#e67300', '#0e374c', '#6d904f', '#8b8b8b', '#30a2da', '#e5ae38', '#fc4f30', '#6d904f', '#8b8b8b', '#0e374c']
c = mycolors[:4]

class SectionSixPlotBase(plot_base.PaperPlot, ABC):
    section = 'section6'

class SectionSixTableBase(plot_base.PaperTable, ABC):
    section = 'section6'


class SectionSixBase(SectionSixPlotBase, ABC):
    
    def run(self, loud=True):
        pass
    
class ConditionaMeanReturns:
    
    def run(self, percentile_rank_chars, chars, regular_chars, return_panel):
        
        port_tgt_chars = [
            'A2ME', 
         'AT', 
         'ATO',
         'B2M',
         'C2A',
            'CF2B',
            'CF2P',
            'CTO',
            'D2A',
            'DPI2A', 
            'E2P', 
            'FC2Y',
            'INV', 
            'LEV',
            'NI',
            'NOA',
            'OA', 
            'OL',
               'OP', 
            'PCM',
            'PM',
            'PROF',
            'Q',
            'RNA',
            'ROA',
            'ROE',
            'S2P',
            'SGA2S',
        ]

        
        port_returns = [[] for _ in chars]
        port_counts = [[] for _ in chars]
        size_ind = np.argwhere(chars == 'ME')[0][0]
        start = 45
        prev_obs_mask = np.any(~np.isnan(percentile_rank_chars[:start]), axis=0)
        for t in range(start, percentile_rank_chars.shape[0] - 6):
            cut = np.nanquantile(regular_chars[t,:,size_ind], q=.95)
            cut = np.nanmax(regular_chars[t,:,size_ind])
            sizes = np.nan_to_num(regular_chars[t,:,size_ind])
            sizes[sizes > cut] = cut
            for i, c in enumerate(chars):
                if c in port_tgt_chars:
                    p1 = np.logical_and(~np.isnan(percentile_rank_chars[t,:,i]) , ~np.isnan(regular_chars[t,:,size_ind]))
                    p1 = np.logical_and(p1, ~np.isnan(return_panel[t+6]))
                    p1_count = np.sum(p1)
                    p1 = p1 * sizes
                    p1 = p1 / np.sum(p1)

                    p2 = np.logical_and(np.isnan(percentile_rank_chars[t,:,i]), prev_obs_mask[:,i]) 
                    p2 = np.logical_and(p2, ~np.isnan(return_panel[t+6]))
                    p2 = np.logical_and(p2, ~np.isnan(regular_chars[t,:,size_ind]))
                    p2_count = np.sum(p2)
                    p2 = p2 * sizes
                    p2 = p2 / np.sum(p2)

                    p1_ret = p1 @ np.nan_to_num(return_panel[t+6])
                    p2_ret = p2 @ np.nan_to_num(return_panel[t+6])

                    port_returns[i].append([p1_ret, p2_ret])
                    port_counts[i].append(p2_count)
        #             if p1_count > 5 and p2_count > 10:
        #                 port_returns[i].append([p1_ret, p2_ret])
        #             else:
        #                 port_returns[i].append([p1_ret, np.nan])
                else:
                    port_returns[i].append([p1_ret, np.nan])
            prev_obs_mask = np.logical_or(prev_obs_mask, ~np.isnan(percentile_rank_chars[t]))
        
        mean_returns = np.mean(port_returns, axis=1)
        mycolors = ['#152eff', '#e67300', '#0e374c', '#6d904f', '#8b8b8b', '#30a2da', '#e5ae38', '#fc4f30', 
                    '#6d904f', '#8b8b8b', '#0e374c']
        
        fig = plt.figure(figsize=(15,10))
        ordering = np.argsort(mean_returns[:,1])
        _ = plt.plot(np.arange(len(port_tgt_chars)), 1200*mean_returns[:,0][ordering][:len(port_tgt_chars)], 
                     label='observed',
                    c=mycolors[2], marker='o')
        _ = plt.plot(np.arange(len(port_tgt_chars)), 1200*mean_returns[:,1][ordering][:len(port_tgt_chars)], 
                     label='missing, not at start',
                    c=mycolors[1], marker='o')
        _ = plt.legend()
        _ = plt.xticks(np.arange(len(port_tgt_chars)), chars[ordering][:len(port_tgt_chars)], rotation=90, fontsize=15)
        _ = plt.ylabel("Mean returns (% per an.)")
        plt.savefig(f'../images-pdfs/section6/ls-missing-obs-ports.pdf', 
                                    bbox_inches='tight')
        plt.show()
        

class UnivariateSort(SectionSixBase):
    description = ''
    name = 'UnivariateSort'

    def setup(self, percentile_rank_chars, return_panel, char_groupings, chars,
        regular_chars, permnos, size_ind, dates):
           
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})
        
        shares = np.load("/home/svenl/data/pelger/data/shares_outstanding.npz")["shrout"]
        shrout_permnos = np.load("/home/svenl/data/pelger/data/shares_outstanding.npz")["permnos"]
        shares = shares[-1*percentile_rank_chars.shape[0]:,np.isin(shrout_permnos, permnos)]
        sizes = regular_chars[:,:,size_ind]
        nyse_mask = section_6_utils.get_nyse_permnos_mask(dates, permnos, percentile_rank_chars)

        imputed_data = imputation_utils.load_imputation("global_bw_in_sample")
        cfa_data =np.copy(percentile_rank_chars)
        cfa_data[np.isnan(cfa_data)] = imputed_data[np.isnan(cfa_data)]

        imputed_data = imputation_utils.load_imputation("global_xs_in_sample")
        cfa_data[np.isnan(cfa_data)] = imputed_data[np.isnan(cfa_data)]
        
        nyse_10buckets = section_6_utils.get_buckets_nyse_cutoffs(nyse_mask, percentile_rank_chars, 
                                 percentile_rank_chars, 10)
        imputed_nyse_10buckets = section_6_utils.get_buckets_nyse_cutoffs(nyse_mask, cfa_data, 
                                percentile_rank_chars, 10)

        base_chars = ["ME", "B2M", "INV", "CF2P", "NI", "SGA2S", "R12_2", "R60_13"]
        ff_chars = ["B2M", "ME", "OP", "INV"]
        for i, c in enumerate(ff_chars):
            char_ind = np.argwhere(chars == c)[0][0]
            observed_masks = []
            imputed_masks = []
            labels = []
            title = f"{c}"
            for dec in [0, 9]:
                
                observed_masks.append(nyse_10buckets[:,:,char_ind, dec])
                imputed_masks.append(imputed_nyse_10buckets[:,:,char_ind, dec])
                label = f"Decile {dec + 1}"
                labels.append(label)
            chars_to_plot = [x for x in base_chars if x != c]
            section_6_utils.plot_page_multiple_maps(percentile_rank_chars, observed_masks, imputed_data, imputed_masks, 
                chars_to_plot, return_panel, sizes, chars, title, shares, labels)


class UnivariateBarDiffs(SectionSixBase):
    description = ''
    name = 'UnivariateBarDiffs'

    def setup(self, percentile_rank_chars, chars, return_panel, char_groupings,
        regular_chars, dates, permnos):
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})
        
        nyse_mask = section_6_utils.get_nyse_permnos_mask(dates, permnos, percentile_rank_chars)
        nyse_10buckets = section_6_utils.get_buckets_nyse_cutoffs(nyse_mask, percentile_rank_chars, 
                                 percentile_rank_chars, 10)
        size_ind = np.argwhere(chars=='ME')[0][0]
        sizes = regular_chars[:,:,size_ind]

        h_port_returns = []
        l_port_returns = []
        full_mask = np.all(~np.isnan(percentile_rank_chars), axis=2)
        for i, c in enumerate(chars):
            char_ind = np.argwhere(chars == c)[0][0]
            char_mask = ~np.isnan(percentile_rank_chars[:,:,char_ind])
            specific_h_mask = np.logical_and(nyse_10buckets[:,:,char_ind,9], char_mask)
            specific_l_mask = np.logical_and(nyse_10buckets[:,:,char_ind,0], char_mask)
            
            fully_h_mask = np.logical_and(nyse_10buckets[:,:,char_ind,9], full_mask)
            fully_l_mask = np.logical_and(nyse_10buckets[:,:,char_ind,0], full_mask)
            
            h_port_returns.append((section_6_utils.get_vw_portfolio_returns(return_panel, specific_h_mask, 4, sizes=sizes),
                                section_6_utils.get_vw_portfolio_returns(return_panel, fully_h_mask, 4, sizes=sizes)))
            
            l_port_returns.append((section_6_utils.get_vw_portfolio_returns(return_panel, specific_l_mask, 4, sizes=sizes),
                                section_6_utils.get_vw_portfolio_returns(return_panel, fully_l_mask, 4, sizes=sizes)))
            
            

        h_port_mean_returns = []
        l_port_mean_returns = []
        h_port_sharpes = []
        l_port_sharpes = []

        for j, c in enumerate(chars):
            h_port_mean_returns.append((np.mean(h_port_returns[j][0][0]),
                                    np.mean(h_port_returns[j][1][0])))

            l_port_mean_returns.append((np.mean(l_port_returns[j][0][0]),
                                    np.mean(l_port_returns[j][1][0])))

            h_port_sharpes.append((h_port_mean_returns[j][0] / np.std(h_port_returns[j][0][0]),
                                h_port_mean_returns[j][1] / np.std(h_port_returns[j][1][0])))

            l_port_sharpes.append((l_port_mean_returns[j][0] / np.std(l_port_returns[j][0][0]),
                                l_port_mean_returns[j][1] / np.std(l_port_returns[j][1][0])))

        data = np.hstack([h_port_mean_returns, l_port_mean_returns, h_port_sharpes, l_port_sharpes])
        columns = ["H mean", "H mean fully observed", "L mean", "L mean fully observed", 
          "H sharpe", "H mean sharpe observed", "L sharpe", "L sharpe fully observed"]


        section_6_utils.bar_plot_mean_sharpe_diffs_seq(h_port_returns, l_port_returns, "Past Returns, Value", [char_groupings['Past Returns'],
                                                                      char_groupings['Value']], 'S',
                              group_names =['Past Returns', 'Value'], h_port_mean_returns=h_port_mean_returns, 
                              l_port_mean_returns=l_port_mean_returns, chars=chars)
        section_6_utils.bar_plot_mean_sharpe_diffs_seq(h_port_returns, l_port_returns, "Past Returns, Value", [char_groupings['Past Returns'],
                                                                            char_groupings['Value']], 'M',
                                    group_names =['Past Returns', 'Value'], h_port_mean_returns=h_port_mean_returns, 
                              l_port_mean_returns=l_port_mean_returns, chars=chars)

        section_6_utils.bar_plot_mean_sharpe_diffs_seq(h_port_returns, l_port_returns, "Investment, Profitability", [char_groupings['Investment'],
                                                                                                    char_groupings['Profitability']], 'S',
                                    group_names =['Investment', 'Profitability'], h_port_mean_returns=h_port_mean_returns, 
                              l_port_mean_returns=l_port_mean_returns, chars=chars)
        section_6_utils.bar_plot_mean_sharpe_diffs_seq(h_port_returns, l_port_returns, "Investment, Profitability", [char_groupings['Investment'],
                                                                                                    char_groupings['Profitability']], 'M',
                                    group_names =['Investment', 'Profitability'], h_port_mean_returns=h_port_mean_returns, 
                              l_port_mean_returns=l_port_mean_returns, chars=chars)
        section_6_utils.bar_plot_mean_sharpe_diffs_seq(h_port_returns, l_port_returns, "Intangibles, Trading Frictions, Other", [char_groupings['Intangibles'],
                                                                            char_groupings['Trading Frictions'],
                                                                                    char_groupings['Other']],
                                    'S',
                                    group_names =['Intangibles', 'Trading Frictions', 'Other'], h_port_mean_returns=h_port_mean_returns, 
                              l_port_mean_returns=l_port_mean_returns, chars=chars)
        section_6_utils.bar_plot_mean_sharpe_diffs_seq(h_port_returns, l_port_returns, "Intangibles, Trading Frictions, Other", [char_groupings['Intangibles'],
                                                                            char_groupings['Trading Frictions'],
                                                                                    char_groupings['Other']],
                                    'M',
                                    group_names =['Intangibles', 'Trading Frictions', 'Other'], h_port_mean_returns=h_port_mean_returns, 
                              l_port_mean_returns=l_port_mean_returns, chars=chars)



class PurePlayRegressionsMasked(SectionSixBase):
    description = ''
    name = 'PurePlayRegressionsMasked'

    def setup(self, percentile_rank_chars, chars, return_panel, char_groupings,
        regular_chars, dates, permnos, rts, char_map, monthly_updates):
        char_mask = chars != "Q"
        chars = chars[char_mask]
        masked_lagged_chars = imputation_utils.load_imputation("logit_fit_data")[:,:,char_mask]
        percentile_rank_chars = percentile_rank_chars[:,:,char_mask]
        start_idx = 0
        exess_returns = return_panel - np.array(rts).reshape([-1, 1])
        gt_nan_mask = np.isnan(percentile_rank_chars)
        update_chars = np.copy(percentile_rank_chars)
        for i, c in enumerate(chars):
            if char_map[c] !='M':
                update_chars[~(monthly_updates == 1),i] = np.nan
        
        global_xs_reg = imputation_utils.load_imputation("local_xs_out_of_sample_logit")[:,:,char_mask]
        global_bw_reg = imputation_utils.load_imputation("local_bw_out_of_sample_logit")[:,:,char_mask]
        
        global_bw_reg[np.isnan(global_bw_reg)] = global_xs_reg[np.isnan(global_bw_reg)]
        replace_mask = ~np.isnan(masked_lagged_chars)
        global_bw_reg[replace_mask] = masked_lagged_chars[replace_mask]
        global_xs_reg[replace_mask] = masked_lagged_chars[replace_mask]
        global_xs_reg[gt_nan_mask] = np.nan
        global_bw_reg[gt_nan_mask] = np.nan

        xs_median = np.zeros_like(global_bw_reg)
        xs_median[replace_mask] = masked_lagged_chars[replace_mask]
        xs_median[gt_nan_mask] = np.nan

        global_ts = imputation_utils.load_imputation("global_ts_out_of_sample_logit")[:,:,char_mask]
        global_ts[replace_mask] = masked_lagged_chars[replace_mask]
        to_impute_zero = np.logical_and(np.isnan(global_ts), ~np.isnan(xs_median))
        global_ts[to_impute_zero] = xs_median[to_impute_zero]
        global_ts[gt_nan_mask] = np.nan

        factors = []
        all_metrics = []

        start=0
        val_end = int(len(percentile_rank_chars) / 2 + len(percentile_rank_chars) / 4)
        train_end=int(len(percentile_rank_chars) / 2)
        return_lag = 6

        mask = np.logical_and(np.all(~np.isnan(percentile_rank_chars[:-return_lag, :, :]), axis=2),
                            ~np.isnan(return_panel[return_lag:]))

        factor_model = FactorModel.FactorRegressionModel()
        factor_model.include_intercept = False
        factor_model.fit(exess_returns[start+return_lag:train_end+return_lag], 
                        percentile_rank_chars[start:train_end, :, :], 
                        mask[start:train_end], 
                        return_panel[train_end+return_lag:val_end+return_lag], 
                        percentile_rank_chars[train_end:val_end, :, :], 
                        mask[train_end:val_end],
                        return_panel[val_end+return_lag:], 
                        percentile_rank_chars[val_end:-return_lag, :, :], 
                        mask[val_end:],
                        np.array(rts[start+return_lag:]), missing_bounds=None, recalc_data=True, reg=0.1)


        factors.append(factor_model.get_factors())

        all_metrics.append(factor_model.eval_metrics(None, None, None, recalc_data=False))

        train_end=int(len(exess_returns) / 2)
        start=0
        val_end = int(len(exess_returns) / 2) + int(len(exess_returns) / 4)
        return_lag = 6

        factor_model = FactorModel.FactorRegressionModel()
        factor_model.include_intercept = False

        mask = np.logical_and(np.all(~np.isnan(global_bw_reg[:-return_lag, :, :]), axis=2),
                            ~np.isnan(return_panel[start+return_lag:]))

        factor_model.fit(exess_returns[start+return_lag:train_end+start+return_lag], global_bw_reg[:train_end, :, :], 
                        mask[:train_end], 
                        return_panel[start+train_end+return_lag:val_end+return_lag+start], 
                        global_bw_reg[train_end:val_end, :, :], mask[train_end:val_end],
                        return_panel[val_end+return_lag+start:], 
                        global_bw_reg[val_end:-return_lag, :, :], mask[val_end:],
                        np.array(rts[start+return_lag:]), missing_bounds=None, recalc_data=True, reg=0.1)


        factors.append(factor_model.get_factors())

        all_metrics.append(factor_model.eval_metrics(None, None, None, recalc_data=False))

        factor_model = FactorModel.FactorRegressionModel()
        factor_model.include_intercept = False
        factor_model.fit(exess_returns[start+return_lag:train_end+start+return_lag], xs_median[:train_end, :, :], 
                        mask[:train_end], 
                        return_panel[start+train_end+return_lag:val_end+return_lag+start], 
                        xs_median[train_end:val_end, :, :], mask[train_end:val_end],
                        return_panel[val_end+return_lag+start:], 
                        xs_median[val_end:-return_lag, :, :], mask[val_end:],
                        np.array(rts[start+return_lag:]), missing_bounds=None, recalc_data=True, reg=0.1)




        factors.append(factor_model.get_factors())

        all_metrics.append(factor_model.eval_metrics(None, None, None, recalc_data=False))
        mask_start = 0

        corrs = []
        r_2 = []
        mean_errors = []
        def r_Squared(x,y):
            return 1 - np.sum(np.square(x-y)) / np.sum(np.square(x - np.mean(x)))
        error_maps = {}
        for j, name in enumerate(chars):
            corrs.append([])
            r_2.append([])
            mean_errors.append([])
            plt.figure(figsize=(7, 5))
            fo = None
            for i, label in enumerate(["Fully Observed", "B-XS",  "XS-Median"]):
                
                if label == "Fully Observed":
                    fo = factors[i][:,j]
                    plt.plot(np.cumsum(factors[i][:,j]), label=label)
                else:
                    corrs[-1].append(np.corrcoef(fo[mask_start:], factors[i][mask_start:,j])[0][1])
                    r_2[-1].append(r_Squared(fo[mask_start:], factors[i][mask_start:,j]))
                    mean_errors[-1].append((np.mean(factors[i][mask_start:,j] - fo[mask_start:]), 
                                            np.sqrt(np.mean(np.square(fo[mask_start:]- factors[i][mask_start:,j])))))
                    plt.plot(np.cumsum(factors[i][:,j]), label=label)
                    error_maps[(name, label)] =  f"rp-error: {mean_errors[-1][-1][-1]:.2g} corr: {corrs[-1][-1]:.2g}"
                                    
            plt.legend(framealpha=0.5)
            plt.title(name)
            save_path = f'../images-pdfs/section6/masked-factor_regression-pure_play-{name}-bw-xsmed.pdf'
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
            plt.show()
        
        xmargin = 0.025
        tick_fontsize = 14
        legend_fontsize= 14
        plt.figure(figsize=(10, 4))
        ordering = np.argsort([np.abs(x[0][0]) for x in mean_errors])
        plt.plot(np.array([np.abs(x[0][0]) for x in mean_errors])[ordering], label='B-XS')
        plt.plot(np.array([np.abs(x[1][0])for x in mean_errors])[ordering], label='XS-Median')
        plt.xticks(np.arange(44), labels=chars[ordering], rotation=90)
        plt.minorticks_off()
        plt.gca().tick_params(axis='both', which='major', labelsize=tick_fontsize)
        plt.legend(framealpha=0.5, fontsize=legend_fontsize)

        plt.margins(x=xmargin)
        save_path = f'../images-pdfs/section6/masked-factor_regression-pure_play-bw-xsmed-mean-abs-error.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
        plt.figure(figsize=(10, 4))
        ordering = np.argsort([np.abs(x[0][1]) for x in mean_errors])
        plt.plot(np.array([np.abs(x[0][1]) for x in mean_errors])[ordering], label='B-XS')
        plt.plot(np.array([np.abs(x[1][1]) for x in mean_errors])[ordering], label='XS-Median')
        plt.xticks(np.arange(44), labels=chars[ordering], rotation=90)
        plt.minorticks_off()
        plt.gca().tick_params(axis='both', which='major', labelsize=tick_fontsize)
        plt.legend(framealpha=0.5, fontsize=legend_fontsize)

        plt.margins(x=xmargin)
        save_path = f'../images-pdfs/section6/masked-factor_regression-pure_play-bw-xsmed-rmse.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
        plt.figure(figsize=(10, 4))
        ordering = np.argsort([x[0] for x in r_2])
        plt.plot(np.array([x[0] for x in r_2])[ordering], label='B-XS')
        plt.plot(np.array([x[1] for x in r_2])[ordering], label='XS-Median')
        plt.xticks(np.arange(44), labels=chars[ordering], rotation=90)
        plt.minorticks_off()
        plt.gca().tick_params(axis='both', which='major', labelsize=tick_fontsize)
        plt.legend(framealpha=0.5, fontsize=legend_fontsize)

        plt.margins(x=xmargin)
        save_path = f'../images-pdfs/section6/masked-factor_regression-pure_play-bw-xsmed-r2.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
        pass
    
class IPCASharpeDiff(SectionSixTableBase):
    description = ''
    name = 'IPCASharpeDiff'
    sigfigs = 3
    
    def get_sharpes(self, percentile_rank_chars, imputed_chars, chars, return_panel, char_groupings,
        regular_chars, dates, permnos, rts, char_map, monthly_updates, nfactor):
        
        rts = np.array(rts)
        NUM_MONTHS_TRAIN = int(len(percentile_rank_chars) / 2)
        return_lag = 6
        
#         imputed_chars = imputation_utils.load_imputation('../../data_07_19/imputation_cache/global_xs_in_sample')
        
        percentile_rank_chars = percentile_rank_chars[:-return_lag]
        imputed_chars = imputed_chars[:-return_lag]
        return_panel = return_panel[return_lag:] - rts[return_lag:].reshape(-1, 1)
        rts = rts[return_lag:] * 0 
        
        ipca_fo = IpcaModelV2.IpcaModel("fully_observed", 
                           num_factors=nfactor, num_chars=45, iter_tol=1e-10, maxiter=10000,
                                       alpha=0.0, l1_ratio=0)
        
        ipca_imputed = IpcaModelV2.IpcaModel("imputed", 
                           num_factors=nfactor, num_chars=45, iter_tol=1e-10, maxiter=100000,
                                       alpha=0.0, l1_ratio=0)
        
        mask_fo = np.logical_and(np.all(~np.isnan(percentile_rank_chars), axis=2), ~np.isnan(return_panel))
        mask_imputed = np.logical_and(np.all(~np.isnan(imputed_chars), axis=2), ~np.isnan(return_panel))
        
        ipca_fo.fit(return_panel, char_panel=percentile_rank_chars, 
                            masks=mask_fo, num_months_train=NUM_MONTHS_TRAIN)
        ipca_imputed.fit(return_panel, char_panel=imputed_chars, 
                    masks=mask_imputed, num_months_train=NUM_MONTHS_TRAIN)
        
        from models import ipca_util 
        
        fp_np_factors = np.copy(ipca_fo.in_factors)
        imputed_np_factors = np.copy(ipca_imputed.in_factors)
        fp_oos_mv_portfolio_returns = []
        imputed_oos_mv_portfolio_returns = []
        for t in range(len(ipca_fo.out_factors)):
            np_risk_free_rates = rts[t:NUM_MONTHS_TRAIN+t]

            weights = ipca_util.calculate_efficient_portofolio(fp_np_factors[:,t:], np_risk_free_rates.squeeze())
            ft = ipca_fo.out_factors[t]
            oos_return = weights.T.dot(ft)
            fp_oos_mv_portfolio_returns.append(oos_return)
            fp_np_factors = np.concatenate([fp_np_factors, ft], axis=1)
            
            weights = ipca_util.calculate_efficient_portofolio(imputed_np_factors[:,t:], np_risk_free_rates.squeeze())
            ft = ipca_imputed.out_factors[t]
            oos_return = weights.T.dot(ft)
            imputed_oos_mv_portfolio_returns.append(oos_return)
            
            imputed_np_factors = np.concatenate([imputed_np_factors, ft], axis=1)
        
        for i in range(2, nfactor+1):
            fp_is_weights = ipca_util.calculate_efficient_portofolio(ipca_fo.in_factors[:i,:],  rts[:NUM_MONTHS_TRAIN].squeeze())
            fp_is_returns = fp_is_weights.T.dot(ipca_fo.in_factors[:i,:])
            
            imputed_is_weights = ipca_util.calculate_efficient_portofolio(ipca_imputed.in_factors[:i,:],
                                                                rts[:NUM_MONTHS_TRAIN].squeeze())
            imputed_is_returns = imputed_is_weights.T.dot(ipca_imputed.in_factors[:i,:])
            
        fp_sharpes = (ipca_util.calculate_sharpe_ratio(fp_is_returns -  rts[:NUM_MONTHS_TRAIN]),
                     ipca_util.calculate_sharpe_ratio(np.array(fp_oos_mv_portfolio_returns) - rts[NUM_MONTHS_TRAIN:]))
        
        imputed_sharpes = (ipca_util.calculate_sharpe_ratio(imputed_is_returns - rts[:NUM_MONTHS_TRAIN]),
                     ipca_util.calculate_sharpe_ratio(np.array(imputed_oos_mv_portfolio_returns) - rts[NUM_MONTHS_TRAIN:]))
        
        return fp_sharpes, imputed_sharpes
    
    def setup(self, percentile_rank_chars, chars, return_panel, char_groupings,
        regular_chars, dates, permnos, rts, char_map, monthly_updates):
        imputed_chars = imputation_utils.load_imputation('local_bw_in_sample')
        imputed_chars[~np.isnan(percentile_rank_chars)] = percentile_rank_chars[~np.isnan(percentile_rank_chars)]
        mask = np.isnan(imputed_chars)
        imputed_chars[mask] = imputation_utils.load_imputation('local_xs_in_sample')[mask]
        
        sharpes = []
        num_elemets = 7
        for i in range(3, 3 + num_elemets):
            sharpes.append(self.get_sharpes(percentile_rank_chars, imputed_chars, chars, return_panel, char_groupings,
                regular_chars, dates, permnos, np.array(rts), char_map, monthly_updates, nfactor=i))

        x_locs = [i*3 + j for i in range(num_elemets) for j in range(2)]
        is_heights = sum([[sharpes[i][0][0], sharpes[i][1][0]] for i in range(num_elemets)], [])
        oos_heights = sum([[sharpes[i][0][1], sharpes[i][1][1]] for i in range(num_elemets)], [])
        colors = sum([['orange', 'blue'] for i in range(num_elemets)], [])
        
        
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
        plt.savefig(f'../images-pdfs/section6/ipca_sharpes_in_sample.pdf'.replace(' ', ''), 
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
        plt.savefig(f'../images-pdfs/section6/ipca_sharpes_outof_sample.pdf'.replace(' ', ''), 
                            bbox_inches='tight')
        plt.show()
