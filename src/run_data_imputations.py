import missing_data_utils
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import imputation_utils, data_loading, imputation_model_simplified
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
from matplotlib import patches


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


def run_all_imputations(missing_data_type, percentile_rank_chars, dates, return_panel, chars, industries,
                        monthly_updates
                       ):
    
    fit_maps = {
        'MAR': "MAR_fit_data",
        'logit': "logit_fit_data",
        'PROB_BLOCK': "prob_block_fit_data"
    }
    
    tag_maps = {
        None: "_in_sample",
        'MAR': "_out_of_sample_MAR",
        'logit': "_out_of_sample_logit",
        'PROB_BLOCK': "_out_of_sample_block"
    }
    T = percentile_rank_chars.shape[0]
    tag = tag_maps[missing_data_type]
    
    if missing_data_type in fit_maps:
        masked_lagged_chars = imputation_utils.load_imputation(fit_maps[missing_data_type])
    else:
        masked_lagged_chars = percentile_rank_chars
    update_chars = None
    
    update_chars = np.copy(masked_lagged_chars)
    for i, c in enumerate(chars):
        if char_map[c] !='M':
            update_chars[~(monthly_updates == 1),i] = np.nan
        
        
#     Global
    
    global_ts = imputation_model_simplified.impute_chars(
        masked_lagged_chars, None, None, 
        suff_stat_method='last_val', 
        constant_beta=True, beta_weight=False
    )
    imputation_utils.save_imputation(global_ts, dates, permnos, chars, "global_ts" + tag)
    del global_ts
    
    base_path = '../data/imputation_cache/'

    gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
            char_panel=masked_lagged_chars, 
            return_panel=np.nan_to_num(return_panel), min_chars=1, K=20, 
            num_months_train=percentile_rank_chars.shape[0],
            reg=0.01,
            time_varying_lambdas=False,
            window_size=548, 
            n_iter=1,
            eval_data=None,
            allow_mean=False)

    oos_xs_imputations = imputation_model_simplified.get_all_xs_vals(masked_lagged_chars, reg=0.01, 
                                     Lmbda=lmbda, time_varying_lmbda=False)
    residuals = masked_lagged_chars - oos_xs_imputations

    xs_imputations = np.concatenate([np.expand_dims(g @ lmbda.T, axis=0) for g in gamma_ts], axis=0)
    residuals = masked_lagged_chars - xs_imputations

    
    global_fwbw = imputation_model_simplified.impute_chars(
        masked_lagged_chars, oos_xs_imputations, residuals, 
        suff_stat_method='fwbw', 
        constant_beta=True, beta_weight=False
    )
    imputation_utils.save_imputation(global_fwbw, dates, permnos, chars, "global_fwbw" + tag)
    del global_fwbw


    global_fw = imputation_model_simplified.impute_chars(
        masked_lagged_chars, oos_xs_imputations, residuals, 
        suff_stat_method='next_val', 
        constant_beta=True, beta_weight=False
    )
    imputation_utils.save_imputation(global_fw, dates, permnos, chars, "global_fw" + tag)
    del global_fw


    global_bw = imputation_model_simplified.impute_chars(
        masked_lagged_chars, oos_xs_imputations, residuals, 
        suff_stat_method='last_val', 
        constant_beta=True, beta_weight=False,
    )
    imputation_utils.save_imputation(global_bw, dates, permnos, chars, "global_bw" + tag)
    del global_bw


    global_xs = oos_xs_imputations
    imputation_utils.save_imputation(global_xs, dates, permnos, chars, "global_xs" + tag)
    del global_xs

    
    # Local  
    
    base_path = '../data/imputation_cache/'
    
    gamma_ts, lmbda = imputation_model_simplified.impute_panel_xp_lp(
            char_panel=masked_lagged_chars, 
            return_panel=np.nan_to_num(return_panel), min_chars=1, K=20, 
            num_months_train=percentile_rank_chars.shape[0],
            reg=0.01,
            time_varying_lambdas=True,
            window_size=1, 
            n_iter=1,
            eval_data=None,
            allow_mean=False)

    oos_xs_imputations = imputation_model_simplified.get_all_xs_vals(masked_lagged_chars, reg=0.01, 
                                     Lmbda=lmbda, time_varying_lmbda=True)

    xs_imputations = np.concatenate([np.expand_dims(g @ l.T, axis=0) for g, l in zip(gamma_ts, lmbda)], axis=0)
    residuals = masked_lagged_chars - xs_imputations


    
    local_bw = imputation_model_simplified.impute_chars(
        masked_lagged_chars, oos_xs_imputations, residuals, 
        suff_stat_method='last_val', 
        constant_beta=False, beta_weight=False
    )
    imputation_utils.save_imputation(local_bw, dates, permnos, chars, "local_bw" + tag)
    del local_bw

    local_xs = oos_xs_imputations
    imputation_utils.save_imputation(local_xs, dates, permnos, chars, "local_xs" + tag)
    del local_xs


    prev_val = imputation_model_simplified.simple_imputation(gamma_ts, masked_lagged_chars, 
                                             "fwbw", None, char_groupings,
                                                            eval_char_data=None)
    imputation_utils.save_imputation(prev_val, dates, permnos, chars, "prev_val" + tag)
    del prev_val


    local_ts = imputation_model_simplified.impute_chars(
        masked_lagged_chars, None, None, 
        suff_stat_method='last_val', 
        constant_beta=False, beta_weight=False
    )
    
    imputation_utils.save_imputation(local_ts, dates, permnos, chars, "local_ts" + tag)
    del local_ts


    xs_median = imputation_model_simplified.simple_imputation(gamma_ts, masked_lagged_chars, 
                                             "fwbw", None, char_groupings, 
                                                            eval_char_data=None,
                                                                              median_imputation=True)
    imputation_utils.save_imputation(xs_median, dates, permnos, chars, "xs_median" + tag)
    del xs_median



    indus_median = imputation_model_simplified.simple_imputation(gamma_ts, percentile_rank_chars, 
                                             "fwbw", None, char_groupings, median_imputation=False,
                                                            eval_char_data=None,
                                           industry_median=True, industries=industries)
    imputation_utils.save_imputation(indus_median, dates, permnos, chars, "indus_median" + tag)
    del indus_median
    
    
if __name__ == '__main__':
    data = np.load('../data/raw_rank_trunk_chars.npz')
    percentile_rank_chars = data['rank_chars']
    regular_chars = data['raw_chars']
    chars = data['chars']
    dates = data['dates']
    return_panel = data['returns']
    permnos = data['permnos']
    rts = data['rfs']
    monthly_updates = data['monthly_updates']
    
    import pandas as pd
    sic_fic = pd.read_csv("../data/sic_fic.csv")
    contained_permnos = sic_fic.LPERMNO.unique()
    industries = np.zeros_like(permnos)
    for i, p in enumerate(permnos):
        if p in contained_permnos:
            industries[i] = sic_fic.loc[sic_fic.LPERMNO == p].sic.unique()[0] // 1000
    industries = industries // 1000
        
    run_all_imputations(missing_data_type=None, percentile_rank_chars=percentile_rank_chars, 
                        dates=dates, return_panel=return_panel, chars=chars, industries=industries,
                       monthly_updates=monthly_updates)
    run_all_imputations(missing_data_type="MAR", percentile_rank_chars=percentile_rank_chars, 
                        dates=dates, return_panel=return_panel, chars=chars, industries=industries,
                       monthly_updates=monthly_updates)
    run_all_imputations(missing_data_type="logit", percentile_rank_chars=percentile_rank_chars, 
                         dates=dates, return_panel=return_panel, chars=chars, industries=industries,
                       monthly_updates=monthly_updates)
    run_all_imputations(missing_data_type="PROB_BLOCK", percentile_rank_chars=percentile_rank_chars, 
                        dates=dates, return_panel=return_panel, chars=chars, industries=industries,
                       monthly_updates=monthly_updates)
    