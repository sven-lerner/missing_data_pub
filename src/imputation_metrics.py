import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import numpy as np

def get_imputation_metrics_time_series(imputed_data, gt_data, mask, tgt_char_inds, max_val=None, 
                           tgt_char=None, mse_mask=None):
    '''
    utility method to get the time series of RMSE by characteristic given imputed and ground truth data
    '''
    if max_val is None:
        max_val = np.nanmax(np.abs(gt_data))
    if mse_mask is None:
        mse_mask = mask
    nan_mask = np.ones_like(mse_mask, dtype=float)
    nan_mask[~mse_mask] = np.nan
    if tgt_char is None:
        mse_by_char = np.nanmean(np.square(imputed_data - gt_data) * nan_mask,
                                               axis=1)
        mmse_over_time = np.nanmean(mse_by_char, axis=1)
        r2_over_time = 1 - (np.nansum(np.nansum(np.square(imputed_data - gt_data) * nan_mask,
                                                axis=1), axis=1) /
                        np.nansum(np.nansum(np.square(gt_data) * nan_mask, axis=1), axis=1))
    else:
        mse_by_char = None
        mmse_over_time = np.nanmean(np.square(imputed_data - gt_data)[:,:,tgt_char]*
                                    nan_mask.squeeze(),
                                    axis=1)
        r2_over_time = 1 - (np.nansum(np.square(imputed_data - gt_data)[:,:,tgt_char] *
                                      nan_mask.squeeze(),
                                      axis=1) /
                        np.nansum(np.square(gt_data)[:,:,tgt_char] * nan_mask.squeeze(), axis=1))
    
    violations_by_char = np.sum(np.logical_and(np.abs(imputed_data) > max_val, mask), axis=1)
    return mmse_over_time, r2_over_time, violations_by_char, mask, mse_by_char

def get_aggregate_imputation_metrics(imputed_data, gt_data, mask, monthly_update_mask,
                                    char_freq_list, net_mask=None, norm_func=None,
                                    clip=True):
    '''
    utility method to get the aggregate RMSE by characteristic given imputed and ground truth data
    '''
    if clip:
        imputed_data = np.clip(imputed_data, -0.5, 0.5) 
    mean_char_errors = []
    if norm_func is None:
        norm_func = np.square
    for c, x in enumerate(char_freq_list):
        char, freq = x
        diffs = imputed_data[:,:,c] - gt_data[:,:,c]
        if net_mask is None:
            eval_mask = np.ones_like(diffs, dtype=bool)
        else:
            eval_mask = np.copy(net_mask[:,:,c])
        if freq != 'M' and monthly_update_mask is not None:
            eval_mask = np.logical_and(eval_mask, monthly_update_mask)
        diffs[~eval_mask] = np.nan
        
        mean_char_errors.append(np.nanmean(norm_func(diffs), axis=1))
    quarter_metrics = [x for x,y in zip(mean_char_errors, char_freq_list) if 
                      y[1] != 'M']
    month_metrics = [x for x,y in zip(mean_char_errors, char_freq_list) if 
                      y[1] == 'M']
    return mean_char_errors, month_metrics, quarter_metrics



def get_flags(raw_char_panel, return_panel):
    '''
    utility method to get wthe state of a characteristic
    - observed
    - missing at the start
    - missing in the middle
    - missing at the end
    - company not observed
    '''
    T, N, C = raw_char_panel.shape
    flag_panel = np.zeros_like(raw_char_panel, dtype=np.int8)
    first_occurances = np.argmax(~np.isnan(raw_char_panel), axis=0)
    not_in_sample = np.all(np.isnan(raw_char_panel), axis=0)
    last_occurances = T - 1 - np.argmax(~np.isnan(raw_char_panel[::-1]), axis=0)
    
    first_return = np.argmax(~np.isnan(return_panel), axis=0)
    last_return = T - 1 - np.argmax(~np.isnan(return_panel[::-1]), axis=0)
    
    totally_missing = np.logical_and(np.all(np.isnan(raw_char_panel), axis=2), np.isnan(return_panel))
    
    flag_panel[~np.isnan(raw_char_panel)] = 1
    
    for t in range(raw_char_panel.shape[0]):
        
        present_return = np.expand_dims(~np.isnan(return_panel[t]), axis=1)
        nan_chars = np.isnan(raw_char_panel[t])
        missing_at_start = np.logical_and(present_return, t < first_occurances)
        missing_at_start = np.logical_and(missing_at_start, nan_chars)
        flag_panel[t, missing_at_start] = -1
        
        missing_in_middle = np.logical_and(present_return, t >= first_occurances)
        missing_in_middle = np.logical_and(missing_in_middle, t <= last_occurances)
        missing_in_middle = np.logical_and(missing_in_middle, nan_chars)
        flag_panel[t, missing_in_middle] = -2
        
        missing_at_end = np.logical_and(present_return, t > last_occurances)
        missing_at_end = np.logical_and(missing_at_end, nan_chars)
        flag_panel[t, missing_at_end] = -3
        
        flag_panel[t, np.logical_and(not_in_sample, present_return)] = -4
        
        in_window = np.logical_and(t >= first_return, t <= last_return)
        flag_panel[t, np.logical_and(in_window, totally_missing[t])] = -5
        
        raw_char_copy = np.copy(raw_char_panel[t])
        raw_char_copy[np.isinf(raw_char_copy)] = np.nan
    
    return flag_panel