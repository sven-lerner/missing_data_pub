import numpy as np
from sklearn import metrics

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

def get_pooled_x_y_from_panel(char_panel, 
                              mask, 
                              chars,
                              fit_chars, 
                              exclude_chars,
                              factors, 
                              include_chars=True,
                              include_factors=False,
                              include_FE=False,
                              include_last_val=False,
                              switch=False,
                              include_missing_gap=False,
                              missing_gaps=None,
                              get_y=True):

    '''
    utility method to get a flattened vector of features from the 3d tensor of the data-set
    '''
    Y = []
    X = []
    regr_chars = np.logical_and(~exclude_chars, ~fit_chars)
    regr_chars_only = chars[regr_chars]
    regr_chars_map = {x: i for i, x in enumerate(regr_chars_only)}
    num_regr_chars = np.sum(regr_chars)
    feature_names = []
    idxs = []
    
    if include_chars:
        feature_names.append(chars[fit_chars])
    if include_factors:
        feature_names.append([f"f-{i}" for i in range(factors.shape[-1])])
    if include_FE:
        feature_names.append([f"FE-{x}" for x in chars[regr_chars]])
    if include_missing_gap:
        feature_names.append("missing gap")
    if include_last_val:
        feature_names.append("last val missing flag")
    
    for t in range(mask.shape[0]):
        for c in range(mask.shape[2]):
            if regr_chars[c]:
                t_c_filter = mask[t, :, c]
                if np.sum(t_c_filter) > 0:
                    if get_y:
                        if switch:
                            assert t > 0
                            Y.append(np.isnan(char_panel[t-1, t_c_filter, c]) != np.isnan(char_panel[t, t_c_filter, c]))
                        else:
                            Y.append(np.isnan(char_panel[t, t_c_filter, c]))
                    features = []
                    if include_chars:
                        features.append(char_panel[t, t_c_filter][:, fit_chars])
                    if include_factors:
                        features.append(factors[t, t_c_filter])
                    if include_FE:
                        fe = np.zeros((np.sum(t_c_filter), num_regr_chars))
                        fe[:,regr_chars_map[chars[c]]] = 1
                        features.append(fe)
                    if include_missing_gap:
                        features.append(missing_gaps[t, t_c_filter, c:c+1])
                    if include_last_val:
                        assert t > 0
                        features.append(np.isnan(char_panel[t-1, t_c_filter, c:c+1]))
                    features = np.concatenate(features, axis=1)
                    X.append(features)
                    idxs.append([(t, x[0], c) for x in np.argwhere(t_c_filter)])
    X = np.concatenate(X, axis=0)
    idxs = np.concatenate(idxs, axis=0)
    if get_y:
        Y = np.concatenate(Y, axis=0)
    else:
        Y = None
    return X, Y, idxs, feature_names


def get_input_for_start_logit_model(percentile_rank_chars, chars, tgt_char_mask, start_idx, end_idx,
                                    regr_chars, exl_char_mask):
    '''
    utility method to get data features for the logit model predicting whether data is observed or not at the start of the sample
    '''
    filter_too_long_gaps = True
    char_present_filter = np.all(~np.isnan(percentile_rank_chars[:,:,tgt_char_mask]), axis=2)
    input_filter = np.zeros_like(percentile_rank_chars, dtype=bool)
    input_filter[:,:,regr_chars] = 1
    input_filter[~char_present_filter,:] = 0
    input_filter[:start_idx,:] = 0
    input_filter[0,:,:] = 0


    not_start = np.any(~np.isnan(percentile_rank_chars[:start_idx]), axis=0)
    for t in range(start_idx, percentile_rank_chars.shape[0]):
        input_filter[t, not_start] = 0
        not_start = np.logical_or(not_start, ~np.isnan(percentile_rank_chars[t]))

    missing_gap = np.zeros_like(input_filter, dtype=int)
#     missing_gap[:, :, :] = 0
    first_occ = np.argmax(np.any(~np.isnan(percentile_rank_chars), axis=2), axis=0)
    for t in range(start_idx, percentile_rank_chars.shape[0]):
        for c in range(input_filter.shape[2]):
            missing_gap[t, input_filter[t, :, c], c] = t - first_occ[input_filter[t, :, c]]
            if filter_too_long_gaps:
                input_filter[:t-10, input_filter[t, :, c], c] = 0
    
    train_input_filter = input_filter[:end_idx]
    
    
    
    X, Y, idxs, feature_names = get_pooled_x_y_from_panel(percentile_rank_chars, 
                                  train_input_filter, 
                                  chars,
                                  tgt_char_mask, 
                                  exl_char_mask,
                                  factors=None, 
                                  include_chars=True,
                                  include_factors=False,
                                  include_FE=True,
                                  include_last_val=False,
                                  switch=True,
                                  include_missing_gap=True,
                                  missing_gaps=missing_gap)
    return X, Y ,idxs, feature_names

def get_input_for_middle_logit_model(percentile_rank_chars, chars, tgt_char_mask, start_idx, end_idx, 
                                     regr_chars, exl_char_mask, monthly_updates):
    '''
    utility method to get data features for the logit model predicting whether data is observed or not in the middle of the sample
    '''
    char_present_filter = np.all(~np.isnan(percentile_rank_chars[:,:,tgt_char_mask]), axis=2)
    input_filter = np.zeros_like(percentile_rank_chars, dtype=bool)
    input_filter[:,:,regr_chars] = 1
    input_filter[~char_present_filter,:] = 0
    input_filter[:start_idx,:] = 0
    input_filter[0,:,:] = 0

    not_start = np.any(~np.isnan(percentile_rank_chars[:start_idx]), axis=0)
    for t in range(start_idx, percentile_rank_chars.shape[0]):
        input_filter[t, ~not_start] = 0
        not_start = np.logical_or(not_start, ~np.isnan(percentile_rank_chars[t]))

    curr_missing_gap = np.zeros(input_filter.shape[1:], dtype=int)
    missing_gap = np.zeros_like(input_filter, dtype=int)

    for t in range(0, end_idx):
        if t > start_idx:            
            for c in range(input_filter.shape[2]):
                missing_gap[t, input_filter[t, :, c], c] = curr_missing_gap[input_filter[t, :, c], c]

        curr_missing_gap += 1
        curr_missing_gap[~np.isnan(percentile_rank_chars[t])] = 0

    for i, c in enumerate(chars):
        if char_map[c] != "M": 
            input_filter[:,:,i] = np.logical_and(input_filter[:,:,i], monthly_updates)
        
    X, Y, idxs, feature_names = get_pooled_x_y_from_panel(percentile_rank_chars[:end_idx], 
                                  input_filter[:end_idx],
                                  chars,
                                  tgt_char_mask, 
                                  exl_char_mask,
                                  factors=None, 
                                  include_chars=True,
                                  include_factors=False,
                                  include_FE=True,
                                  include_last_val=True,
                                  switch=True,
                                  include_missing_gap=True,
                                  missing_gaps=missing_gap[:end_idx])
    return X, Y ,idxs, feature_names

def get_input_for_end_logit_model(percentile_rank_chars, chars, tgt_char_mask, start_idx, end_idx,
                                  regr_chars, exl_char_mask):
    '''
    utility method to get data features for the logit model predicting whether data is observed or not at the end of the sample
    '''
    input_filter = np.zeros_like(percentile_rank_chars, dtype=bool)
    char_present_filter = np.all(~np.isnan(percentile_rank_chars[:,:,tgt_char_mask]), axis=2)
    input_filter[:,:,regr_chars] = 1
    input_filter[~char_present_filter,:] = 0
    input_filter[:start_idx,:] = 0
    input_filter[0,:,:] = 0

    for t in range(start_idx, percentile_rank_chars.shape[0]):
        last_gap = np.sum(np.isnan(percentile_rank_chars[t:-1]) != np.isnan(percentile_rank_chars[t+1:]), axis=0) <= 1
        input_filter[t, ~last_gap] = 0
    
    X, Y, idxs, feature_names = get_pooled_x_y_from_panel(percentile_rank_chars[:end_idx], 
                                  input_filter[:end_idx], 
                                  chars,
                                  tgt_char_mask, 
                                  exl_char_mask,
                                  factors=None, 
                                  include_chars=True,
                                  include_factors=False,
                                  include_FE=True,
                                  include_last_val=False,
                                  switch=False,
                                  include_missing_gap=False,
                                  missing_gaps=None)
    return X, Y ,idxs, feature_names

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model as lm



def create_logit_model(prediction_type, percentile_rank_chars, chars, monthly_updates, model_type='logit'):
    '''
    fit a logit model to predict whether data is observed or not, depending on when in the lifecyle of a company we want to fit
    [START MIDDLE or END]
    '''
    
    tgt_chars = ['ME', 'R2_1', 'D2P', 'IdioVol', 'TURN', 'SPREAD', 'VAR']
    exl_chars = [ 'RVAR']
    regr_chars = np.logical_and(~np.isin(chars, tgt_chars),
                               ~np.isin(chars, exl_chars))
    tgt_char_mask = np.isin(chars, tgt_chars)
    exl_char_mask = np.isin(chars, exl_chars)
    
    fit_end_idx = percentile_rank_chars.shape[0] // 2
    fit_start_idx = fit_end_idx // 2
    
    if prediction_type == 'START':
        X, Y ,idxs, feature_names = get_input_for_start_logit_model(percentile_rank_chars, chars,
                                                                  tgt_char_mask, fit_start_idx, fit_end_idx,
                                                                    regr_chars, exl_char_mask)
        X_oos, Y_oos ,_, _ = get_input_for_start_logit_model(percentile_rank_chars, chars,
                                                                  tgt_char_mask, fit_end_idx, percentile_rank_chars.shape[0],
                                                                    regr_chars, exl_char_mask)
    elif prediction_type == 'MIDDLE':
        X, Y ,idxs, feature_names = get_input_for_middle_logit_model(percentile_rank_chars, chars,
                                                                  tgt_char_mask, fit_start_idx, fit_end_idx,
                                                                    regr_chars, exl_char_mask,
                                                                     monthly_updates)
        X_oos, Y_oos ,_, _ = get_input_for_middle_logit_model(percentile_rank_chars, chars,
                                                                  tgt_char_mask, fit_end_idx, percentile_rank_chars.shape[0],
                                                                    regr_chars, exl_char_mask,
                                                                     monthly_updates)
    elif prediction_type == 'END':
        X, Y ,idxs, feature_names = get_input_for_end_logit_model(percentile_rank_chars, chars,
                                                                  tgt_char_mask, fit_start_idx, fit_end_idx,
                                                                 regr_chars, exl_char_mask)
        X_oos, Y_oos ,_, _ = get_input_for_end_logit_model(percentile_rank_chars, chars,
                                                                  tgt_char_mask, fit_end_idx, percentile_rank_chars.shape[0],
                                                                 regr_chars, exl_char_mask)
    else:
        raise ValueError("prediction_type must be one of ['START', 'MIDDLE', 'END']")
    
    if model_type == 'logit':
        result = lm.LogisticRegression(random_state=0, penalty='none', max_iter=1000).fit(X, Y)
    elif model_type == 'forest':
        result = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=0)
        result.fit(X, Y)
        
    train_fpr, train_tpr, _ = metrics.roc_curve(1.0*Y, result.predict(X), drop_intermediate=False)
    test_fpr, test_tpr, _ = metrics.roc_curve(1.0*Y_oos, result.predict(X_oos), drop_intermediate=False)
    
    return result


def generate_logit_masked_data(start_model, middle_model, end_model, percentile_rank_chars, chars,
                              monthly_mask_tgt_perc, monthly_updates, multiplier=1):
    '''
    utility method to generate logit masked data as describerd in the paper
    First we fit the logit models to predict the missing pattern, then we use the logit models to mask the data-set
    '''
    tgt_chars = ['ME', 'R2_1', 'D2P', 'IdioVol', 'TURN', 'SPREAD', 'VAR']
    exl_chars = [ 'RVAR']
    regr_chars = np.logical_and(~np.isin(chars, tgt_chars),
                               ~np.isin(chars, exl_chars))
    tgt_char_mask = np.isin(chars, tgt_chars)
    exl_char_mask = np.isin(chars, exl_chars)
    start_idx = 1 
    masked_len = np.zeros(percentile_rank_chars.shape[1:])
    
    conditioning_chars_present = np.expand_dims(np.all(~np.isnan(percentile_rank_chars[start_idx:,:,tgt_char_mask]), 
                                                   axis=2), axis=2)
    masked_entries = np.isnan(percentile_rank_chars)
    

    before_any_obs = np.all(np.isnan(percentile_rank_chars[:start_idx]), axis=0)
    after_prev_obs = np.any(~np.isnan(percentile_rank_chars[:start_idx]), axis=0)

    missing_gaps = np.full(percentile_rank_chars.shape[1:], np.nan)
    missing_gaps[np.logical_and(np.any(~np.isnan(percentile_rank_chars[0]), axis=1, keepdims=True), 
                               np.isnan(percentile_rank_chars[0]))] = 1
    missing_gaps[np.logical_and(np.any(~np.isnan(percentile_rank_chars[0]), axis=1, keepdims=True), 
                               ~np.isnan(percentile_rank_chars[0]))] = 0
    
    
    
    for t in range(start_idx, percentile_rank_chars.shape[0]):
        found_enough = False
        
        total_t = np.sum(~np.isnan(percentile_rank_chars[t, :, regr_chars]))
        while not found_enough:
            to_sample_m1 = np.expand_dims(np.logical_and(conditioning_chars_present[t - start_idx],
                                             before_any_obs), axis=0)
            to_sample_m1 = np.concatenate([np.zeros_like(to_sample_m1, dtype=bool), to_sample_m1], axis=0)
            to_sample_m1[:,:,~regr_chars] = 0
            to_sample_m2 = np.expand_dims(np.logical_and(conditioning_chars_present[t - start_idx],
                                             after_prev_obs), axis=0)
            to_sample_m2 = np.concatenate([np.zeros_like(to_sample_m2, dtype=bool), to_sample_m2], axis=0)
            to_sample_m2[:,:,~regr_chars] = 0

            char_input = np.copy(percentile_rank_chars[t-1:t+1])
            
            
            char_input[0, masked_entries[t-1]] = np.nan

            
            missing_gaps[np.logical_and(np.isnan(missing_gaps), 
                                        np.any(~np.isnan(percentile_rank_chars[t]), axis=1, keepdims=True))] = 0
            
            input_missing_gaps = np.copy(np.tile(missing_gaps, [2,1,1]))
            input_missing_gaps[0] = np.nan
    
            X_start, Y_start, idxs_start, feature_names = get_pooled_x_y_from_panel(char_input, 
                                      to_sample_m1, 
                                      chars,
                                      tgt_char_mask, 
                                      exl_char_mask,
                                      factors=None, 
                                      include_chars=True,
                                      include_factors=False,
                                      include_FE=True,
                                      include_last_val=False,
                                      switch=True,
                                      include_missing_gap=True,
                                      missing_gaps=input_missing_gaps)
            y_probs = start_model.predict(X_start)
            assert np.all(~np.isnan(y_probs))
            y_start_mask_samples = np.random.binomial(1, p=y_probs) != 1
            for i, idx in enumerate(idxs_start):
                newprob = 1 - (1 - y_probs[i])**multiplier
                if y_start_mask_samples[i]:
                    _, j, k = idx
                    if (missing_gaps[j, k] < 36) and (np.sum(masked_entries[t, j,:]) < 35):
                        masked_entries[t, j, k] = 1

            del X_start, Y_start, idxs_start, feature_names


            X_middle, Y_middle, idxs_middle, feature_names = get_pooled_x_y_from_panel(char_input, 
                                      to_sample_m2, 
                                      chars,
                                      tgt_char_mask, 
                                      exl_char_mask,
                                      factors=None, 
                                      include_chars=True,
                                      include_factors=False,
                                      include_FE=True,
                                      include_last_val=True,
                                      switch=True,
                                      include_missing_gap=True,
                                      missing_gaps=input_missing_gaps)

            y_probs = middle_model.predict_proba(X_middle)[:,1]
            y_middle_mask_samples = np.random.binomial(1, p=y_probs) == 1


            for i, idx in enumerate(idxs_middle):
                _, j, k = idx
                if ~masked_entries[t, j, k] and ~masked_entries[t - 1, j, k]:
                    y_middle_mask_samples[i] = y_middle_mask_samples[i]
                if y_middle_mask_samples[i] and ~masked_entries[t, j, k]:
                    masked_entries[t, j, k] = ~masked_entries[t-1, j, k]
                    masked_len[j, k] += 1
                    if char_map[chars[k]] != 'Q':
                        masked_len[j, k] += 2
                        masked_entries[t+1:t+3, j, k] = True
                else:
                    masked_len[j, k] = 0
            
            masked_t = np.logical_and(masked_entries[t], ~np.isnan(percentile_rank_chars[t]))
            percent_masked = np.sum(masked_t) / total_t

            if percent_masked >= monthly_mask_tgt_perc or t == start_idx:
                found_enough = True


            del X_middle, Y_middle, idxs_middle, feature_names

        before_any_obs = np.logical_and(before_any_obs, masked_entries[t])

        after_prev_obs = np.logical_or(after_prev_obs, ~masked_entries[t])
        missing_gaps += 1
        missing_gaps[~masked_entries[t]] = 0 
    
    masked_entries = np.logical_and(masked_entries, ~np.isnan(percentile_rank_chars))
    update_chars = np.copy(percentile_rank_chars)
    for i, c in enumerate(chars):
        if char_map[c] !='M':
            update_chars[~(monthly_updates == 1),i] = np.nan
    
    
    masked_lagged_chars = np.copy(percentile_rank_chars)
    only_missing_chars = np.copy(percentile_rank_chars)
    only_missing_chars[:,:,:] = np.nan
    for c in range(45):
        if char_map[chars[c]] != 'M':
            for t in range(start_idx,masked_entries.shape[0]):
                missing = masked_entries[t,:,c]
                only_missing_chars[t:t+3,missing,c] = np.copy(masked_lagged_chars[t:t+3,missing,c])
                masked_lagged_chars[t:t+3,missing,c] = np.nan
        else:
            for t in range(start_idx,masked_entries.shape[0]):
                missing = masked_entries[t,:,c]
                only_missing_chars[t,missing,c] = np.copy(masked_lagged_chars[t,missing,c])
                masked_lagged_chars[t,missing,c] = np.nan

    only_mimissing_chars = np.copy(update_chars)
    only_mimissing_chars[:start_idx] = np.nan
    for t in range(start_idx, masked_entries.shape[0]):
        only_mimissing_chars[t,~masked_entries[t]] = np.nan

    return masked_lagged_chars, only_missing_chars