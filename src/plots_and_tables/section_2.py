from plots_and_tables import plot_base
import missing_data_utils
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm 
import imputation_utils, logit_models_and_masking
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import pandas as pd
import statsmodels.api as sm


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

            # print("creating the model", X.shape, "training examples", np.sum(Y), "positives")
            logit_model = sm.Logit(Y, X) #Create model instance
            # print("fitting the model")
            result_start = logit_model.fit(method = "newton", maxiter=50, disp=False,
                                        kwargs={"tol":1e-8}) #Fit model, 0.652114
            train_fpr, train_tpr, _ = metrics.roc_curve(Y, result_start.predict(X))
            start_agg_train_aocs.append(metrics.auc(train_fpr, train_tpr))

            # print(metrics.auc(train_fpr, train_tpr))
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
            # print(metrics.auc(test_fpr, test_tpr))
            start_agg_test_aocs.append(metrics.auc(test_fpr, test_tpr))
            
            # plt.plot(test_fpr, test_tpr, label='test')
            # plt.plot(train_fpr, train_tpr, label='train')
            # plt.legend()
            # plt.show()

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

            # print("creating the model", X.shape, "training examples", np.sum(Y), "positives")
            logit_model = sm.Logit(Y, X) #Create model instance
            # print("fitting the model")
            result_middle = logit_model.fit(method = "newton", maxiter=50, disp=False,
                                        kwargs={"tol":1e-8}) #Fit model
            train_fpr, train_tpr, _ = metrics.roc_curve(Y, result_middle.predict(X))
            

            middle_agg_train_aocs.append(metrics.auc(train_fpr, train_tpr))

            # print(metrics.auc(train_fpr, train_tpr), metrics.log_loss(Y, result_middle.predict(X)))

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
            # print(metrics.auc(test_fpr, test_tpr), metrics.log_loss(Y, result_middle.predict(X)))
            # plt.plot(test_fpr, test_tpr, label='test')
            # plt.plot(train_fpr, train_tpr, label='train')
            # plt.legend()
            # plt.show()
            
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

            # print("creating the model")
            logit_model = sm.Logit(Y, X, disp=False) #Create model instance
            # print("fitting the model")
            result_end = logit_model.fit(method = "newton", maxiter=50, disp=False) #Fit model
            train_fpr, train_tpr, _ = metrics.roc_curve(Y, result_end.predict(X))
            end_agg_train_aocs.append(metrics.auc(train_fpr, train_tpr))
            # print(metrics.auc(train_fpr, train_tpr))

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