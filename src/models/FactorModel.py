from models.model import ModelBase
import numpy as np
from models import ipca_util
import statsmodels.api as sm

class FactorRegressionModel(ModelBase):

    def __init__(self):
        super().__init__(self, False, 100)
        self.train_factors = None
        self.val_factors = None
        self.test_factors = None
        self.data = None
        self.shift = 0
        self.reg = 0
        self.threshold = False
        self.get_pvalues = False
        self.pvalues = []
        self.include_intercept = True
        self.conds = []

    def fit(self, train_return, train_individualFeature, train_mask, val_return, val_individualFeature, val_mask,
                test_return, test_individualFeature, test_mask, rfts, missing_bounds=None, recalc_data=False, reg=0):
        self.reg = reg
        if recalc_data:
            if self.include_intercept:
                shape = list(train_individualFeature.shape)
                shape[-1] = 1
                train_individualFeature = np.concatenate([np.ones(shape), train_individualFeature], axis=2)
                
                shape = list(val_individualFeature.shape)
                shape[-1] = 1
                val_individualFeature = np.concatenate([np.ones(shape), val_individualFeature], axis=2)
                
                shape = list(test_individualFeature.shape)
                shape[-1] = 1
                test_individualFeature = np.concatenate([np.ones(shape), test_individualFeature], axis=2)
            self.data = [train_return, train_individualFeature, train_mask, val_return, val_individualFeature,
                         val_mask,  test_return, test_individualFeature, test_mask, rfts]
        else:
            train_return, train_individualFeature, train_mask, \
                val_return, val_individualFeature, val_mask, \
                test_return, test_individualFeature, test_mask, rfts = self.data
        if self.threshold:
            train_individualFeature[train_mask] = np.clip(train_individualFeature[train_mask], -0.5, 0.5)
            val_individualFeature[val_mask] = np.clip(val_individualFeature[val_mask], -0.5, 0.5)
            test_individualFeature[test_mask] = np.clip(test_individualFeature[test_mask], -0.5, 0.5)
        if missing_bounds is not None:
            min_missing, max_missing = missing_bounds
            train_missing_counts, val_missing_counts, test_missing_counts = self.get_train_val_test_missing_counts(train_path,
                                                                                                               val_path, 
                                                                                                               test_path)
            train_mask = np.logical_and(train_mask, np.logical_and(train_missing_counts >= min_missing,
                                                                  train_missing_counts <= max_missing))
            val_mask = np.logical_and(val_mask, np.logical_and(val_missing_counts >= min_missing,
                                                                  val_missing_counts <= max_missing))
            test_mask = np.logical_and(test_mask, np.logical_and(test_missing_counts >= min_missing,
                                                                  test_missing_counts <= max_missing))

            
        self.rfts = rfts
        self.train_factors = self.regress_factors(train_return, train_individualFeature, train_mask)
        self.val_factors = self.regress_factors(val_return, val_individualFeature, val_mask)
        self.test_factors =  self.regress_factors(test_return, test_individualFeature, test_mask)
    
    def get_train_val_test_missing_counts(self, train_path, val_path, test_path):
        tmp = np.load(train_path)
        train_missing_counts = tmp['missing_counts']
        
        tmp = np.load(val_path)
        val_missing_counts = tmp['missing_counts']
        
        tmp = np.load(test_path)
        test_missing_counts = tmp['missing_counts']

        return train_missing_counts, val_missing_counts, test_missing_counts
    
    def predict(self, features, loadings):
        return np.stack([features[t].dot(loadings[t]) for t in range(loadings.shape[0])])
    
    def eval_metrics_over_decile_portfolios(self, train_path, val_path, test_path, test_partitions, chars_to_form, size_data,
                                           num_portfolios =10, mean_return=False, predicted_mean_return=False):
        if train_path is None:
            train_returns, train_individualFeatures, train_masks, \
            val_returns, val_individualFeatures, val_masks, \
            test_returns, test_individualFeatures, test_masks, rfts = self.data 
        else:
            train_returns, train_individualFeatures, train_masks, \
                val_returns, val_individualFeatures, val_masks, \
                test_returns, test_individualFeatures, test_masks, rfts = self.get_precomputed_data(train_path,
                                                                                                    val_path,
                                                                                                    test_path)
        
        train_missing_counts, val_missing_counts, test_missing_counts = self.get_train_val_test_missing_counts(train_path,
                                                                                                               val_path, 
                                                                                                               test_path)
        if test_partitions is None:
            test_partitions = [(0, 45)]
        ret_vals = {}
        
        masks = np.concatenate([train_masks, val_masks, test_masks])
        returns = np.concatenate([train_returns, val_returns, test_returns])
        features = np.concatenate([train_individualFeatures, val_individualFeatures, test_individualFeatures])
        
        preds = self.predict(features, np.concatenate([self.train_factors, self.val_factors, self.test_factors], axis=0))
        
        if mean_return:
            preds = np.array([np.mean(returns[t,masks[t]]) for t in range(600)])
        elif predicted_mean_return:
            preds = np.array([np.mean(preds[t,masks[t]]) for t in range(600)])
            
        
        if self.bin_data:
            self.bin_data = False
            _, un_binned_train_individualFeatures, _, \
                _, un_binned_val_individualFeatures, _, \
                _, un_binned_test_individualFeatures, _, _ = self.get_precomputed_data(train_path, val_path, test_path)
            unbinned_features = np.concatenate([un_binned_train_individualFeatures, un_binned_val_individualFeatures,
                                               un_binned_test_individualFeatures])
            self.bin_data = True
        else:
            unbinned_features = features
            
        
#         for p in test_partitions:
#             partition_train_mask = np.logical_and(np.logical_and(train_missing_counts >= p[0], 
#                                                                train_missing_counts <= p[1]),
#                                                  train_masks)
#             partition_val_mask = np.logical_and(np.logical_and(val_missing_counts >= p[0], 
#                                                                val_missing_counts <= p[1]),
#                                                  val_masks)
#             partition_test_mask = np.logical_and(np.logical_and(test_missing_counts >= p[0], 
#                                                                test_missing_counts <= p[1]),
#                                                  test_masks)
        
        for char, char_ind in chars_to_form:
            portfolios = [[] for _ in range(num_portfolios)]
            for t in range(preds.shape[0]):
                present_stocks = np.argwhere(masks[t]).squeeze()
                n_stocks = present_stocks.shape[0]
                portfolio_size = int(np.ceil(n_stocks / num_portfolios))
                sorted_present_stocks = present_stocks[np.argsort(unbinned_features[t,masks[t],char_ind])]
                for i in range(num_portfolios):
                    in_portfolio = sorted_present_stocks[i*portfolio_size:(1+i)*portfolio_size]
                    assert np.all(masks[t,in_portfolio])
                    if size_data is None:
                        if mean_return or predicted_mean_return:
                            pred_return = preds[t]
                        else:
                            pred_return = np.mean(preds[t,in_portfolio])
                        gt_return = np.mean(returns[t,in_portfolio])
                    else:
                        if mean_return or predicted_mean_return:
                            pred_return = preds[t]
                        else:
                            pred_return = np.average(preds[t,in_portfolio], weights=size_data[t,in_portfolio])
                        gt_return = np.average(returns[t,in_portfolio], weights=size_data[t,in_portfolio])
                    portfolios[i].append((pred_return, gt_return))
            ret_vals[char] = portfolios
            
        return ret_vals
        

    def eval_metrics(self, train_path, val_path, test_path, test_partitions=None,
                    get_error_spread=False, prediction=False, recalc_data=False):
        if not recalc_data:
            train_returns, train_individualFeatures, train_masks, \
            val_returns, val_individualFeatures, val_masks, \
            test_returns, test_individualFeatures, test_masks, rfts = self.data 
        else:
            train_returns, train_individualFeatures, train_masks, \
                val_returns, val_individualFeatures, val_masks, \
                test_returns, test_individualFeatures, test_masks, rfts = self.get_precomputed_data(train_path,
                                                                                                    val_path,
                                                                                                    test_path)

        if self.threshold:
            train_individualFeatures[train_masks] = np.clip(train_individualFeatures[train_masks], -0.5, 0.5)
            val_individualFeatures[val_masks] = np.clip(val_individualFeatures[val_masks], -0.5, 0.5)
            test_individualFeatures[test_masks] = np.clip(test_individualFeatures[test_masks], -0.5, 0.5)
        train_missing_counts = np.sum(np.isnan(train_individualFeatures), axis=2)
        val_missing_counts = np.sum(np.isnan(val_individualFeatures), axis=2)
        test_missing_counts = np.sum(np.isnan(test_individualFeatures), axis=2)
        
        if test_partitions is None:
            test_partitions = [(0, 45)]
        ret_vals = []
        for p in test_partitions:
            partition_train_mask = np.logical_and(np.logical_and(train_missing_counts >= p[0], 
                                                               train_missing_counts <= p[1]),
                                                 train_masks)
            partition_val_mask = np.logical_and(np.logical_and(val_missing_counts >= p[0], 
                                                               val_missing_counts <= p[1]),
                                                 val_masks)
            partition_test_mask = np.logical_and(np.logical_and(test_missing_counts >= p[0], 
                                                               test_missing_counts <= p[1]),
                                                 test_masks)
            
            print(np.sum(partition_test_mask), np.sum(test_masks))
            r2 = self.get_r2(train_returns, train_individualFeatures, partition_train_mask, \
                             val_returns, val_individualFeatures, partition_val_mask, \
                             test_returns, test_individualFeatures, partition_test_mask, 
                             self.train_factors, self.val_factors, self.test_factors,
                             prediction=prediction)
            xs_r2 = self.get_xs_r2(train_returns, train_individualFeatures, partition_train_mask, \
                                   val_returns, val_individualFeatures, partition_val_mask, \
                                   test_returns, test_individualFeatures, partition_test_mask,
                                   self.train_factors, self.val_factors, self.test_factors,
                                   prediction=prediction)
            srs = self.get_sr(self.train_factors, self.val_factors, self.test_factors)
            if get_error_spread:
                spreads = self.get_prediction_spread_over_time(
                    np.concatenate([self.train_factors, self.val_factors, self.test_factors], axis=0), 
                    np.concatenate([train_individualFeatures, val_individualFeatures, test_individualFeatures], axis=0),
                    np.concatenate([train_returns, val_returns, test_returns], axis=0),
                    np.concatenate([partition_train_mask, partition_val_mask, 
                                    partition_test_mask], axis=0))
                ret_vals.append((r2, xs_r2, srs, spreads))
            else:
                
                ret_vals.append((r2, xs_r2, srs))
        return ret_vals

    def get_r2(self, train_returns, train_individualFeatures, train_masks, \
            val_returns, val_individualFeatures, val_masks, \
            test_returns, test_individualFeatures, test_masks,
              train_factors, val_factors, test_factors, prediction=False):
        factors = np.concatenate([x for x in [train_factors, val_factors, test_factors] if x is not None], axis=0)
        features = np.concatenate([x for x in [train_individualFeatures,
                                               val_individualFeatures, test_individualFeatures] if x is not None], axis=0)
        returns = np.concatenate([x for x in [train_returns, val_returns, test_returns] if x is not None], axis=0)
        masks = np.concatenate([x for x in [train_masks, val_masks, test_masks] if x is not None], axis=0)
        if prediction:
            factors = factors[:-1]
            features = features[1:]
            returns = returns[1:]
            masks = masks[1:]
            
        return self.r2(factors, features, returns, masks)

    def get_xs_r2(self, train_returns, train_individualFeatures, train_masks, \
            val_returns, val_individualFeatures, val_masks, \
            test_returns, test_individualFeatures, test_masks,
              train_factors, val_factors, test_factors, prediction=False):
        factors = np.concatenate([x for x in [train_factors, val_factors, test_factors] if x is not None], axis=0)
        features = np.concatenate([x for x in [train_individualFeatures, val_individualFeatures, 
                                               test_individualFeatures] if x is not None], axis=0)
        returns = np.concatenate([x for x in [train_returns, val_returns, test_returns] if x is not None], axis=0)
        masks = np.concatenate([x for x in [train_masks, val_masks, test_masks] if x is not None], axis=0)
        if prediction:
            factors = factors[:-1]
            features = features[1:]
            returns = returns[1:]
            masks = masks[1:]
            
        return self.xs_r2(factors, features, returns, masks)

    def get_sr(self, train_factors, val_factors, test_factors):
        train_factors = np.concatenate([train_factors, val_factors])
        train_rfts = self.rfts[:train_factors.shape[0]]
        test_rfts = self.rfts[train_factors.shape[0]:]
        portfolio_weights = ipca_util.calculate_efficient_portofolio(train_factors.T, train_rfts.T)
        is_sharpe = self.sharpe(train_factors.dot(portfolio_weights), train_rfts)
        oos_sharpe = self.sharpe(test_factors.dot(portfolio_weights), test_rfts)
        return is_sharpe, oos_sharpe

    def r2(self, factors, chars, returns, mask):
#         total_var = 0
#         total_resid_sq = 0
#         for t in range(factors.shape[0]):
#             res = chars[t, mask[t], :].dot(factors[t]) - returns[t, mask[t]]
#             total_resid_sq += res.T.dot(res)
#             total_var += returns[t, mask[t]].T.dot(returns[t, mask[t]])
#         return 1 - total_resid_sq / total_var
        tot_return_sq = 0
        tot_resid_sq = 0
        for t in range(factors.shape[0]):
            resid_t = chars[t, mask[t], :].dot(factors[t]) - returns[t, mask[t]]
            tot_resid_sq += np.mean(np.square(resid_t))
            tot_return_sq += np.mean(np.square(returns[t, mask[t]]))
        return 1 - tot_resid_sq / tot_return_sq
    
    
    def get_prediction_spread_over_time(self, factors, chars, returns, mask):
        mean_sq_error = []
        sq_error_var = []
        for t in range(factors.shape[0]):
            sq_resid_t = np.square(chars[t, mask[t], :].dot(factors[t]) - returns[t, mask[t]])
            mean_sq_error.append(np.mean(sq_resid_t))
            sq_error_var.append(np.var(sq_resid_t))
        return (mean_sq_error, sq_error_var)
        
    
    def xs_r2(self, factors, chars, returns, mask):
        T = chars.shape[0]
        sum_squared_expected_returns = 0
        sum_squared_expected_residuals = 0
        for i in range(returns.shape[1]):
            reses = []
            p_inds = mask[:,i]
            ti = np.sum(p_inds)
            if ti > 0:
                preds = np.sum(chars[p_inds, i] * factors[p_inds], axis=1)
                reses = preds - returns[p_inds, i]
                sum_squared_expected_returns += (ti / T) * np.mean(returns[p_inds, i]) ** 2
                sum_squared_expected_residuals += (ti / T) * np.mean(reses) ** 2
#                 sum_squared_expected_returns += (ti / T) * np.mean(np.square(returns[p_inds, i]))
#                 sum_squared_expected_residuals += (ti / T) * np.mean(np.square(reses))
        return 1 - sum_squared_expected_residuals / sum_squared_expected_returns

    def sharpe(self, portfolio, rf):
        return np.mean(portfolio - rf)/ np.std(portfolio - rf)


    def regress_factors(self, return_panel, char_panel, mask):

        if self.get_pvalues:
            factors = np.zeros((char_panel.shape[0], char_panel.shape[2]))
            for t in range(return_panel.shape[0]):
                ct = char_panel[t, mask[t], :]
#                 print(ct.shape)
                rt = return_panel[t, mask[t]]
                sm.add_constant(ct) # adding a constant

                model = sm.OLS(endog=rt, exog=ct).fit()
                self.pvalues.append(model.pvalues)
#                 print(model.summary())
                factors[t] =  model.params
        else:
            factors = np.zeros((char_panel.shape[0], char_panel.shape[2]))
            for t in range(return_panel.shape[0]):
                if np.sum(mask[t]) > 0:
                    ct = char_panel[t, mask[t], :]
#                     print(ct.shape)
                    rt = return_panel[t, mask[t]]
                    factors[t] = np.linalg.lstsq(ct.T.dot(ct) + self.reg*np.eye(ct.shape[1]), ct.T.dot(rt), rcond=0)[0]
                    self.conds.append(np.linalg.cond(ct))
#                 factors[t] = np.linalg.lstsq(ct, rt)[0]
        return factors
    
    def get_factors(self):
        return np.concatenate([self.train_factors, self.val_factors, self.test_factors], axis=0)

    
class MeanModel(FactorRegressionModel):
    
    
    def fit(self, train_path, val_path, test_path, missing_bounds=None, recalc_data=False):
        train_returns, train_individualFeatures, train_masks, \
                val_returns, val_individualFeatures, val_masks, \
                test_returns, test_individualFeatures, test_masks, rfts = self.get_precomputed_data(train_path,
                                                                                                    val_path,
                                                                                                    test_path)
        

        self.predictions = np.array([np.mean(train_returns[t, train_masks[t]]) for t in range(train_masks.shape[0])] + 
                                          [np.mean(val_returns[t, val_masks[t]]) for t in range(val_masks.shape[0])]+
                                          [np.mean(test_returns[t, test_masks[t]]) for t in range(test_masks.shape[0])]
                                           )
        
    def eval_metrics(self, train_path, val_path, test_path, test_partitions=None,
                    get_error_spread=False, prediction=False, recalc_data=False):
        if not recalc_data:
            train_returns, train_individualFeatures, train_masks, \
            val_returns, val_individualFeatures, val_masks, \
            test_returns, test_individualFeatures, test_masks, rfts = self.data 
        else:
            train_returns, train_individualFeatures, train_masks, \
                val_returns, val_individualFeatures, val_masks, \
                test_returns, test_individualFeatures, test_masks, rfts = self.get_precomputed_data(train_path,
                                                                                                    val_path,
                                                                                                    test_path)

        
        train_missing_counts, val_missing_counts, test_missing_counts = self.get_train_val_test_missing_counts(train_path,
                                                                                                               val_path, 
                                                                                                               test_path)
        if test_partitions is None:
            test_partitions = [(0, 45)]
        ret_vals = []
        for p in test_partitions:
            partition_train_mask = np.logical_and(np.logical_and(train_missing_counts >= p[0], 
                                                               train_missing_counts <= p[1]),
                                                 train_masks)
            partition_val_mask = np.logical_and(np.logical_and(val_missing_counts >= p[0], 
                                                               val_missing_counts <= p[1]),
                                                 val_masks)
            partition_test_mask = np.logical_and(np.logical_and(test_missing_counts >= p[0], 
                                                               test_missing_counts <= p[1]),
                                                 test_masks)
            
            predictions = self.predictions
            returns = np.concatenate([train_returns, val_returns, test_returns], axis=0)
            masks = np.concatenate([partition_train_mask, partition_val_mask,  partition_test_mask], axis=0)
            r2 = self.r2(predictions, returns, masks)
            xs_r2 = self.xs_r2(predictions, returns, masks)
            train_count = train_returns.shape[0] + val_returns.shape[0]
            train_preds, test_preds = self.predictions[:train_count], self.predictions[train_count:]
            srs = np.mean(train_preds) / np.std(train_preds), np.mean(test_preds) / np.std(test_preds)
            ret_vals.append((r2, xs_r2, srs))
        return ret_vals
       
    def r2(self, predictions, returns, mask):
        tot_return_sq = 0
        tot_resid_sq = 0
        for t in range(returns.shape[0]):
            resid_t = predictions[t] - returns[t, mask[t]]
            tot_resid_sq += np.mean(np.square(resid_t))
            tot_return_sq += np.mean(np.square(returns[t, mask[t]]))
        return 1 - tot_resid_sq / tot_return_sq
    

    def xs_r2(self, predictions, returns, mask):
        T = predictions.shape[0]
        sum_squared_expected_returns = 0
        sum_squared_expected_residuals = 0
        for i in range(returns.shape[1]):
            reses = []
            p_inds = mask[:,i]
            ti = np.sum(p_inds)
            if ti > 0:
                preds = predictions[p_inds]
                reses = preds - returns[p_inds, i]
                sum_squared_expected_returns += (ti / T) * np.mean(returns[p_inds, i]) ** 2
                sum_squared_expected_residuals += (ti / T) * np.mean(reses) ** 2
        return 1 - sum_squared_expected_residuals / sum_squared_expected_returns