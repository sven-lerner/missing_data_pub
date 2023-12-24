from models.model import ModelBase
import numpy as np
from models import ipca_util
from tqdm import tqdm
from collections import defaultdict
from models.ipca import InstrumentedPCA
from models.ipca_util import CHAR_COLS, fit_ipca, calculate_sharpe_ratio, calculate_r2, calculate_is_cross_sectional_r2,\
    calculate_efficient_portofolio, get_f_t_from_first_order_conditions, predict, calcuate_XS_r2

class IpcaModel(ModelBase):

    def __init__(self, name, num_factors, num_chars=46, iter_tol=10e-6, maxiter=1000,
                intercept=False, alpha=0, l1_ratio=1.0):
        super().__init__(self, False, 1000)
        self.train_fitter = None
        self.num_factors = num_factors
        self.num_chars = num_chars
        self.in_factors = []
        self.out_factors = []
        self.iter_tol = iter_tol
        self.maxiter=maxiter
        self.intercept = intercept
        self.alpha = alpha
        self.l1_ratio=l1_ratio

    def get_train_test_data(self, train_path, val_path, test_path):
        train_return, train_individualFeature, train_mask, \
            val_return, val_individualFeature, val_mask, \
            test_return, test_individualFeature, test_mask, rfts,\
            test_missing_counts = self.get_precomputed_data(train_path, val_path, test_path,
                                                                                            get_test_missing_counts=True)
        self.rfts = rfts
        train_return = np.concatenate([train_return, val_return], axis=0)
        train_individualFeature = np.concatenate([train_individualFeature, val_individualFeature], axis=0)
        train_mask = np.concatenate([train_mask, val_mask], axis=0)
        return train_return, train_individualFeature, train_mask, test_return, test_individualFeature, test_mask, \
            test_missing_counts

    
    def fit(self, return_panel, char_panel, masks, num_months_train):
        self.NUM_MONTHS_TRAIN = num_months_train
        
        kelly_char_panel_train, kelly_return_panel_train, kelly_ind_panel_train, _, _, _, _ = \
            self.get_kelly_x_y(char_panel[:num_months_train], return_panel[:num_months_train],
                               masks[:num_months_train])
        
        kelly_char_panel, kelly_return_panel, kelly_ind_panel, t_inds, zts, yts, pts =  self.get_kelly_x_y(char_panel, 
                                                                                                           return_panel,
                                                                                                           masks)

        print('fitting on the train data')
        print(self.iter_tol)
        self.train_fitter = InstrumentedPCA(n_factors=self.num_factors, iter_tol=self.iter_tol, intercept=self.intercept,
                                           l1_ratio=self.l1_ratio, alpha=self.alpha, max_iter=self.maxiter)
        print(self.train_fitter.max_iter)
        print("l-1")
        res = self.train_fitter.fit(kelly_char_panel_train, kelly_return_panel_train.squeeze(), kelly_ind_panel_train,
                              quiet=True)
        print("l-2")
        self.in_factors = self.train_fitter.Factors
        
        self.g_b = self.train_fitter.Gamma
        print('fit the train data, fittings the out of sample data')

        for t in tqdm(list(range(self.NUM_MONTHS_TRAIN, len(zts)))):
            zt, yt = zts[t], yts[t]
            ft = get_f_t_from_first_order_conditions(self.g_b, zt, yt, None,
                                                     interact_characteristics_fully=True)

            self.out_factors.append(ft)
        return

    def eval_metrics(self, train_returns, train_individualFeatures, train_masks, test_returns,
                     test_individualFeatures, test_masks, test_missing_counts, test_partitions=None):

        kelly_char_panel_train, kelly_return_panel_train, kelly_ind_panel_train, _, _, _, _ = \
            self.get_kelly_x_y(train_individualFeatures, train_returns, train_masks)

        tot_r2 = self.train_fitter.score(kelly_char_panel_train, kelly_return_panel_train, kelly_ind_panel_train)

        preds = self.train_fitter.predict_panel(kelly_char_panel_train, kelly_ind_panel_train, 
                                    np.unique(kelly_ind_panel_train[:,1]).shape[0])
        xs_r2 = self.calculate_is_cross_sectional_r2(preds, kelly_return_panel_train.squeeze(), kelly_ind_panel_train)

        factors = self.train_fitter.Factors
        risk_free_rates = self.rfts[:factors.shape[1]]
        weights = calculate_efficient_portofolio(factors, risk_free_rates)
        mv_portfolio_excess_return = weights.dot(factors) - risk_free_rates
        print(mv_portfolio_excess_return.shape)
        sr = calculate_sharpe_ratio(mv_portfolio_excess_return)

        if test_partitions is None:
            test_partitions = [(0, 45)]
        
        ret_metrics = []
        for p in test_partitions:
            partition_test_mask = np.logical_and(np.logical_and(test_missing_counts >= p[0], 
                                                           test_missing_counts <= p[1]),
                                             test_masks)
            return_panel = np.concatenate([train_returns, test_returns], axis=0)
            char_panel = np.concatenate([train_individualFeatures, test_individualFeatures], axis=0)
            masks = np.concatenate([train_masks, partition_test_mask], axis=0)

            kelly_char_panel, kelly_return_panel, kelly_ind_panel, t_inds, zts, yts, pts = \
                self.get_kelly_x_y(char_panel, return_panel, masks)

            oos_total_r2, oos_xs_r2, oos_sr =  self.get_oos_metrics(zts, yts, pts, self.rfts[factors.shape[1]:])
            ret_metrics.append([oos_total_r2, oos_xs_r2, oos_sr])

        return [['total r2', 'xs-r2', 'sr'], [tot_r2, xs_r2, sr], ret_metrics]
                

    def calculate_is_cross_sectional_r2(self, preds, gts, inds):
        T = np.unique(inds[:,1]).shape[0]
        sum_squared_expected_returns = 0
        sum_squared_expected_residuals = 0
        for permno in np.unique(inds[:,0]):
            p_inds = permno == inds[:,0]
            ti = np.sum(p_inds)
            res = preds[p_inds] - gts[p_inds]
            sum_squared_expected_returns += (ti / T) * np.mean(gts[p_inds]) ** 2
            sum_squared_expected_residuals += (ti / T) * np.mean(res) ** 2
        return 1 - sum_squared_expected_residuals / sum_squared_expected_returns

    def get_kelly_x_y(self, char_panel, return_panel, masks):
        kelly_char_panel = np.zeros((np.sum(masks), self.num_chars))
        kelly_return_panel = np.zeros((np.sum(masks), 1))
        kelly_ind_panel = np.zeros((np.sum(masks), 2))

        zts, yts, pts = [], [], []

        ids = np.arange(char_panel.shape[1])
        curr = 0
        t_inds = [0]
        for t in range(char_panel.shape[0]):
            t_count = np.sum(masks[t]) + curr
            kelly_char_panel[curr:t_count] = char_panel[t][masks[t], :]
            kelly_return_panel[curr:t_count] = np.expand_dims(return_panel[t, masks[t]], axis=1)
            kelly_ind_panel[curr:t_count, 0] = ids[masks[t]]
            kelly_ind_panel[curr:t_count, 1] = t
            curr = t_count
            zts.append(char_panel[t][masks[t], :])
            yts.append(np.expand_dims(return_panel[t, masks[t]], axis=1))
            pts.append(ids[masks[t]])
            t_inds.append(curr)


        return kelly_char_panel, kelly_return_panel, kelly_ind_panel, t_inds, zts, yts, pts
    
    def get_oos_metrics(self, zts, yts, pnts, rfts):
        oos_mv_portfolio_returns = []
        total_residuals = []
        returns = []
        assets_to_returns_and_preds = defaultdict(dict)
        np_factors = np.copy(self.in_factors)

        for t, data in enumerate(zip(zts[self.NUM_MONTHS_TRAIN:], yts[self.NUM_MONTHS_TRAIN:])):
            zt, yt = data
            returns.append(yt)
            ft = self.out_factors[t]

            np_factors = np.hstack([np_factors, ft])

            f_pred = predict(self.g_b, ft, zt)

            total_residuals.append(yt - f_pred)

            #np_factors = fitter.Factors
            np_risk_free_rates = self.rfts[:self.NUM_MONTHS_TRAIN+t]
            weights = calculate_efficient_portofolio(np_factors[:,:self.NUM_MONTHS_TRAIN+t], np_risk_free_rates.squeeze())

            oos_return = weights.T.dot(ft)
            oos_mv_portfolio_returns.append(oos_return)

            pts = pnts[self.NUM_MONTHS_TRAIN + t]
            for permno, res, ret in zip(pts, yt - f_pred, yt):
                if 'returns' not in assets_to_returns_and_preds[permno]:
                    assets_to_returns_and_preds[permno]['returns'] = []
                    assets_to_returns_and_preds[permno]['residuals'] = []
                assets_to_returns_and_preds[permno]['returns'].append(ret)
                assets_to_returns_and_preds[permno]['residuals'].append(res)

        tot_return = sum(y.T.dot(y) for y in returns)
        total_r2 = 1 - sum((res).T.dot(res) for res in total_residuals) / tot_return

        T = np_factors.shape[0]

        cross_sectional_r2 = calcuate_XS_r2(assets_to_returns_and_preds, T)
        return total_r2[0][0], cross_sectional_r2, calculate_sharpe_ratio(np.array(oos_mv_portfolio_returns) - rfts)