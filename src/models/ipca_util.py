from tqdm.notebook import tqdm
from collections import defaultdict
import time

import numpy as np
import re
from models.ipca.ipca.ipca import InstrumentedPCA

approx_matcher = re.compile('rank\-[0-9]\-approx')

DATE_COLS = ['yy', 'mm', 'date']
ID_COL = 'permno'
TARGET_COL = 'ret'
CHAR_COLS = ['A2ME', 'AC', 'AT', 'ATO', 'BEME', 'Beta', 'C', 'CF',
             'CF2P', 'CTO', 'D2A', 'D2P', 'DPI2A', 'E2P', 'FC2Y', 'IdioVol',
             'Investment', 'Lev', 'LME', 'LT_Rev', 'LTurnover', 'MktBeta', 'NI',
             'NOA', 'OA', 'OL', 'OP', 'PCM', 'PM', 'PROF', 'Q', 'r2_1', 'r12_2',
             'r12_7', 'r36_13', 'Rel2High', 'Resid_Var', 'RNA', 'ROA', 'ROE',
             'S2P', 'SGA2S', 'Spread', 'ST_REV', 'SUV', 'Variance']
CHAR_COLS = sorted(CHAR_COLS)
PAST_RETURN_CHARS = ['r2_1', 'r12_2', 'r12_7', 'r36_13', 'ST_REV', 'LT_Rev']
TRADING_FRICTIONS_CHARS = ['AT', 'Beta', 'IdioVol', 'LME', 'LTurnover', 'MktBeta', 'Rel2High', 'Resid_Var',
                           'Spread', 'SUV', 'Variance']
VALUE_CHARS = ['A2ME', 'BEME', 'C', 'CF', 'CF2P', 'D2P', 'E2P', 'Q', 'S2P', 'Lev']
INVESTMENT_CHARS = ['Investment', 'NOA', 'DPI2A', 'NI']
PROFITABILITY_CHARS = ['PROF', 'ATO', 'CTO', 'FC2Y', 'OP', 'PM', 'RNA', 'ROA', 'ROE', 'SGA2S', 'D2A']
INTANGIBLES_CHARS = ['AC', 'OA', 'OL', 'PCM']

TOLERANCE = 10e-8


def format_date(date_in_yyyymmdd):
    date = str(date_in_yyyymmdd)
    return f'{date[:4]}-{date[4:6]}-{date[6:]}'


def extract_z_y_from_data_at_month(month, data_by_date, scale="zihan", id_col="permno", rf_data=None):
    zt, yt, _ = extract_z_y_rf_from_data_at_month(month, data_by_date, scale, id_col, rf_data=rf_data)
    return zt, yt


def percentile_rank(x, UNK=-99.99):
    mask = (x != UNK)
    x_mask = x[mask]
    n = len(x_mask)
    temp = [(i, x_mask[i]) for i in range(n)]
    temp_sorted = sorted(temp, key=lambda t: t[1])
    idx = sorted([(temp_sorted[i][0], i) for i in range(n)], key=lambda t: t[0])
    x[mask] = np.array([idx[i][1] for i in range(n)]) / (n - 1)
    return x


def extract_z_y_rf_from_data_at_month(month, data_by_date, scale="zihan", id_col="permno", rf_data=None,
                                      char_cols=CHAR_COLS):
    """
    extract characteristics from target month, returns from target month + 1 where
    assets are present in both months
    Parameters:
        offset_returns_by_one:
        subtract_rf: whether or not to make returns excess
        month: target month
        data_by_date: data frame of characteristic data grouped by month
        scale: scaling method for characterstics "paper" for method used in paper, "naive" simply scales to [-0.5, -0.5]
            but preserves relative spread between values as opposed to ranking
    """
    date_data = data_by_date.get_group(month)

    y_t = date_data.sort_values(by=id_col)[TARGET_COL].to_numpy()

    pt = date_data[id_col].to_numpy()

    rf_t = None
    if rf_data is not None:
        rf_t = float(rf_data.loc[rf_data.iloc[:, 0] == str(month)[:-2]]['RF'].to_numpy()) / 100

    z_t = date_data.sort_values(by=id_col)[char_cols]
    if scale == "naive":
        z_t = z_t.to_numpy()
        z_t_scaled = (z_t - np.min(z_t, axis=0)) / (np.max(z_t, axis=0) - np.min(z_t, axis=0)) - .05
        return z_t_scaled, y_t.reshape(y_t.shape[0], 1), rf_t, pt
    elif scale == "paper":
        z_t_ranked = (z_t.rank(method='dense').to_numpy() - 1) / z_t.shape[0] - 0.5
        return z_t_ranked, y_t.reshape(y_t.shape[0], 1), rf_t, pt
    elif scale == "zihan":
        z_t = z_t.to_numpy()
        for i in range(z_t.shape[1]):
            z_t[:, i] = percentile_rank(z_t[:, i])
        return z_t, y_t.reshape(y_t.shape[0], 1), rf_t, pt
    else:
        z_t = z_t.to_numpy()
        return z_t, y_t.reshape(y_t.shape[0], 1), rf_t, pt


def get_chars_returns_indexes(monthly_z_data, monthly_y_data):
    assert len(monthly_y_data) == len(monthly_z_data)
    indexes = np.vstack(np.repeat(i, monthly_z_data[i].shape[0]).reshape((monthly_z_data[i].shape[0], 1)) for i in
                        range(len(monthly_z_data)))
    stacked_z = np.vstack(monthly_z_data)
    stacked_y = np.vstack(monthly_y_data)
    return stacked_z, stacked_y, indexes


def get_initial_gamma(zs, ys, K):
    """
    Return first K eigenvectors of second moment matrix of managed returns as
    initial point for gamma_b matrix
    """
    xs = [z_t.T.dot(y_t) / (y_t.shape[0]) for z_t, y_t in zip(zs, ys)]
    sum_second_moment = sum([x.dot(x.T) for x in xs])
    vals, vects = np.linalg.eig(sum_second_moment)
    return vects[:, :K]


def get_f_t_from_first_order_conditions(gamma_b, z_t, y_t, gamma_alpha=None, interact_characteristics_fully=True,
                                        char_interactions=None):
    """
    Apply first order conditions for f_t, step 1 of ALS
    """
    if interact_characteristics_fully:
        inv_value = np.linalg.inv(gamma_b.T.dot(z_t.T.dot(z_t)).dot(gamma_b))
    else:
        if char_interactions is None:
            inv_value = np.linalg.inv((gamma_b.T.dot(gamma_b)))
        elif approx_matcher.match(char_interactions):
            rank = int(char_interactions.split('-')[1])
            temp = np.zeros((z_t.shape[1], z_t.shape[1]))
            vals, vects = np.linalg.eig(z_t.T.dot(z_t))
            for i, eigval in enumerate(vals):
                if i < rank:
                    print((eigval * vects[i].dot(vects[i].T)).shape)
                    temp += eigval * vects[i].dot(vects[i].T)
                else:
                    temp += vects[i].dot(vects[i].T)
            inv_value = np.linalg.inv((gamma_b.T.dot(temp).dot(gamma_b)))
        else:
            raise NotImplementedError('have not added other methods yet')
    if gamma_alpha is None:
        ft = inv_value.dot(gamma_b.T).dot(z_t.T).dot(y_t)
        size = ft.shape[0]
        return ft.reshape(size, 1)
    else:
        ft = inv_value.dot(gamma_b.T).dot(z_t.T).dot(
            y_t - z_t.dot(gamma_alpha))
        size = ft.shape[0]
        return ft.reshape(size, 1)


from numba import jit

@jit(nopython=True)
def get_outer_product_numba(zts,fts, yts):
    kron_sum = np.zeros((322, 1))
    for i in range(len(zts)):
        kron_sum += np.kron(zts[i], fts[i].T).T.dot(yts[i])
    return kron_sum


def get_gamma_from_first_order_conditions(fts, zts, yts, L, K):
    """
    Apply first order conditions for gamma, step 2 of ALS
    """
    one = time.time()
    outer_kroenecker_sum = sum([np.kron(z_t.T.dot(z_t), f_t.dot(f_t.T)) for z_t, f_t in zip(zts, fts)])
    print('one', time.time() - one)
    two = time.time()
    outer_kroenecker_sum_inv = np.linalg.inv(outer_kroenecker_sum)
    print('two', time.time() - two)
    # three = time.time()
    # inner_kroenecker = [np.kron(z_t, f_t.T).T.dot(y_t) for z_t, f_t, y_t in zip(zts, fts, yts)]
    # print('three', time.time() - three)
    four = time.time()
    # inner_kroenecker_sum = sum(inner_kroenecker)
    inner_kroenecker_sum = get_outer_product_numba(zts, fts, yts)
    # print(inner_kroenecker_sum.shape)
    print('four', time.time() - four)
    five = time.time()
    res1 = outer_kroenecker_sum_inv.dot(inner_kroenecker_sum).reshape(L, K)
    print('five', time.time() - five)
    # #
    # outer_kroenecker_2 = [np.kron(z_t, f_t.T) for z_t, f_t in zip(zts, fts)]
    # outer_kroenecker_sum_2 = sum([A.T.dot(A) for A in outer_kroenecker_2])
    # # Y += A.T.dot(returns[i, nonzero_index])[:, np.newaxis]
    # inner_kroenecker_sum_2 = sum([A.T.dot(y_t) for A, y_t in zip(outer_kroenecker_2, yts)])
    #
    # # res2 = np.linalg.pinv(outer_kroenecker_sum_2).dot(inner_kroenecker_sum_2).reshape(L, K)
    # res2 = np.linalg.inv(outer_kroenecker_sum_2).dot(inner_kroenecker_sum_2).reshape(L, K)

    # print(np.max(np.abs(res1 - res2)))

    # return outer_kroenecker_sum_inv.dot(inner_kroenecker_sum).reshape(L, K)
    return res1


def fit_ipca(zts, yts, K=5, L=len(CHAR_COLS), fix_alpha_zero=True,
             max_iter=2, start_gamma=None, enforce_constraints=True,
             log_convergence=True):
    """
    Fit model to characteristics data using IPCA
    """
    # allow for hot-starting gamma

    if start_gamma is None:
        gamma_b = get_initial_gamma(zts, yts, K)
        gamma_a = np.zeros((zts[0].shape[1], 1))
    else:
        gamma_a, gamma_b = start_gamma

    fts = [get_f_t_from_first_order_conditions(gamma_b, zt, yt, gamma_a,
                                               interact_characteristics_fully=True)
           for zt, yt in zip(zts, yts)]
    three = time.time()
    last_fts_change, last_gamma_change = 0, 0
    for i in range(max_iter):
        one = time.time()
        # ALS step 1
        prev_fts = fts
        fts = [get_f_t_from_first_order_conditions(gamma_b, zt, yt, gamma_a,
                                                   interact_characteristics_fully=True)
               for zt, yt in zip(zts, yts)]
        two = time.time()
        print('1-2', two - one)
        # ALS step 2
        prev_gamma_b = gamma_b
        if fix_alpha_zero:
            gamma_b = get_gamma_from_first_order_conditions(fts, zts, yts, L, K)
            three = time.time()
            print('2-3', three - two)
        else:
            padded_fts = [np.vstack([[1], f]) for f in fts]
            gamma = get_gamma_from_first_order_conditions(padded_fts, zts, yts, L, K + 1)
            gamma_a = gamma[:, 0].reshape(L, 1)
            gamma_b = gamma[:, 1:]

            q, r = np.linalg.qr(gamma_b)
            projection = np.sum(q.dot(q.T.dot(gamma_a)), axis=1).reshape(L, 1)
            gamma_a = gamma_a - projection

        if enforce_constraints:
            # enforce constraints to certify unique gamma/f
            gamma_b, fts = project_constraints(fts, gamma_b, K)
            four = time.time()
            print('3-4', four - three)

        last_gamma_change = np.max(np.abs(prev_gamma_b - gamma_b))
        last_fts_change = max([np.max(np.abs(fnew - fold)) for fnew, fold in zip(fts, prev_fts)])
        # max_change = max([np.linalg.norm(fnew-fold)/np.linalg.norm(fnew) for fnew, fold in zip(fts, prev_fts)])
        if max(last_gamma_change, last_fts_change) < TOLERANCE and i > 0:
            # if max_change < TOLERANCE and i > 0:
            if log_convergence:
                print(f'converged, terminating after {i} iterations')
            break
    if log_convergence:
        print(last_fts_change, last_gamma_change)
    return gamma_a, gamma_b, fts


def fit_factors_alternative(gamma_b, gamma_a, yts, zts, characterisitic_interactions_allowed=None):
    fts_n_c_i = [get_f_t_from_first_order_conditions(gamma_b, zt, yt, gamma_a,
                                                     interact_characteristics_fully=False,
                                                     char_interactions=characterisitic_interactions_allowed)
                 for zt, yt in zip(zts, yts)]
    gamma_b_n_c_i = get_gamma_from_first_order_conditions(fts_n_c_i, zts, yts, gamma_b.shape[0], gamma_b.shape[1])
    gamma_a_n_c_i = gamma_a
    return fts_n_c_i, gamma_b_n_c_i, gamma_a_n_c_i


def project_constraints(fts, gamma_b, K):
    # enforce gamma_b' dot gamma_b = I_k, descending diagonal of factors 2nd moment
    F_New = np.stack(fts, axis=1).reshape((K, len(fts)))
    R1 = np.linalg.cholesky(gamma_b.T.dot(gamma_b)).T
    R2, _, _ = np.linalg.svd(R1.dot(F_New).dot(F_New.T).dot(R1.T))
    gamma_b = np.linalg.lstsq(gamma_b.T, R1.T, rcond=None)[0].dot(R2)
    F_New = np.linalg.solve(R2, R1.dot(F_New))

    # enforce sign convention for gamma_b and f_ts
    sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
    sg[sg == 0] = 1
    gamma_b = np.multiply(gamma_b, sg.T)
    F_New = np.multiply(F_New, sg)
    fts = [F_New[:, i].reshape((K, 1)) for i in range(F_New.shape[1])]
    return gamma_b, fts


def predict(gamma_b, ft, zt, gamma_a=None):
    """
    Predict returns given factors, characteristics, and loading matrix
    """
    if gamma_a is None:
        gamma_a = np.zeros((gamma_b.shape[0], 1))
    return zt.dot(gamma_a) + zt.dot(gamma_b).dot(ft)


def calculate_r2(gamma_b, fts, zts, yts, gamma_a=None):
    """
    Calculate in sample r2 values
    """
    preds = [predict(gamma_b, ft, zt, gamma_a) for zt, ft in zip(zts, fts)]
    tot_return = sum(y.T.dot(y) for y in yts)

    lmbda = sum(fts) / len(fts)
    lmbda_preds = [predict(gamma_b, lmbda, zt, gamma_a) for zt in zts]

    total_r2 = 1 - sum((pred - y).T.dot(pred - y) for pred, y in zip(preds, yts)) / tot_return
    pred_r2 = 1 - sum((pred - y).T.dot(pred - y) for pred, y in zip(lmbda_preds, yts)) / tot_return

    return total_r2[0][0], pred_r2[0][0]


def calculate_oos_r2(kelly_char_panel_train, kelly_return_panel_train, kelly_ind_panel_train,
                     t_inds, zts, yts, rfts, pnts, num_factors,
                     char_cols=CHAR_COLS, rf_data=None, interact_characteristics=True,
                     iter_tol=1e-5,
                     NUM_MONTHS_TRAIN=450,
                     recalculate_gb=False,
                     fitter=None):
    """
    Calculate out of sample r2 values
    """
    L = len(char_cols)

    oos_mv_portfolio_returns = []
    total_residuals = []
    factors = []
    g_b = None
    gammas = []
    predicted_residuals = []
    returns = []
    assets_to_returns_and_preds = defaultdict(dict)
    np_factors=None

    for t in tqdm(list(range(NUM_MONTHS_TRAIN, len(zts)))):
        if g_b is None or recalculate_gb:
            fitter.fit(kelly_char_panel_train[:t_inds[t]], kelly_return_panel_train[:t_inds[t]],
                       kelly_ind_panel_train[:t_inds[t]], Gamma=g_b, quiet=True)
            
            np_factors = fitter.Factors
            g_b = fitter.Gamma
#             print(g_b)
        
        zt, yt = zts[t], yts[t]
        returns.append(yt)

        ft = get_f_t_from_first_order_conditions(g_b, zt, yt, None,
                                                 interact_characteristics_fully=interact_characteristics)
#         print(ft)

        if not interact_characteristics:
            fts, g_b, g_a = fit_factors_alternative(g_b, None, yts[:t], zts[:t])

        gammas.append((None, g_b))
        factors.append(ft)
        np_factors = np.hstack([np_factors, ft])

        #lmbda = np.mean(np.hstack(fitter.Factors + [ft]), axis=1).reshape((num_factors, 1))
        lmbda = np.mean(np_factors, axis=1).reshape((num_factors, 1))

        f_pred = predict(g_b, ft, zt)
        lmbda_pred = predict(g_b, lmbda, zt)

        total_residuals.append(yt - f_pred)
        predicted_residuals.append(yt - lmbda_pred)

        #np_factors = fitter.Factors
        np_risk_free_rates = np.hstack(rfts[:t])
        weights = calculate_efficient_portofolio(np_factors[:,:t], np_risk_free_rates)

        oos_return = weights.T.dot(ft)
        oos_mv_portfolio_returns.append(oos_return)

        pts = pnts[t]
        for permno, res, ret in zip(pts, yt - f_pred, yt):
            if 'returns' not in assets_to_returns_and_preds[permno]:
                assets_to_returns_and_preds[permno]['returns'] = []
                assets_to_returns_and_preds[permno]['residuals'] = []
            assets_to_returns_and_preds[permno]['returns'].append(ret)
            assets_to_returns_and_preds[permno]['residuals'].append(res)

    tot_return = sum(y.T.dot(y) for y in returns)
    total_r2 = 1 - sum((res).T.dot(res) for res in total_residuals) / tot_return
    pred_r2 = 1 - sum((res).T.dot(res) for res in predicted_residuals) / tot_return

    T = len(factors)

    cross_sectional_r2 = calcuate_XS_r2(assets_to_returns_and_preds, T)
    print(assets_to_returns_and_preds)

    return total_r2[0][0], pred_r2[0][0], cross_sectional_r2, factors, gammas, calculate_sharpe_ratio(
        oos_mv_portfolio_returns)


# def calculate_oos_r2_np(zts, yts, pnts, num_factors, rfts, interact_characteristics=True,
#                      NUM_MONTHS_TRAIN=350, max_iter=10):
#     """
#     Calculate out of sample r2 values
#     """
#     L = zts[0].shape[1]

#     oos_mv_portfolio_returns = []
#     total_residuals = []
#     factors = []
#     gammas = []
#     predicted_residuals = []
#     returns = []

#     prev_gamma = None
#     assets_to_returns_and_preds = defaultdict(dict)

#     for t in tqdm(list(range(NUM_MONTHS_TRAIN, len(zts) - 1))):
#         if prev_gamma is None:
#             g_a, g_b, fts = fit_ipca(zts[:t], yts[:t], K=num_factors, L=L, fix_alpha_zero=True, max_iter=max_iter*2,
#                                      enforce_constraints=False,
#                                      log_convergence=False)
#             prev_gamma = (g_a, g_b)
#         else:
#             g_a, g_b, fts = fit_ipca(zts[:t], yts[:t], K=num_factors, L=L, fix_alpha_zero=True, start_gamma=prev_gamma,
#                                      max_iter=max_iter,
#                                      enforce_constraints=False,
#                                      log_convergence=False)
#             prev_gamma = (g_a, g_b)

#         zt, yt = zts[t], yts[t]
#         returns.append(yt)
#         ft = get_f_t_from_first_order_conditions(g_b, zt, yt, g_a,
#                                                  interact_characteristics_fully=interact_characteristics)
#         if not interact_characteristics:
#             fts, g_b, g_a = fit_factors_alternative(g_b, g_a, yts[:t], zts[:t])

#         gammas.append((g_a, g_b))
#         factors.append(ft)

#         lmbda = np.mean(np.hstack(fts + [ft]), axis=1).reshape((num_factors, 1))

#         f_pred = predict(g_b, ft, zt, g_a)
#         lmbda_pred = predict(g_b, lmbda, zt, g_a)

#         total_residuals.append(yt - f_pred)
#         predicted_residuals.append(yt - lmbda_pred)

#         np_factors = np.hstack(fts)
#         np_risk_free_rates = np.hstack(rfts[:t])
#         weights = calculate_efficient_portofolio(np_factors, np_risk_free_rates)

#         oos_return = weights.T.dot(ft)
#         oos_mv_portfolio_returns.append(oos_return)

#         pts = pnts[t]
#         for permno, res, ret in zip(pts, yt - f_pred, yt):
#             if 'returns' not in assets_to_returns_and_preds[permno]:
#                 assets_to_returns_and_preds[permno]['returns'] = []
#                 assets_to_returns_and_preds[permno]['residuals'] = []
#             assets_to_returns_and_preds[permno]['returns'].append(ret)
#             assets_to_returns_and_preds[permno]['residuals'].append(res)
#         fts.append(ft)

#     tot_return = sum(y.T.dot(y) for y in returns)
#     total_r2 = 1 - sum((res).T.dot(res) for res in total_residuals) / tot_return
#     pred_r2 = 1 - sum((res).T.dot(res) for res in predicted_residuals) / tot_return

#     T = len(factors)

#     cross_sectional_r2 = calcuate_XS_r2(assets_to_returns_and_preds, T)

#     return total_r2[0][0], pred_r2[0][0], cross_sectional_r2, factors, gammas, calculate_sharpe_ratio(
#         oos_mv_portfolio_returns)


def calcuate_XS_r2(assets_to_returns_and_preds, T):
    sum_squared_expected_returns = 0
    sum_squared_expected_residuals = 0
    for permno in assets_to_returns_and_preds:
        ti = len(assets_to_returns_and_preds[permno]['returns'])
        sum_squared_expected_returns += (ti / T) * np.mean(assets_to_returns_and_preds[permno]['returns']) ** 2
        sum_squared_expected_residuals += (ti / T) * np.mean(assets_to_returns_and_preds[permno]['residuals']) ** 2
    return 1 - sum_squared_expected_residuals / sum_squared_expected_returns


def calculate_efficient_portofolio(factors, rf_rates):
#     print((factors.shape, rf_rates.shape))
    f_minus_rf = factors - rf_rates
    mu = np.mean(factors, axis=1)
#     print(mu.shape)
    cov = np.cov(factors)
    weights = np.linalg.lstsq(cov, mu, rcond=None)[0]
    return weights


def calculate_sharpe_ratio(portfolio):
    return np.mean(portfolio) / np.std(portfolio)


def calculate_is_cross_sectional_r2(gamma_b, fts, zts, yts, pts, gamma_a=None):
    """
    Calculate in sample r2 values
    """
    T = len(fts)

    assets_to_returns_and_preds = defaultdict(dict)
    preds = [predict(gamma_b, ft, zt, gamma_a) for zt, ft in zip(zts, fts)]
    for pt, pred, yt in zip(pts, preds, yts):
        res = pred - yt
        for i, permno in enumerate(pt):
            if 'returns' not in assets_to_returns_and_preds[permno]:
                assets_to_returns_and_preds[permno]['returns'] = []
                assets_to_returns_and_preds[permno]['residuals'] = []
            assets_to_returns_and_preds[permno]['returns'].append(yt[i])
            assets_to_returns_and_preds[permno]['residuals'].append(res[i])

    return calcuate_XS_r2(assets_to_returns_and_preds, T)
