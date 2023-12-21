import numpy as np
from tqdm.notebook import tqdm
import pandas as pd


def percentile_rank(x, UNK=np.nan):
    mask = np.logical_not(np.isnan(x))
    x_copy = np.copy(x)
    x_mask = x_copy[mask]
    n = len(x_mask)
    if n > 1:
        temp = [(i, x_mask[i]) for i in range(n)]
        temp_sorted = sorted(temp, key=lambda t: t[1])
        idx = sorted([(temp_sorted[i][0], i) for i in range(n)], key=lambda t: t[0])
        x_copy[mask] = np.array([idx[i][1] for i in range(n)]) / (n - 1)
    elif n == 1:
        x_copy[mask] = 0.5
    return x_copy

def percentile_rank_panel(char_panel):
    ret_panel = np.zeros(char_panel.shape)
    ret_panel[:, :, :] = np.nan
    for t in tqdm(range(char_panel.shape[0])):
        for i in range(char_panel.shape[2]):
            ret_panel[t, :, i] = percentile_rank(char_panel[t, :, i])
        assert np.sum(np.isnan(ret_panel[t])) > 0, 'something fucky'
    return ret_panel

def get_data_dataframe(data_panel, return_panel, char_names, dates, permnos, monthly_updates, mask):
    T, N, C = data_panel.shape
    if mask is None:
        nonnan_returns = np.argwhere(np.logical_or(~np.isnan(return_panel), np.any(~np.isnan(data_panel),
                                                                                   axis=2)))
        num_nonnan_returns = np.sum(np.logical_or(~np.isnan(return_panel), np.any(~np.isnan(data_panel),
                                                                                   axis=2)))
    else:
        nonnan_returns = np.argwhere(mask)
        num_nonnan_returns = np.sum(mask)
    
    data_matrix = np.zeros((num_nonnan_returns, C+4))
    columns = np.append(char_names, ["return", "date", "permno", "monthly_update"])
    for i in range(nonnan_returns.shape[0]):
        nonnan_return = nonnan_returns[i]
        data_matrix[i,:C] = data_panel[nonnan_return[0], nonnan_return[1], :]
        data_matrix[i,C] = return_panel[nonnan_return[0], nonnan_return[1]]
        data_matrix[i,C+1] = dates[nonnan_return[0]]
        data_matrix[i,C+2] = permnos[nonnan_return[1]]
        data_matrix[i,C+3] = monthly_updates[nonnan_return[0], nonnan_return[1]]
    
    chars_and_returns_df = pd.DataFrame(data_matrix)
    chars_and_returns_df.columns = columns
    
    return chars_and_returns_df

def get_data_panel(path, rf_path, computstat_data_present_filter=True, financial_firm_filter=True, start_date=None):
    data = pd.read_feather(path)
    if start_date is not None:
        data = data.loc[data.date >= start_date]
    print(data.columns)
    dates = data['date'].unique()
    dates.sort()
    permnos = data['permno'].unique().astype(int)
    permnos.sort()
    rf_data = pd.io.parsers.read_csv(rf_path).to_numpy()

    date_vals = [int(date) for date in dates]
    chars = np.array(data.columns.tolist()[:-4])
    print(chars)
    chars.sort()

    char_data = np.zeros((len(date_vals), permnos.shape[0], len(chars)))
    monthly_updates = np.zeros((len(date_vals), permnos.shape[0]))
    char_data[:, :, :] = np.nan
    returns = np.zeros((len(date_vals), permnos.shape[0]))
    returns[:, :] = np.nan
    rfts = []

    permno_map = np.zeros(int(max(permnos)) + 1, dtype=int)
    for i, permno in enumerate(permnos):
        permno_map[permno] = i

    for i, date in enumerate(dates):
        date_data = data.loc[data['date'] == date].sort_values(by='permno')
        date_permnos = date_data['permno'].to_numpy().astype(int)
        permno_inds_for_date = permno_map[date_permnos]
        char_data[i, permno_inds_for_date, :] = date_data[chars].to_numpy()
        monthly_updates[i, permno_inds_for_date] = date_data["monthly_update"].to_numpy()
        returns[i, permno_inds_for_date] = date_data['return'].to_numpy()
        rft_idx = np.argwhere(rf_data[:,0] == str(int(date // 100)))[0][0]
        rfts.append(float(rf_data[rft_idx,1]) / 100)

    percentile_rank_chars = percentile_rank_panel(char_data) - 0.5
    
    assert np.all(np.isnan(percentile_rank_chars) == np.isnan(char_data))
    
    if computstat_data_present_filter:
        cstat_permnos = pd.read_csv("../data/compustat_permnos.csv")["PERMNO"].to_numpy()
        permno_filter = np.isin(permnos, cstat_permnos)
        percentile_rank_chars = percentile_rank_chars[:,permno_filter,:]
        char_data = char_data[:,permno_filter,:]
        permnos = permnos[permno_filter]
        monthly_updates = monthly_updates[:,permno_filter]
        returns = returns[:,permno_filter]
        
    if financial_firm_filter:
        sic_fic = pd.read_csv("../data/sic_fic.csv")
        non_fininancial_permnos = ~np.isin(permnos, sic_fic.loc[sic_fic['sic']//1000 == 6]['LPERMNO'].unique())
        percentile_rank_chars = percentile_rank_chars[:,non_fininancial_permnos,:]
        char_data = char_data[:,non_fininancial_permnos,:]
        permnos = permnos[non_fininancial_permnos]
        monthly_updates = monthly_updates[:,non_fininancial_permnos]
        returns = returns[:,non_fininancial_permnos]
    
    return percentile_rank_chars, char_data, chars, date_vals, returns, permnos, rfts, monthly_updates