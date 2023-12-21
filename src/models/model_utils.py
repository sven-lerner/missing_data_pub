import numpy as np


def expand_quantiles(features, target_dim_expansion, mask, dividors, vals):
#         vals = np.unique(features[mask])
    T, N, C = features.shape
#         assert len(vals) == target_dim_expansion, f'{len(vals)} was not  {target_dim_expansion}, {vals}'
    return_features = np.zeros((T, N, C*(len(dividors)-1)))
    for t in range(T):
        for c in range(C):
            for i, v in enumerate(dividors[:-1]):
                val_mask = np.logical_and(features[t,:,c] >= v, features[t,:,c] <= dividors[i+1])
                if np.sum(val_mask) == 0:
                    print(f'failed at c={c}, v={v}, t={t}')
#                         val_mask = np.min(np.argwhere(features[t,:,c] == vals[i-1]))
#                         return_features[t, val_mask, c * target_dim_expansion + i] = 1
                else:
                    return_features[t, val_mask, c * target_dim_expansion + i] = 1
    return return_features    

    
def quantize_data(train_features, val_features, test_features, num_bins,
                  train_mask, val_mask, test_mask):
    min_val = min([np.min(train_features[train_mask]), 
                   np.min(val_features[val_mask]), 
                   np.min(test_features[test_mask])])
    max_val = max([np.max(train_features), np.max(val_features), np.max(test_features)])
    step_size = (max_val - min_val) / (num_bins)
    print('minval, maxval', min_val, max_val, step_size)
    nan_mask_train = np.isnan(train_features)
    nan_mask_test = np.isnan(test_features)
    nan_mask_val = np.isnan(val_features)
    for i in range(num_bins):
        train_bin_mask = np.logical_and(train_features >= min_val + i * step_size,
                                       train_features <= min_val + (i+1) * step_size)
        val_bin_mask = np.logical_and(val_features >= min_val + i * step_size,
                                       val_features <= min_val + (i+1) * step_size)
        test_bin_mask = np.logical_and(test_features >= min_val + i * step_size,
                                       test_features <= min_val + (i+1) * step_size)
        bin_val = round(min_val + step_size * i + step_size / 2, 2)
        print(i, bin_val, np.sum(train_bin_mask), np.sum(val_bin_mask), np.sum(test_bin_mask))
        train_features[train_bin_mask] = bin_val
        val_features[val_bin_mask] = bin_val
        test_features[test_bin_mask] = bin_val
#     bins = np.arange(min_val, max_val, step_size)
#     bin_vals = np.arange(min_val, max_val + step_size, step_size)
#     step_size = 1 / (num_bins)
#     bins = np.arange(-0.5, 0.5, step_size) + step_size
#     bin_vals = np.append(np.arange(-0.5, 0.5 + step_size, 1 / (num_bins - 1)), [np.nan])
#     train_features[:,:,:] = bin_vals[np.digitize(train_features, bins)]
#     val_features[:,:,:] = bin_vals[np.digitize(val_features, bins)]
#     test_features[:,:,:] = bin_vals[np.digitize(test_features, bins)]
    train_features[nan_mask_train] = np.nan
    val_features[nan_mask_val] = np.nan
    test_features[nan_mask_test] = np.nan
