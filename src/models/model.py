import numpy as np
from abc import ABC, abstractmethod
from models.model_utils import expand_quantiles, quantize_data


class ModelBase(ABC):

    def __init__(self, name, bin_data=False, num_bins=100, include_missing_count=False):
        self.name = name
        self.bin_data = bin_data
        self.num_bins = num_bins
        self.shift = 0
        self.scale = 1
        self.expand_dims = False
        self.include_missing_count = include_missing_count
        self.intercept = False
        self.factor_cutoff = None
        

    def get_precomputed_data(self, train_path, val_path, test_path, get_test_missing_counts=False):
        tmp = np.load('/home/svenl/data/pelger/data/rfts_None-46-11.npz')
        rfts = tmp['rfts']

        tmp = np.load(train_path)
        data_1 = tmp['data']

        train_return = data_1[:,:,0]
        train_individualFeature = data_1[:,:,1:] + self.shift
        train_mask = (train_return != -99.99)

        tmp = np.load(val_path)
        data_2 = tmp['data']

        val_return = data_2[:,:,0]
        val_individualFeature = data_2[:,:,1:] + self.shift
        val_mask = (val_return != -99.99)

        tmp = np.load(test_path)
        data_3 = tmp['data']

        test_return = data_3[:,:,0]
        test_individualFeature = data_3[:,:,1:] + self.shift
        test_mask = (test_return != -99.99)
        
        if self.bin_data:
#             quantize_data(train_individualFeature, val_individualFeature, test_individualFeature, self.num_bins,
#                          train_mask, val_mask, test_mask)
            if self.num_bins == 5:
                dividors = [-500, -0.3, -0.1, 0.1, 0.3, 500]
                vals = [-.5, -.25, 0, .25, .5]
            elif self.num_bins == 10:
                dividors = [-500, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 500]
                vals = [-.5, -.4, -.3, -.2, -.1, 0.1, 0.2, 0.3, 0.4, .5]
            if self.expand_dims:
                train_individualFeature = expand_quantiles(train_individualFeature, self.num_bins, mask=train_mask,
                                                          dividors=dividors,
                                                          vals=vals)
                val_individualFeature = expand_quantiles(val_individualFeature, self.num_bins, mask=val_mask,
                                                          dividors=dividors,
                                                          vals=vals)
                test_individualFeature = expand_quantiles(test_individualFeature, self.num_bins, mask=test_mask,
                                                          dividors=dividors,
                                                          vals=vals)

        if self.include_missing_count:
            train_missing_counts, val_missing_counts, test_missing_counts = self.get_train_val_test_missing_counts(train_path,
                                                                                                               val_path, 
                                                                                                               test_path)
            print(train_individualFeature.shape, np.expand_dims(train_missing_counts, axis=2).shape)
            train_individualFeature = np.concatenate([train_individualFeature, np.expand_dims(train_missing_counts, axis=2)], 
                                                     axis=2)
            val_individualFeature = np.concatenate([val_individualFeature, np.expand_dims(val_missing_counts, axis=2)], 
                                                   axis=2)
            test_individualFeature = np.concatenate([test_individualFeature, np.expand_dims(test_missing_counts, axis=2)], 
                                                    axis=2)
            
        if self.factor_cutoff is not None:
            train_individualFeature = train_individualFeature[:,:,:self.factor_cutoff]
            val_individualFeature = val_individualFeature[:,:,:self.factor_cutoff]
            test_individualFeature = test_individualFeature[:,:,:self.factor_cutoff]
            
        if self.intercept:
            train_individualFeature = np.concatenate([train_individualFeature, np.ones((train_individualFeature.shape[0],
                                                                                       train_individualFeature.shape[1],
                                                                                       1))], axis=2)
            val_individualFeature = np.concatenate([val_individualFeature, np.ones((val_individualFeature.shape[0],
                                                                                       val_individualFeature.shape[1],
                                                                                       1))], axis=2)
            test_individualFeature = np.concatenate([test_individualFeature, np.ones((test_individualFeature.shape[0],
                                                                                       test_individualFeature.shape[1],
                                                                                       1))], axis=2)
        if get_test_missing_counts:
            test_missing_counts = tmp['missing_counts']
            return train_return, train_individualFeature, train_mask, \
                val_return, val_individualFeature, val_mask, \
                test_return, test_individualFeature, test_mask, rfts, test_missing_counts
        else:
            return train_return, train_individualFeature, train_mask, \
                val_return, val_individualFeature, val_mask, \
                test_return, test_individualFeature, test_mask, rfts

    @abstractmethod
    def fit(self, train_path, val_path, test_path, missing_bounds=None):
        pass

    @abstractmethod
    def eval_metrics(self, train_path, val_path, test_path, partitions=None):
        pass
