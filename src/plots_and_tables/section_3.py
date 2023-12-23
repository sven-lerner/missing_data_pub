from plots_and_tables import plot_base
import numpy as np
from abc import ABC
import matplotlib.pyplot as plt

def get_ordered_missing_mat(ct):
    missing_0 = np.isnan(ct)
    missing_counts_by_stock = np.sum(missing_0, axis=1)
    missing_counts_by_char = np.sum(missing_0, axis=0)
    stock_inds_orderd_by_missing = np.array([x[1] for x in sorted([(x, i) for 
                                                                   i, x in enumerate(missing_counts_by_stock)])])

    char_inds_orderd_by_missing = np.array([x[1] for x in sorted([(x, i) for 
                                                                  i, x in enumerate(missing_counts_by_char)])])
    img = np.isnan(ct)[stock_inds_orderd_by_missing, :][:,char_inds_orderd_by_missing]
    return img, char_inds_orderd_by_missing

class SectionThreePlotBase(plot_base.PaperPlot, ABC):
    section = 'section3'

class SectionThreeTableBase(plot_base.PaperTable, ABC):
    section = 'section3'

class MissingMARPlot(SectionThreePlotBase):
    
    name = 'missing_image_mar'
    description = ""

    def setup(self, return_panel, percentile_rank_chars, chars, date_ind):
        present_returns = ~np.isnan(return_panel[date_ind])
        data_0 = percentile_rank_chars[date_ind, present_returns]
        percent_missing = np.sum(np.isnan(data_0)) / (data_0.shape[0] * data_0.shape[1])
        mar_mat = (np.random.rand(data_0.shape[0], data_0.shape[1]) < percent_missing) * 1.0
        mar_mat[mar_mat == 1] = np.nan
        mar_img, ordering = get_ordered_missing_mat(mar_mat)
        plt.figure(figsize=(20, 10))
        plt.xticks([])
        plt.minorticks_off()
        plt.yticks([])
        plt.imshow(mar_img,  interpolation='nearest', aspect='auto')

class MissingOnDatePlot(SectionThreePlotBase):

    name = 'missing-'
    description = ""
    
    def setup(self, return_panel, percentile_rank_chars, date_ind, dates, chars):
        self.name = f"{self.name}-{dates[date_ind]}"
        present_returns = ~np.isnan(return_panel[date_ind])
        data_0 = percentile_rank_chars[date_ind, present_returns]
        img, ordering = get_ordered_missing_mat(data_0)
        size = 100
        shrunked_img = np.zeros((100, 46))
        step_size = int(data_0.shape[0] / 100)
        plt.figure(figsize=(20, 10))
        plt.imshow(img,  interpolation='nearest', aspect='auto')
        plt.xticks(np.arange(45), chars[ordering], rotation=90)
        plt.yticks([])
        plt.gca().tick_params(axis='both', which='major', labelsize=25)
        plt.minorticks_off()
        
