# Result Replication Code

## Notes about Data

Note, there are two differences between the data-set we have provided and the data-set used in the paper
1. The data-set we have provided is a truncated version of the data-set in the paper. This is because the full data-set is around 20 GB, and running all the results requires making multiple copies of the data-set, which is very time and space consuming
2. The returns in the data-set we have provided have been altered such as not to violate the terms of service from their source. Therefore, one should not expect this data to replicate any of the paper results concerning returns, nor should it replicate standard results.

## Running the Code

1. install the required packages `pip -install -r requirements.txt`

2. create the necessary directions `setup_directories.sh`

3. generate the masked data `cd src & python generate_masked_data.py`

4. run the imputations `cd src & python run_data_imputations.py`

5. run the desired notebook for the particular results in question, ensure to run the first cell of the notebook to import the required modules and load the required data


## Paper Results and Their Locations

Main Text
1. Figure 1: Missing Values over Time
2. Figure 2: Missing Observations by Characteristic
3. Figure 3: Missing Observations by Characteristic Quintiles
4. Table 1: Logistic Regressions Explaining Missingess `run_section_2_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_2.py#L132) `images-pdfs/section2/MissingLogitRegressions.tex`
5. Figure 4: Autocorrelation of Characteristic Ranks
6. Figure 5: Heatmap of Pairwise Correlation
7. Figure 6: Joint Distribution of Missing Patterns `run_section_3_plots.ipynb`  code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_3.py#L42) result written to `images-pdfs/section3/missing--20180331.pdf`
8. Figure 7: Eigenvalues of Σ `run_section_4_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_4.py#L63)  result written to `images-pdfs/section4/figure_2_avg_cov_ev.pdf`
9. Figure 8: Number of Factors and Regularization `run_section_4_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_4.py#L87)  result written to `images-pdfs/section4/[number_of_factors_determination_xs-MAR0-True.pdf, number_of_factors_determination_xs-MAR0_0001-True.pdf, number_of_factors_determination_xs-MAR0_001-True.pdf,  number_of_factors_determination_xs-MAR0_01-True.pdf, number_of_factors_determination_xs-logit0-True.pdf, number_of_factors_determination_xs-logit0_0001-True.pdf, number_of_factors_determination_xs-logit0_001-True.pdf, number_of_factors_determination_xs-logit0_01-True.pdf, number_of_factors_determination_xs-prob_block0-True.pdf, number_of_factors_determination_xs-prob_block0_0001-True.pdf, number_of_factors_determination_xs-prob_block0_001-True.pdf,  number_of_factors_determination_xs-prob_block0_01-True.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-MAR0-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-MAR0_0001-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-MAR0_001-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-MAR0_01-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-logit0-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-logit0_0001-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-logit0_001-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-logit0_01-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-prob_block0-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-prob_block0_0001-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-prob_block0_001-True-incremental.pdf, metrics_by_char_vol_sort-number_of_factors_determination_xs-prob_block0_01-True-incremental.pdf]`
11. Figure 9: Optimal Regularization `run_section_4_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_4.py#L205)  result written to `images-pdfs/section4/`
12. Table 3: Imputation Error for Different Imputation Methods `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L240) and [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L331)  result written to `images-pdfs/section5/[AggregateImputationErrorsFullDataset.tex, AggregateImputationR2FullDataset.tex]`
13. Table 4: Imputation Error for Extreme Characteristic Quintiles `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L786) result written to `images-pdfs/section5/ImputationErrorsByCharQuintileFullDS.tex`
14. Figure 10: Illustrative Model-Implied and Imputed Time-Series `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L23) result written to `images-pdfs/section5/[HAS-one-year-mask-AT.pdf, HAS-one-year-mask-ME.pdf, HAS-one-year-mask-Q.pdf, HAS-one-year-mask-VAR.pdf, MSFT-one-year-mask-AT.pdf, MSFT-one-year-mask-ME.pdf, MSFT-one-year-mask-Q.pdf, MSFT-one-year-mask-VAR.pdf]`
15. Table 5: Imputation Error for Types of Missingness `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L689) result written to `images-pdfs/section5/[ImputationErrorsByMissingTypeEnd.tex, ImputationErrorsByMissingTypeMiddle.tex, ImputationErrorsByMissingTypeStart.tex]`
16. Figure 11: Imputation Error for Individual Characteristics `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L586) result written to `images-pdfs/section5/[metrics_by_char_vol_sort-table_1_in_sample.pdf, metrics_by_char_vol_sort-table_1_out_of_sample_MAR.pdf, metrics_by_char_vol_sort-table_1_out_of_sample_block.pdf, metrics_by_char_vol_sort-table_1_out_of_sample_logit.pdf]`
17. Figure 12: Information Used for Imputation `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L969) result written to `images-pdfs/section5/InfoUsedForImputationBW-bw_beta_weights.pdf`
18. Table 6: Imputation Error for Alternative Methods `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L1047) result written to `images-pdfs/section5/`
19. Figure 13: Market Premium Conditional on Observing a Firm Characteristic `run_section_6_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L26) result written to `images-pdfs/section6/ls-missing-obs-ports.pdf`
20. Figure 14: Sharpe Ratios with IPCA Factors `run_section_6_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L441) result written to `images-pdfs/section6/[ipca_sharpes_in_sample.pdf, ipca_sharpes_outof_sample.pdf]`
21. Figure 15: Univariate Sorts with and without Missing Values `run_section_6_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L120) result written to `images-pdfs/section6/[portfolio-sorts-B2M-Mean_Return.pdf, portfolio-sorts-B2M-Percent_Used.pdf, portfolio-sorts-B2M-Sharpe_Ratio.pdf, portfolio-sorts-B2M-Volatility.pdf, portfolio-sorts-INV-Mean_Return.pdf, portfolio-sorts-INV-Percent_Used.pdf, portfolio-sorts-INV-Sharpe_Ratio.pdf, portfolio-sorts-INV-Volatility.pdf, portfolio-sorts-ME-Mean_Return.pdf, portfolio-sorts-ME-Percent_Used.pdf, portfolio-sorts-ME-Sharpe_Ratio.pdf, portfolio-sorts-ME-Volatility.pdf, portfolio-sorts-OP-Mean_Return.pdf, portfolio-sorts-OP-Percent_Used.pdf, portfolio-sorts-OP-Sharpe_Ratio.pdf, portfolio-sorts-OP-Volatility.pdf]`
22. Figure 16: Imputation Bias in Pure-Play Mimicking Portfolios `run_section_6_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L257) result written to `images-pdfs/section6/[masked-factor_regression-pure_play-bw-xsmed-mean-abs-error.pdf, masked-factor_regression-pure_play-bw-xsmed-corr.pdf]`
23. Figure 17: Characteristic Mimicking Factor Portfolios `run_section_6_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L257) result written to `images-pdfs/section6/[masked-factor_regression-pure_play-B2M-bw-xsmed.pdf, masked-factor_regression-pure_play-INV-bw-xsmed.pdf, masked-factor_regression-pure_play-ME-bw-xsmed.pdf, masked-factor_regression-pure_play-S2P-bw-xsmed.pdf]`
24. Table A.1: Imputation Error for Alternative Implementations `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/appendix.py#L471) result written to `images-pdfs/appendix/[]`
25. Simulations
    - Figure A.1: Errors with Missing-Completely-at-Random `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/appendix.py#L21) result written to `images-pdfs/appendix/`
    - Figure A.2: Imputation Errors with Missing-Conditionally-at-Random `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/appendix.py#L21) result written to `images-pdfs/appendix/`
26. Table C.2: Missing by Characteristic Quintiles `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_2.py#L468) result written to `images-pdfs/section2/`
27. Table C.3: Lengths of Missing Blocks `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_2.py#L445) result written to `images-pdfs/section2/`
28. Figure D.1: Missing Observations over Time By Characteristics `run_appendix_plots.ipynb` code located [here]()
29. Figure D.2: Missing Observations by Characteristic Pooled by Stocks `run_appendix_plots.ipynb` code located [here]()
30. Figure D.3: Heatmap of Pairwise Correlation from 1967–1976 `run_appendix_plots.ipynb` code located [here]()
31. Figure D.4: Standard Deviation of Characteristic Ranks `run_appendix_plots.ipynb` code located [here]()
32. Figure D.5: Generalized Correlation of Global and Local Factor Weights `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_4.py#L308) result written to `images-pdfs/section4/`
33. Figure D.6: Composition of Proxy Factors by Characteristic Categories `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/appendix.py#L356) result written to `images-pdfs/appendix/[]`
34. Figure D.8: Global and Local Imputation for Individual Characteristics `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L586) result written to `images-pdfs/section5/metrics_by_char_vol_sort-table_2_out_of_sample_block.pdf`
35. Figure D.9: Top and Bottom Deciles with and without Missing Values `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L167) result written to `images-pdfs/section6/[hl-portfolios-Intangibles,TradingFrictions,Other-MeanReturn.pdf, hl-portfolios-Intangibles,TradingFrictions,Other-SharpeRatio.pdf, hl-portfolios-Investment,Profitability-MeanReturn.pdf, hl-portfolios-Investment,Profitability-SharpeRatio.pdf, hl-portfolios-PastReturns,Value-MeanReturn.pdf, hl-portfolios-PastReturns,Value-SharpeRatio.pdf]`
36. Figure D.10: Sharpe Ratios with Non-parametric IPCA Factors `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/appendix.py#L56) result written to `images-pdfs/appendix/[]`




