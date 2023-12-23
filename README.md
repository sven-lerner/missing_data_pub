# Result Replication Code

## Notes about Data

Note, there are two differences between the data-set we have provided and the data-set used in the paper
1. The data-set we have provided is a truncated version of the data-set in the paper. This is because the full data-set is around 20 GB, and running all the results requires making multiple copies of the data-set, which is very time and space consuming
2. The returns in the data-set we have provided have been altered such as not to violate the terms of service from their source. Therefore, one should not expect this data to replicate any of the paper results concerning returns, nor should it replicate standard results.

## Running the Code

1. install the required packages `pip -install -r requirements.txt`

2. generate the masked data `cd src & python generate_masked_data.py`

3. run the imputations `cd src & python run_data_imputations.py`

4. run the desired notebook for the particular results in question


## Paper Results and Their Locations

Main Text
1. Figure 1: Missing Values over Time
2. Figure 2: Missing Observations by Characteristic
3. Figure 3: Missing Observations by Characteristic Quintiles
4. Table 1: Logistic Regressions Explaining Missingess `run_section_2_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_2.py#L132)
5. Figure 4: Autocorrelation of Characteristic Ranks
6. Figure 5: Heatmap of Pairwise Correlation
7. Figure 6: Joint Distribution of Missing Patterns `run_section_3_plots.ipynb`  code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_3.py#L42)
8. Figure 7: Eigenvalues of Σ `run_section_4_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_4.py#L63)
9. Figure 8: Number of Factors and Regularization `run_section_4_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_4.py#L87)
10. Figure 9: Optimal Regularization `run_section_4_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_4.py#L205)
11. Table 3: Imputation Error for Different Imputation Methods `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L240) and [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L331)
12. Table 4: Imputation Error for Extreme Characteristic Quintiles `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L786)
13. Figure 10: Illustrative Model-Implied and Imputed Time-Series `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L23)
14. Table 5: Imputation Error for Types of Missingness `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L689)
15. Figure 11: Imputation Error for Individual Characteristics `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L586)
16. Figure 12: Information Used for Imputation `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L969)
17. Table 6: Imputation Error for Alternative Methods `run_section_5_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L1047)
18. Figure 13: Market Premium Conditional on Observing a Firm Characteristic `run_section_6_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L26)
19. Figure 14: Sharpe Ratios with IPCA Factors `run_section_6_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L441)
20. Figure 15: Univariate Sorts with and without Missing Values `run_section_6_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L120)
21. Figure 16: Imputation Bias in Pure-Play Mimicking Portfolios `run_section_6_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L257)
22. Figure 17: Characteristic Mimicking Factor Portfolios `run_section_6_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L257)
24. Table A.1: Imputation Error for Alternative Implementations `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/appendix.py#L471)
25. Simulations
    - Figure A.1: Errors with Missing-Completely-at-Random `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/appendix.py#L21)
    - Figure A.2: Imputation Errors with Missing-Conditionally-at-Random `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/appendix.py#L21)
26. Table C.2: Missing by Characteristic Quintiles `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_2.py#L468)
27. Table C.3: Lengths of Missing Blocks `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_2.py#L445)
28. Figure D.1: Missing Observations over Time By Characteristics `run_appendix_plots.ipynb` code located [here]()
29. Figure D.2: Missing Observations by Characteristic Pooled by Stocks `run_appendix_plots.ipynb` code located [here]()
30. Figure D.3: Heatmap of Pairwise Correlation from 1967–1976 `run_appendix_plots.ipynb` code located [here]()
31. Figure D.4: Standard Deviation of Characteristic Ranks `run_appendix_plots.ipynb` code located [here]()
32. Figure D.5: Generalized Correlation of Global and Local Factor Weights `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_4.py#L308)
33. Figure D.6: Composition of Proxy Factors by Characteristic Categories `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/appendix.py#L356)
34. Figure D.8: Global and Local Imputation for Individual Characteristics `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_5.py#L586)
35. Figure D.9: Top and Bottom Deciles with and without Missing Values `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/section_6.py#L167)
36. Figure D.10: Sharpe Ratios with Non-parametric IPCA Factors `run_appendix_plots.ipynb` code located [here](https://github.com/sven-lerner/missing_data_pub/blob/main/src/plots_and_tables/appendix.py#L56)




