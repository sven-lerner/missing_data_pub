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


1. Figure 1: Missing Values over Time
2. Figure 2: Missing Observations by Characteristic
3. Figure 3: Missing Observations by Characteristic Quintiles
4. Table 1: Logistic Regressions Explaining Missingess
5. Figure 4: Autocorrelation of Characteristic Ranks
6. Figure 5: Heatmap of Pairwise Correlation
7. Figure 6: Joint Distribution of Missing Patterns
8. Figure 7: Eigenvalues of Σ
9. Figure 8: Number of Factors and Regularization
10. Figure 9: Optimal Regularization
11. Table 3: Imputation Error for Different Imputation Methods
12. Table 4: Imputation Error for Extreme Characteristic Quintiles
13. Figure 10: Illustrative Model-Implied and Imputed Time-Series
14. Table 5: Imputation Error for Types of Missingness
15. Figure 11: Imputation Error for Individual Characteristics
16. Figure 12: Information Used for Imputation
17. Table 6: Imputation Error for Alternative Methods
18. Figure 13: Market Premium Conditional on Observing a Firm Characteristic
19. Figure 14: Sharpe Ratios with IPCA Factors
20. Figure 15: Univariate Sorts with and without Missing Values
21. Figure 16: Imputation Bias in Pure-Play Mimicking Portfolios
22. Figure 17: Characteristic Mimicking Factor Portfolios
23. Table A.1: Imputation Error for Alternative Implementations
24. Simulations
    - Figure A.1: Errors with Missing-Completely-at-Random
    - Figure A.2: Imputation Errors with Missing-Conditionally-at-Random
25. Table C.2: Missing by Characteristic Quintiles
26. Table C.3: Lengths of Missing Blocks
27. Figure D.1: Missing Observations over Time By Characteristics
28. Figure D.2: Missing Observations by Characteristic Pooled by Stocks
29. Figure D.3: Heatmap of Pairwise Correlation from 1967–1976 ATO ATO OL OL CTO CTO PROF PROF D2A D2A NOA NOA AC AC OA OA R2_1 R2_1 SUV SUV
30. Figure D.4: Standard Deviation of Characteristic Ranks
31. Figure D.5: Generalized Correlation of Global and Local Factor Weights
32. Figure D.6: Composition of Proxy Factors by Characteristic Categories
33. Figure D.8: Global and Local Imputation for Individual Characteristics
34. Figure D.9: Top and Bottom Deciles with and without Missing Values
35. Figure D.10: Sharpe Ratios with Non-parametric IPCA Factors




