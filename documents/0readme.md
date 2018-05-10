# data
1. read_src
2. read_raw
3. load_data


# project structure  
1. First layer:base function
2. Second layer:based on First layer
3. application layer call the second layer function



# recheck
1. If we use filtered data, the max and min in summary statistics will change significantly
2. 
3. The benchmark adjusted tvalues seem to be messy.In details,
    For beta,D_1M,D_3M is positive,the others are negative
    For size,all of the signs are negative.
    For value,positive
    For mom,most of the t excess return is insignificant,and three of the ff3 tvalues are positive.All
        the others are negative
    For reversal,all of them are significant negatively     
    For liquidity,roll2 and amihud is positive,the others are negative.
    For skewness,1_D,3_D,6_D is positive for ff3,ff5,ffc
        For ff3,ffc,ff5,1_D,3_D,6_D is positive
        For capm,hxz,only 1_D is positive
        For all the other scenerys,the value is negative
    For idio,all of them are negative,except some of them are not significant for t excess return 
        The others are significant
       
    For skewness,it shows the same pattern as beta
        By the way,there are some abnormal values in the first stage of fm regression
            
    The benchmark models blow up all the magnitude of tvalue.
    


# Time relationship
For predictors with accounting data updated annually
(e.g.,earnings-to-price,accruals),variables constructed using data from 
year t are used to forecast returns for 12 months beginning in July
of year t+1,unless otherwise noted.For predictors that are updated on
a monthly interval (e.g.,momentum,long-term reversal),variables constructed
in month t are used to forecast returns in month t+1,unless otherwise noted.





# steps
1. use Dataset to construct stacked df containing all the needed data
2. send the df to Univariate or Bivariate

# data
1. read the documents of GTA about the calculation of the indicators.What's more,the paper it mentions is great guidence for further research.
2. At time t,we calculate the indicators and use this indicators to get sorted-portfolios
    then use these portfolio to calculate the mimicking portfolio returns(long-short factors) in time t+1.So the
    time corresponding to indicators are t,the time for factors are time t+1.
3. Steps:
    1. din.py
    2. din_filter_outliers.py
    3. calculate_indicators  '.py's
    3. indicators_filter_outliers.py

# functions to add
1. show those significant result with bold characters as table A5 of Asness, Clifford S., Andrea Frazzini, and Lasse Heje Pedersen. “Quality Minus Junk.” SSRN Scholarly Paper. Rochester, NY: Social Science Research Network, June 5, 2017. https://papers.ssrn.com/abstract=2312432.
2. D. The risk of quality stocks(Figure 4) in chapter 4 of Asness, Clifford S., Andrea Frazzini, and Lasse Heje Pedersen. “Quality Minus Junk.” SSRN Scholarly Paper. Rochester, NY: Social Science Research Network, June 5, 2017. https://papers.ssrn.com/abstract=2312432.
3. table 5 of mispricing factors

# TODO:
0. multiprocessing
1. use line_profiler to do line-by-line profiling and speedup the project.
1. change the frequency,weekly
2. principal component analysis
3. for any model involved 'BP' or 'EP',we have to delete those stocks with negative value in these fields.
    refer to page 6 of 20170313-华泰证券-华泰证券五因子模型A股实证研究：Fama-French 五因子模型实证.pdf
4. VIF>2.5? multicollearity


0. regime change (subsample,market status)
0. move all the related function into the current project
1. handle the TODOs one by one
1. Calculate indicators as other papers
2. detect anomalies in A-share
4. read my notes and other papers about cross section
5. D:\app\python27\zht\researchTopics\assetPricing
5. refer to D:\app\python27\zht\researchTopics\assetPricing\regressionModels.py
6. refer to D:\app\python27\zht\researchTopics\assetPricing\multi_getFactorPortId.py for multiprocess method
7. refer to D:\app\python27\zht\researchTopics\assetPricing\tools.py for visualization
8. After finishing this new project refer to the former projects(D:\app\python27\zht\researchTopics\crossSection) to refine my frame
9. add two-procedure as Maio and Philip, “Economic Activity and Momentum Profits.” mentions in page 5.refer to xiong's high moments
10. panel B of table1 in Stambaugh, Yu, and Yuan, “Arbitrage Asymmetry and the Idiosyncratic Volatility Puzzle.”,use
    this table to describle the possible asymmetry distribution of different characteristics.
11. figure 2 of Stambaugh, Yu, and Yuan, “Arbitrage Asymmetry and the Idiosyncratic Volatility Puzzle.”
    is an alternative for the part A in panel B of table 9.6 in Bali.
12. add the function to detect the asymmetrical effects of long and short leg,or market condition or sentiment
13. add GRS as chapter 5.3 of Pan, Tang, and Xu, “Speculative Trading and Stock Returns.”
14. add a sample matching method to calculate the spread returns as stated in the notes of Xie and Mo, “Index Futures Trading and Stock Market Volatility in China.”
15. add a function to compare the result of total sample with that of subsample,such as page
    221 of Ellul and Panayides, “Do Financial Analysts Restrain Insiders’ Informational Advantage?”
16. dissect the correlation for every year as page 10 of Pan, Tang, and Xu, “Speculative Trading and Stock Returns.”
17. use dummy variable to test the asymmetrical effect of limits of arbitrage on stock returns as table 7 of Gu, Kang, and Xu, “Limits of Arbitrage and Idiosyncratic Volatility.”
18. summary for each year as table 1 of Cakici, Chan, and Topyan, “Cross-Sectional Stock Return Predictability in China.”
19. analyse the annual return as figure 1 of Cakici, Chan, and Topyan, “Cross-Sectional Stock Return Predictability in China.”
20. test the factors worldwide as Han, Hu, and Lesmond, “Liquidity Biases and the Pricing of Cross-Sectional Idiosyncratic Volatility around the World.”
21. use regression residual approach to 'orthogonalize' highly related variables as Han, Hu, and Lesmond, “Liquidity Biases and the Pricing of Cross-Sectional Idiosyncratic Volatility around the World.”
22. Read the Notes in zotero to find new methods and inspirations.For example Bekaert, Hodrick, and Zhang, “Aggregate Idiosyncratic Volatility.” may give you the following inspirations:
    1. international correlation
    2. international evidence
    3. regime switching
    4. determinants of the factor dynamics
    5. model selection or model reduction techniques (How to gauge the relative importance of the various variables)
    6. business cycle
23. add regression figure as [this one](https://github.com/nakulnayyar/FF3Factor/blob/master/FamaFrench3Factor.ipynb)
24. add a trading strategy for the new factors as page 9 of Lao, Tian, and Zhao, “Will Order Imbalances Predict Stock Returns in Extreme Market Situations? Evidence from China.”
25. add quantile regression as the papers shown in zotero by searching quantile.
    draw fig as fig3 and fig4 in  Xue and Zhang, “Stock Return Autocorrelations and Predictability in the Chinese Stock Market-Evidence from Threshold Quantile Autoregressive Models.”
26. how about analyse industry-by-industry?
27. stocks with high analyst-coverage and low analyst-coverage
28. stocks with high trading volume and low trading volume
29. stocks with low institutional ownership and high....
30. in the releasing window of policy or not? refer to 田利辉, 王冠英, and 张伟, “三因素模型定价.”
30. small stocks and high stocks
31. Any indicators (or anomalies,whatever) can be used to create subsamples or subperiods
32. investor sentiment,...findout all possible asymmetrical controling variable in zotero.
33. FF sort stocks on size (univariate or bivariate) to test the predictability of factors on stock returns.We can sort market status,
    high or low institutional ownership (univariate or bivariate) to test the predictability of factors (indicators,whatever) on factors
    refer to
    1. Cheema and Nartea, “Momentum Returns, Market States, and Market Dynamics.”
    2. Li and Galvani, “Market States, Sentiment, and Momentum in the Corporate Bond Market.”
    3. NVIX (or VIX) Manela and Moreira, “News Implied Volatility and Disaster Concerns.”
    
34. compare with the new method in Safdar and Yan, “Information Risk, Stock Returns, and Asset Pricing: Evidence from China.”

35. 流通A股占总股本比例 statistic,thresh

# The difference between assetPricing1 and assetPricing2
1. the date in assetPricing2 seem to be calender date rather than buisiness
date as assetPricing1.
2. when calculate beta ,AP2 use eret but AP1 use ret.In the book of Bali,eret is used
3.


# factor zoo
1. Winner-minus-loser (WML) portfolios, following Carhart (1997) and Fama and French (2012), we also construct Winner-minus-Loser (WML) portfolios. 
