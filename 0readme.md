
# steps
1. use Dataset to construct stacked df containing all the needed data
2. send the df to Univariate or Bivariate

# data
1. read the documents of GTA about the calculation of the indicators.What's more,the paper it mentions is great guidence for further research.
2. At time t,we calculate the indicators and use this indicators to get sorted-portfolios
    then use these portfolio to calculate the mimicking portfolio returns(long-short factors) in time t+1.So the
    time corresponding to indicators are t,the time for factors are time t+1.

# functions to add
1. show those significant result with bold characters as table A5 of Asness, Clifford S., Andrea Frazzini, and Lasse Heje Pedersen. “Quality Minus Junk.” SSRN Scholarly Paper. Rochester, NY: Social Science Research Network, June 5, 2017. https://papers.ssrn.com/abstract=2312432.
2. D. The risk of quality stocks(Figure 4) in chapter 4 of Asness, Clifford S., Andrea Frazzini, and Lasse Heje Pedersen. “Quality Minus Junk.” SSRN Scholarly Paper. Rochester, NY: Social Science Research Network, June 5, 2017. https://papers.ssrn.com/abstract=2312432.
3. table 5 of mispricing factors

# TODO:
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



# The difference between assetPricing1 and assetPricing2
1. the date in assetPricing2 seem to be calender date rather than buisiness
date as assetPricing1.
2. when calculate beta ,AP2 use eret but AP1 use ret.In the book of Bali,eret is used
3.



