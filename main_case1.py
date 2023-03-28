import pandas as pd
import numpy as np
from utilities import hs_measurements, whs_measurements, calculate_portfolio_metrics, calculate_portfolio_metrics_whs, plausibility_check, bootstrap_statistical

# Setting the Dataframes and parameters

index_df = pd.read_csv('_indexes.csv', index_col=0)
euro_stock50_df = pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
euro_stock50_df.fillna(axis=0, method='ffill', inplace=True)
euro_stock50_df.fillna(axis=0, method='bfill', inplace=True)
delta = 1

'''
Point a: Compute daily VaR and ES with a 3y estimation using the dataset provided via a Historical Simulation
         approach and a Bootstrap method with 200 simulations
'''
stocks_portfolio1 = ['TotalEnergies', 'Danone', 'Sanofi', 'Volkswagen Group']
ticker_portfolio1 = [index_df[index_df['Name'] == stock].Ticker.values.tolist()[0] for stock in stocks_portfolio1]
portfolio1_df = euro_stock50_df[ticker_portfolio1].copy()
portfolio1_df = portfolio1_df.loc["2016-03-18":"2019-03-20"]

# Setting parameters

alpha = 0.99                            # significant level
n_shares = [25e3, 20e3, 20e3, 10e3]     # n of shares for each company

# Compute the Var and ES with HS

weights_1_hs, returns_1_hs, value_portfolio_hs = calculate_portfolio_metrics(n_shares, portfolio1_df, delta)
var_hs, es_hs = hs_measurements(returns_1_hs, alpha, weights_1_hs, value_portfolio_hs)
var_check_hs = plausibility_check(returns_1_hs, weights_1_hs, alpha, delta, value_portfolio_hs)

print("The Historical Simulation results for the portfolio are:")
print("Value at Risk (VaR): {:.5f}".format(var_hs))
print("Expected Shortfall (ES): {:.5f}".format(es_hs))
print("Plausibility Check: {:.5f}".format(var_check_hs[0][0]))
print("   ")

# Compute Var and ES with Bootstrap

portfolio_size = np.array(portfolio1_df).shape
numberOfSamplesToBootstrap = 200

index = bootstrap_statistical(portfolio_size, numberOfSamplesToBootstrap)
returns_1_boot = returns_1_hs.iloc[index].copy()
weights_1_boot = weights_1_hs
value_portfolio_boot = value_portfolio_hs
var_boot, es_boot = hs_measurements(returns_1_boot, alpha, weights_1_boot, value_portfolio_boot)
var_check_boot = plausibility_check(returns_1_boot, weights_1_boot, alpha, delta, value_portfolio_boot)

print("The Bootstrap results for the portfolio are:")
print("Value at Risk (VaR): {:.5f}".format(var_boot))
print("Expected Shortfall (ES): {:.5f}".format(es_boot))
print("Plausibility Check: {:.5f}".format(var_check_boot[0][0]))
print("   ")


'''
Point b: Compute daily VaR and ES with a 3y estimation using the dataset provided via a Weighted Historical Simulation 
         approach with lambda = 0.97
'''

stocks_portfolio2 = ['Adidas', 'Airbus', 'BBVA', 'BMW', 'Schneider Electric']
ticker_portfolio2 = [index_df[index_df['Name'] == stock].Ticker.values.tolist()[0] for stock in stocks_portfolio2]
portfolio2_df = euro_stock50_df[ticker_portfolio2].copy()
portfolio2_df = portfolio2_df.loc["2016-03-18":"2019-03-20"]

# Setting parameters

lambda_portfolio2 = 0.97
n2 = len(stocks_portfolio2)                         # number of companies

value_portfolio_whs = value_portfolio_hs
weights_2, returns_2 = calculate_portfolio_metrics_whs(n2, portfolio2_df, delta)
var_whs, es_whs = whs_measurements(returns_2, alpha, weights_2, lambda_portfolio2, value_portfolio_whs)
var_check_whs = plausibility_check(returns_2, weights_2, alpha, delta, value_portfolio_whs)
print("The Weighted Historical Simulation results for the portfolio are:")
print("Value at Risk (VaR): {:.5f}".format(var_whs))
print("Expected Shortfall (ES): {:.5f}".format(es_whs))
print("Plausibility Check: {:.5f}".format(var_check_whs[0][0]))

'''
Point c: Compute 10 days VaR and ES with a 3y estimation using the dataset provided via a Gaussian parametric PCA 
         approach using the first n principal components, with the parameter n =1,..,6.
'''
'''
stocks_portfolio3 = np.linspace(0, 19, 20)
portfolio3_df = euro_stock50_df.iloc[:, stocks_portfolio3].copy()
portfolio3_df.drop("ADYEN.AS", axis=1, inplace=True)
yearly_covariance = np.cov(portfolio3_df)
print(yearly_covariance)
'''