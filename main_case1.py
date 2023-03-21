import pandas as pd
import numpy as np
from utilities import hs_measurements

# Setting the Dataframes and parameters

index_df = pd.read_csv('_indexes.csv', index_col=0)
euro_stock50_df = pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
euro_stock50_df.fillna(axis=0, method='ffill', inplace=True)
delta = 1

'''
Point a: Compute daily VaR and ES with a 3y estimation using the dataset provided via a Historical Simulation
         approach and a Bootstrap method with 200 simulations
'''
stocks_portfolio1 = ['TotalEnergies', 'Danone', 'Sanofi', 'Volkswagen Group']
ticker_portfolio1 = [index_df[index_df['Name'] == stock].Ticker.values.tolist()[0] for stock in stocks_portfolio1]
portfolio1_df = euro_stock50_df[ticker_portfolio1].copy()

# Setting parameters

alpha = 0.99                            # significant level
n_shares = [25e3, 20e3, 20e3, 10e3]     # n of shares for each company

# Compute the Var and ES with HS

weights_1 = np.multiply(n_shares, portfolio1_df.loc["2016-03-18":"2019-03-20"].copy())
weights_1.drop(index=weights_1.index[0], axis=0, inplace=True)      # Drop first row
shares_1 = portfolio1_df.loc["2016-03-18":"2019-03-20"].copy()    # Consider only the shares of the last 3y
returns_1 = shares_1 / shares_1.shift(delta)                          # Compute the return for each company shares
returns_1.drop(index=returns_1.index[0], axis=0, inplace=True)      # Drop first row

var_hs, es_hs = hs_measurements(returns_1, alpha, weights_1)

'''
Point b: Compute daily VaR and ES with a 3y estimation using the dataset provided via a Weighted Historical Simulation 
         approach with lambda = 0.97
'''

stocks_portfolio2 = ['Adidas', 'Airbus', 'BBVA', 'BMW', 'Schneider Electric']
ticker_portfolio2 = [index_df[index_df['Name'] == stock].Ticker.values.tolist()[0] for stock in stocks_portfolio2]
portfolio2_df = euro_stock50_df[ticker_portfolio2].copy()

# Setting parameters

lambda_portfolio2 = 0.97
n2 = len(stocks_portfolio2)         # number of companies
weights_2 = np.ones(n2)/n2          # equally weighted portfolio
weights_2.drop(index=weights_2.index[0], axis=0, inplace=True)      # Drop first row
shares_2 = portfolio2_df.loc["2016-03-18":"2019-03-20"].copy()    # Consider only the shares of the last 3y
returns_2 = shares_2 / shares_2.shift(delta)                          # Compute the return for each company shares
returns_2.drop(index=returns_2.index[0], axis=0, inplace=True)      # Drop first row

whs_measurements(returns, alpha, weights_2, lambda_portfolio)