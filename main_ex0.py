import pandas as pd
from utilities import analytical_normal_measures

# Extract shares prices of our Portfolio

index_df = pd.read_csv('_indexes.csv', index_col=0)
stocks = ['Adidas', 'Allianz', 'Munich Re', "L'Oréal"]
ticker = [index_df[index_df['Name'] == stock].Ticker.values.tolist()[0] for stock in stocks]
euro_stock50_df = pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
euro_stock50_df.fillna(axis=0, method='ffill', inplace=True)

stocks_df = euro_stock50_df[ticker].copy()      # dataset of the stocks we are interested in

# Setting parameters

alpha = 0.95                        # significant level
weights = [0.25, 0.25, 0.25, 0.25]  # we assume equally weighted
delta = 1                           # delta time lag
notional = 1e7                      # 1 MIO of notional

# Compute the Var and ES

shares = stocks_df.loc["2016-03-18":"2019-03-20"].copy()        # Consider only the shares of the last 3y
returns = shares / shares.shift(delta)                          # Compute the return for each company shares
returns.drop(index=returns.index[0], axis=0, inplace=True)      # Drop first row
portfolio_value = notional

var, es = analytical_normal_measures(alpha, weights, delta, portfolio_value, returns)
print("The Analytical Normal Measure results for the portfolio are:")
print("Value at Risk (VaR): {:.5f}".format(var))
print("Expected Shortfall (ES): {:.5f}".format(es))
print("   ")
