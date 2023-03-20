import pandas as pd
from utilities import analytical_normal_measures

'''
 Extract shares prices of our Portfolio
'''
index_df = pd.read_csv('_indexes.csv', index_col=0)
stocks = ['Adidas', 'Allianz', 'Munich Re', "L'Oréal"]
ticker = [index_df[index_df['Name'] == stock].Ticker.values.tolist()[0] for stock in stocks]
euro_stock50_df = pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
euro_stock50_df.fillna(axis=0, method='ffill', inplace=True)

stocks_df = euro_stock50_df[ticker].copy()

'''
Setting parameters
'''
alpha = 0.95                        # significant level
weights = [0.25, 0.25, 0.25, 0.25]  # we assume equally weighted
delta = 1                           # delta time lag
notional = 1e6                      # 1 MIO of notional

'''
Compute the Var and ES
'''
shares = stocks_df.loc["2013-01-02":"2019-03-19"].copy()
returns = shares / shares.shift(delta)
returns.drop(index=returns.index[0], axis=0, inplace=True)  # Drop first row

var, es = analytical_normal_measures(alpha, weights, returns)
final_var = var * notional
final_es = es * notional
