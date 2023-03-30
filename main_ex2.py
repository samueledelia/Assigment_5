import numpy as np
import pandas as pd
import datetime as dt
from utilities import full_montecarlo_var

# Setting the Dataframes and parameters

index_df = pd.read_csv('_indexes.csv', index_col=0)
euro_stock50_df = pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
euro_stock50_df.fillna(axis=0, method='ffill', inplace=True)
euro_stock50_df.fillna(axis=0, method='bfill', inplace=True)


stocks_portfolio = ['Vonovia']
ticker_portfolio = [index_df[index_df['Name'] == stock].Ticker.values.tolist()[0] for stock in stocks_portfolio]
portfolio_df = euro_stock50_df[ticker_portfolio].copy()
portfolio_df = portfolio_df.loc["2021-01-31":"2023-01-31"]

# Setting Parameters

stock_price = portfolio_df.iloc[-1]
notional = 25870000
n_shares = notional/stock_price
n_puts = n_shares
strike = 25
# rate =
volatility = 0.154
dividend_yield = 0.031
alpha = 0.99
delta = 10
n_days_per_y = 365

# Compute the rate

discounts = [0.999462721181408, 0.995946245361081]                 # disc from T0 t T0+10 e al 5 apr 2023
ttm = (dt.date(2023, 4, 5).toordinal()-dt.date(2023, 1, 31).toordinal())/365
rate = -1/ttm * np.log(discounts[1])

# Start computing
returns = portfolio_df / portfolio_df.shift(delta)
index = np.arange(delta, len(portfolio_df), delta)
returns = returns.iloc[index]

# full_montecarlo_var(x, n_shares, n_puts, stock_price, strike, rate, dividend, volatility, ttm, delta, alpha, n_days_per_y)