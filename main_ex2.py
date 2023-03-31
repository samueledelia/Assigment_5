import numpy as np
import utilities as ut
import datetime as dt
import pandas as pd

# Setting the Dataframes and parameters
index_df = pd.read_csv('_indexes.csv', index_col=0)
euro_stock50_df = pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
euro_stock50_df.fillna(axis=0, method='ffill', inplace=True)
euro_stock50_df.fillna(axis=0, method='bfill', inplace=True)

stocks_portfolio = ['Vonovia']
ticker_portfolio = [index_df[index_df['Name'] == stock].Ticker.values.tolist()[0] for stock in stocks_portfolio]
portfolio_df = euro_stock50_df[ticker_portfolio].copy()
portfolio_df = portfolio_df.loc["2021-01-31":"2023-01-31"].copy()

# Setting Parameters
stock_price = portfolio_df.iloc[-1]
notional = 25870000
n_shares = notional / stock_price
n_puts = n_shares
strike = 25
volatility = 0.154
dividend_yield = 0.031
alpha = 0.99
delta = 1
n_days_per_y = 365
risk_measure_time_interval_in_y = delta / n_days_per_y

# Compute the rate
discounts = [0.999462721181408, 0.995946245361081]  # disc from T0 t T0+10 e al 5 apr 2023
ttm = (dt.date(2023, 4, 5).toordinal() - dt.date(2023, 1, 31).toordinal()) / n_days_per_y
rate = -1 / ttm * np.log(discounts[1])

# Start computing
returns = portfolio_df / portfolio_df.shift(delta)
returns.drop(index=returns.index[0], axis=0, inplace=True)
logReturns = np.log(returns) * np.sqrt(10)
portValue = notional + n_puts * ut.putprice_BS(stock_price, strike, volatility, rate, dividend_yield, ttm)

var = ut.FullMonteCarloVaR(logReturns, n_shares, n_puts, stock_price, strike, rate, dividend_yield, volatility, ttm,
                           risk_measure_time_interval_in_y, alpha, n_days_per_y)
var_delta = ut.DeltaNormalVaR(logReturns, n_shares, n_puts, stock_price, strike, rate, dividend_yield, volatility, ttm,
                              risk_measure_time_interval_in_y, alpha, n_days_per_y)
var_deltagamma = ut.DeltaGammaNormalVaR(logReturns, n_shares, n_puts, stock_price, strike, rate, dividend_yield,
                                        volatility, ttm, risk_measure_time_interval_in_y, alpha, n_days_per_y)
print("The values of the VaR obtained are:")
print("VaR Full MC: {:.5f}".format(var))
print("VaR Delta normal: {:.5f}".format(var_delta))
print("VaR Delta-Gamma normal: {:.5f}".format(var_deltagamma))
print("   ")
