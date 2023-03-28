import numpy as np
import pandas as pd

# Setting the Dataframes and parameters

index_df = pd.read_csv('_indexes.csv', index_col=0)
euro_stock50_df = pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
euro_stock50_df.fillna(axis=0, method='ffill', inplace=True)
euro_stock50_df.fillna(axis=0, method='bfill', inplace=True)
delta = 10

stocks_portfolio = ['Vonovia']
ticker_portfolio = [index_df[index_df['Name'] == stock].Ticker.values.tolist()[0] for stock in stocks_portfolio]
portfolio_df = euro_stock50_df[ticker_portfolio].copy()
portfolio_df = portfolio_df.loc["2019-04-05":"2023-01-31"]

# Setting Parameters

notional = 25870000
n_shares = notional/portfolio_df.iloc[-1]
