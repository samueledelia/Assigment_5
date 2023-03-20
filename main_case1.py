import pandas as pd

'''
 Setting the Dataframes
'''
index_df = pd.read_csv('_indexes.csv', index_col=0)
euro_stock50_df = pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
euro_stock50_df.fillna(axis=0, method='ffill', inplace=True)

'''
Point a: Compute daily VaR and ES with a 3y estimation using the dataset provided via a Historical Simulation
         approach and a Bootstrap method with 200 simulations
'''
stocks_portfolio1 = ['Total', 'Danone', 'Sanofi', "Volkswagen"]
ticker_portfolio1 = [index_df[index_df['Name'] == stock].Ticker.values.tolist()[0] for stock in stocks_portfolio1]
portfolio1_df = euro_stock50_df[ticker_portfolio1].copy()



