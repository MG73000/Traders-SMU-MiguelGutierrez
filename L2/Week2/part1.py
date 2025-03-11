import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 10 Stocks
tickers = ['CRM', 'META', 'BKNG', 'BA', 'RIVN', 'LLY', 'TSLA', 'VRT', 'CVNA', 'ASML']

# Get data for one year of daily data
def fetch_data(ticker, period='1y'):
    data = yf.download(ticker, period=period)
    return data

# Calculate implied volatility
def calculate_implied_volatility(ticker):
    data = fetch_data(ticker)
    if data.empty:
        return np.nan  

    # Use the 'Close' column since auto_adjust=True applies the adjustment automatically
    data['Daily_Return'] = data['Close'].pct_change()
    
    # STDEV of daily returns and annualize the volatility
    daily_std = data['Daily_Return'].std()
    annual_volatility = daily_std * np.sqrt(252)
    
    return annual_volatility

# Implied volatility for each stock
volatility_dict = {}
for ticker in tickers:
    vol = calculate_implied_volatility(ticker)
    volatility_dict[ticker] = vol

volatility_df = pd.DataFrame(list(volatility_dict.items()), columns=['Ticker', 'Implied_Volatility'])
volatility_df.sort_values(by='Implied_Volatility', ascending=False, inplace=True)

print(volatility_df)

top_5 = volatility_df.head(5)
print("\nTop 5 most volatile stocks:")
print(top_5)
