import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TradingProfessional:
    def __init__(self, data):
        """
        Initialize the TradingProfessional with historical stock data.
        The data should be a pandas DataFrame containing a 'Close' column.
        """
        self.data = data.copy()
    
    def calculate_bollinger_bands(self, window=20, num_std=2):
        """
        Calculate Bollinger Bands:
         - Middle Band: 20-day simple moving average (SMA)
         - Upper Band: SMA + (2 x standard deviation)
         - Lower Band: SMA - (2 x standard deviation)
        """
        self.data['SMA'] = self.data['Close'].rolling(window=window).mean()
        self.data['Rolling_STD'] = self.data['Close'].rolling(window=window).std()
        self.data['Upper_Band'] = self.data['SMA'] + (self.data['Rolling_STD'] * num_std)
        self.data['Lower_Band'] = self.data['SMA'] - (self.data['Rolling_STD'] * num_std)
        return self.data

    def generate_signals(self):
        """
        Generate trading signals:
        - Buy (signal = 1) when price crosses above the lower band.
        - Sell (signal = -1) when price crosses below the upper band.
        - Hold (signal = 0) when price is between the bands.
        """
        self.data['Signal'] = 0
        
        # Ensure we have 1D arrays by squeezing the Series
        close = self.data['Close'].squeeze().values
        lower_band = self.data['Lower_Band'].squeeze().values
        upper_band = self.data['Upper_Band'].squeeze().values

        # Shifted values for yesterday's data
        close_shift = self.data['Close'].squeeze().shift(1).values
        lower_band_shift = self.data['Lower_Band'].squeeze().shift(1).values
        upper_band_shift = self.data['Upper_Band'].squeeze().shift(1).values
        
        # Buy signal: Yesterday's price was below the lower band and today's price is above it.
        buy_signals = (close_shift < lower_band_shift) & (close > lower_band)
        
        # Sell signal: Yesterday's price was above the upper band and today's price is below it.
        sell_signals = (close_shift > upper_band_shift) & (close < upper_band)
        
        # Convert boolean arrays into Series with the same index as self.data
        buy_signals_series = pd.Series(buy_signals, index=self.data.index)
        sell_signals_series = pd.Series(sell_signals, index=self.data.index)
        
        # Assign signals back to the DataFrame
        self.data.loc[buy_signals_series, 'Signal'] = 1
        self.data.loc[sell_signals_series, 'Signal'] = -1
        
        return self.data


    def backtest_strategy(self):
        """
        Backtest the trading strategy:
         - Use the generated signals to simulate trades.
         - Calculate daily market returns and strategy returns.
         - Strategy return on day t is defined as the previous day's signal 
           multiplied by the market return on day t.
        """
        # Calculate market returns based on the adjusted close price (using 'Close' here)
        self.data['Market_Return'] = self.data['Close'].pct_change()
        
        # Shift the signal to simulate taking action the next day after signal generation
        self.data['Strategy_Return'] = self.data['Signal'].shift(1) * self.data['Market_Return']
        
        # Calculate cumulative returns for the strategy
        self.data['Cumulative_Strategy'] = (1 + self.data['Strategy_Return']).cumprod()
        self.data['Cumulative_Market'] = (1 + self.data['Market_Return']).cumprod()
        
        return self.data

    def visualize_strategy(self):
        """
        Show the Bollinger Bands, trading signals, and the stock's price.
        Buy are green upward triangles, and sell are red downward triangles.
        """
        plt.figure(figsize=(14, 7))
        
        # Close price and Bollinger Bands
        plt.plot(self.data.index, self.data['Close'], label='Close', color='blue')
        plt.plot(self.data.index, self.data['SMA'], label='20-Day SMA', color='black', linestyle='--')
        plt.plot(self.data.index, self.data['Upper_Band'], label='Upper Band', color='orange', linestyle='--')
        plt.plot(self.data.index, self.data['Lower_Band'], label='Lower Band', color='purple', linestyle='--')
        
        # Buy signals 
        buy_signals = self.data[self.data['Signal'] == 1]
        plt.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='green', label='Buy Signal')
        
        # Sell signals
        sell_signals = self.data[self.data['Signal'] == -1]
        plt.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='red', label='Sell Signal')
        
        plt.title('Bollinger Bands and Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()


# Test
if __name__ == "__main__":
    ticker = 'VRT'
    # Download data
    data = yf.download(ticker, period='1y')
    
    trader = TradingProfessional(data)
    trader.calculate_bollinger_bands()
    trader.generate_signals()
    trader.backtest_strategy()
    
    final_strategy_return = trader.data['Cumulative_Strategy'].iloc[-1]
    final_market_return = trader.data['Cumulative_Market'].iloc[-1]
    print(f"Final Cumulative Strategy Return: {final_strategy_return:.2f}")
    print(f"Final Cumulative Market Return: {final_market_return:.2f}")
    
    trader.visualize_strategy()
