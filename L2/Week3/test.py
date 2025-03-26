import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series, cutoff=0.05):
    """
    Perform the Augmented Dickey-Fuller test on a series and print the p-value.
    Returns True if the series is stationary (p < cutoff), False otherwise.
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f'ADF p-value for {series.name}: {p_value}')
    if p_value < cutoff:
        print(f"{series.name} is likely stationary (I(0)).\n")
    else:
        print(f"{series.name} is likely non-stationary (I(1)).\n")
    return p_value < cutoff

def main():
    # Define the date range
    start_date = "2020-01-01"
    end_date = "2022-12-31"

    # Download data for Home Depot (HD) and Lowe's (LOW)
    data = yf.download(['HD', 'LOW'], start=start_date, end=end_date)

    # Extract the 'Close' prices and rename the columns for clarity
    data = data['Close']
    HD = data['HD'].rename('HD')
    LOW = data['LOW'].rename('LOW')

    # Print the first few rows to verify
    print("Price Data (first 5 rows):")
    print(data.head(), "\n")

    # Plot the price series
    plt.figure(figsize=(10, 6))
    plt.plot(HD.index, HD.values, label='Home Depot (HD)')
    plt.plot(LOW.index, LOW.values, label="Lowe's (LOW)")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title("Price Series: Home Depot vs. Lowe's")
    plt.legend()
    plt.show()

    # Check stationarity of the raw series
    check_stationarity(HD)
    check_stationarity(LOW)

    # Perform linear regression: LOW = a + beta * HD + error
    HD_const = sm.add_constant(HD)  # add constant for intercept
    results_HD_LOW = sm.OLS(LOW, HD_const).fit()
    print(results_HD_LOW.summary())

    # Extract the beta coefficient for HD
    beta_HD = results_HD_LOW.params['HD']
    print("Estimated beta for HD:", beta_HD, "\n")

    # Compute cointegrating residuals (using the model's residuals)
    residuals = results_HD_LOW.resid
    residuals.name = "Cointegrating Residual (LOW - β*HD)"

    # Plot the residuals
    plt.figure(figsize=(10, 6))
    plt.plot(residuals.index, residuals.values, label=residuals.name)
    plt.xlabel("Date")
    plt.ylabel("Residual Value")
    plt.title("Cointegrating Residuals: LOW - β*HD")
    plt.legend()
    plt.show()

    # Check stationarity of the residuals
    check_stationarity(residuals)

if __name__ == '__main__':
    main()
