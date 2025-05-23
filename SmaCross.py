from backtesting import Backtest, Strategy
from backtesting.lib import crossover, plot_heatmaps
from backtesting.test import SMA
import yfinance as yf
import numpy as np
import pandas as pd

# Set up parameters
cash = 10000  # Initial cash amount in USD
commission = 0.001  # Commission per trade
exclusive_orders = True  # Only one order at a time

# Download and prepare AVAX-USD data from Yahoo Finance
ticker_name = "AVAX-USD"
data = yf.download(ticker_name, start='2020-01-01', end='2025-04-20', interval='1d')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Initial run moving averages
n1 = 35   # Fast
n2 = 170  # Slow

# Uncomment the following line to see the data before running the backtest
# print(data.tail())

class SmaCross(Strategy):
    n1 = n1
    n2 = n2

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        # Get the current date
        current_date = self.data.index[-1]

        #
        # Uncomment one of the following blocks to restrict the backtest to a specific year
        #

        # Operate in 2025
        # if current_date > pd.Timestamp('2025-12-31') or current_date < pd.Timestamp('2025-01-01'):
        #     return
        
        # Operate in 2024
        # if current_date > pd.Timestamp('2024-12-31') or current_date < pd.Timestamp('2024-01-01'):
        #     return
        
        # Operate in 2023
        # if current_date > pd.Timestamp('2023-12-31') or current_date < pd.Timestamp('2023-01-01'):
        #     return

        # Operate in 2022
        # if current_date > pd.Timestamp('2022-12-31') or current_date < pd.Timestamp('2022-01-01'):
        #     return

        # Operate in 2021
        # if current_date > pd.Timestamp('2021-12-31') or current_date < pd.Timestamp('2021-01-01'):
        #     return
        
        # Operate in 2020
        # if current_date > pd.Timestamp('2020-12-31') or current_date < pd.Timestamp('2020-01-01'):
        #     return

        if crossover(self.sma1, self.sma2):
            # LONG: If the fast SMA crosses above the slow SMA, close the position and buy
            self.position.close()
            self.buy(sl=self.data.Close[-1] * 0.95)  # 5% stop loss
        elif crossover(self.sma2, self.sma1):
            # SHORT: If the fast SMA crosses below the slow SMA, close the position and sell
            self.position.close()
            self.sell(sl=self.data.Close[-1] * 1.05)  # 5% stop loss

# Initialize the backtest
bt = Backtest(data, SmaCross, cash=cash, commission=commission, exclusive_orders=exclusive_orders)

# Print results of the backtest
print(f"\nInitial Backtest Results (n1={n1}, n2={n2}):\n")
print(bt.run())

# Run optimization
optimization_results, heatmap = bt.optimize(
    n1=range(2, 52, 2),  # Fast SMA from 2 to 50, step 2
    n2=range(10, 202, 2),  # Slow SMA from 10 to 202, step 2
    maximize='Return [%]',  # Maximize Return
    constraint=lambda p: p.n1 < p.n2,  # Ensure fast SMA is shorter than slow SMA
    return_heatmap=True # Return heatmap data
)

# Print optimization results
best_n1 = optimization_results._strategy.n1
best_n2 = optimization_results._strategy.n2
print(f"\nOptimized Backtest Results (n1={best_n1}, n2={best_n2}):\n")
print(optimization_results)

# Run the backtest with the best parameters
best_run = bt.run(n1=best_n1, n2=best_n2)

# Plot the results of the best run
bt.plot()

# Copy the trades DataFrame
trades = best_run['_trades'].copy()

# Add the trade type
trades['Type'] = np.where(trades['Size'] > 0, 'Long', 'Short')

# Compute return in percentage
trades['ReturnPct'] = np.where(
    trades['Type'] == 'Long',
    (trades['ExitPrice'] - trades['EntryPrice']) / trades['EntryPrice'] * 100,
    (trades['EntryPrice'] - trades['ExitPrice']) / trades['EntryPrice'] * 100
)

# Cleanup and reorder columns
useful_cols = ['Type', 'EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct', 'Duration']
trades = trades[useful_cols]

# Print the trades
print("\nBest Run Trades:\n")
print(trades.to_string())

# Save trades to a CSV file
trades.to_csv('trades.csv', index=False)

# Plot Heatmap
plot_heatmaps(heatmap, agg='mean')