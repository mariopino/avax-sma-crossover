from backtesting import Backtest, Strategy
from backtesting.lib import crossover, plot_heatmaps
from backtesting.test import SMA
import yfinance as yf
import numpy as np

# Set up parameters
cash = 10000  # Initial cash amount in USD
commission = 0.001  # Commission per trade
exclusive_orders = True  # Only one order at a time

# Download and prepare AVAX-USD data from Yahoo Finance
tickerName = "AVAX-USD"
data = yf.download(tickerName, start='2020-01-01', end='2025-04-20', interval='1d')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Moving averages
n1 = 35   # Fast
n2 = 170  # Slow

# print(data.tail())

class SmaCross(Strategy):
    n1 = n1
    n2 = n2

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        # Uncomment the following lines to restrict the backtest to a specific date range
        # This is useful if you want to test the strategy only after a certain date.
        
        # Operate from 2025-01-01 onwards
        # current_date = self.data.index[-1]
        # if current_date < pd.Timestamp('2025-01-01'):
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
print(f"\nOptimization Results (n1={best_n1}, n2={best_n2}):\n")
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

# Optional: Calculate trade duration if not already present
if 'Duration' not in trades.columns and 'EntryTime' in trades and 'ExitTime' in trades:
    trades['Duration'] = trades['ExitTime'] - trades['EntryTime']

# Define useful columns (check that each actually exists)
useful_cols = ['Type', 'EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct', 'Duration']
useful_cols = [col for col in useful_cols if col in trades.columns]

# Reorder columns
trades = trades[useful_cols]

# Print the trades of the best run
print("\nBest Run Trades:\n")
print(trades.to_string())

# Save best run trades to a CSV file
trades.to_csv('trades.csv', index=False)

# Plot the best run heatmap
plot_heatmaps(heatmap, agg='mean')