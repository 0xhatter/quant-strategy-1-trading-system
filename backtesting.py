"""
Backtesting Module
Realistic backtesting engine with transaction costs, slippage, and risk management.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from risk_management import RiskManager


class Backtest:
    """
    Comprehensive backtesting engine for trading strategies.
    Includes realistic costs, slippage, and risk management.
    """
    
    def __init__(self, initial_capital: float = 10000,
                 commission_rate: float = 0.0004,
                 slippage_rate: float = 0.0005,
                 risk_manager: Optional[RiskManager] = None):
        """
        Initialize backtest.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission per trade (as fraction)
            slippage_rate: Slippage per trade (as fraction)
            risk_manager: RiskManager instance (creates new if None)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        self.risk_manager = risk_manager or RiskManager(
            total_capital=initial_capital
        )
        
        # Results tracking
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.positions_history = []
    
    def run_backtest(self, df: pd.DataFrame, predictions: np.ndarray,
                    symbol: str = 'BTC', signal_threshold: float = 0.1) -> pd.DataFrame:
        """
        Run backtest on historical data with predictions.
        
        Args:
            df: DataFrame with OHLCV and features
            predictions: Model predictions (position signals)
            symbol: Trading symbol
            signal_threshold: Minimum signal strength to trade
            
        Returns:
            DataFrame with equity curve
        """
        print(f"\nRunning backtest for {symbol}...")
        print("=" * 60)
        
        # Ensure predictions match dataframe length
        if len(predictions) != len(df):
            print(f"Warning: Predictions length {len(predictions)} != Data length {len(df)}")
            min_len = min(len(predictions), len(df))
            predictions = predictions[:min_len]
            df = df.iloc[:min_len].copy()
        
        # Initialize
        capital = self.initial_capital
        position = None
        equity = [capital]
        
        # Iterate through data
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            signal = predictions[i][0] if predictions[i].ndim > 0 else predictions[i]
            volatility = df.iloc[i].get('volatility_24h', 0.5)
            
            # Check for stop loss or take profit
            if position is not None:
                if position['direction'] == 'long':
                    if current_price <= position['stop_loss']:
                        capital = self._close_position(position, current_price, 'stop_loss')
                        position = None
                    elif current_price >= position['take_profit']:
                        capital = self._close_position(position, current_price, 'take_profit')
                        position = None
                else:  # short
                    if current_price >= position['stop_loss']:
                        capital = self._close_position(position, current_price, 'stop_loss')
                        position = None
                    elif current_price <= position['take_profit']:
                        capital = self._close_position(position, current_price, 'take_profit')
                        position = None
            
            # Generate trading signals
            if abs(signal) > signal_threshold:
                # Close opposite position
                if position is not None:
                    if (signal > 0 and position['direction'] == 'short') or \
                       (signal < 0 and position['direction'] == 'long'):
                        capital = self._close_position(position, current_price, 'signal_flip')
                        position = None
                
                # Open new position
                if position is None:
                    self.risk_manager.total_capital = capital
                    
                    new_position = self.risk_manager.calculate_position_size(
                        symbol=symbol,
                        signal_strength=signal,
                        current_price=current_price,
                        volatility=volatility
                    )
                    
                    # Check if position is acceptable
                    is_acceptable, _ = self.risk_manager.check_portfolio_risk(new_position)
                    
                    if is_acceptable:
                        # Apply transaction costs
                        commission = new_position['position_value'] * self.commission_rate
                        slippage = new_position['position_value'] * self.slippage_rate
                        total_cost = commission + slippage
                        
                        capital -= total_cost
                        new_position['entry_cost'] = total_cost
                        new_position['entry_index'] = i
                        new_position['entry_time'] = df.iloc[i]['timestamp']
                        
                        position = new_position
                        self.positions_history.append(position.copy())
            
            # Update equity
            if position is not None:
                # Mark-to-market
                if position['direction'] == 'long':
                    unrealized_pnl = (current_price - position['entry_price']) * position['num_units']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['num_units']
                
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            
            equity.append(current_equity)
        
        # Close any remaining position
        if position is not None:
            final_price = df.iloc[-1]['close']
            capital = self._close_position(position, final_price, 'backtest_end')
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            'timestamp': df['timestamp'].values,
            'equity': equity[1:],  # Skip initial equity
            'returns': pd.Series(equity[1:]).pct_change()
        })
        
        self.equity_curve = equity_df
        
        print(f"Backtest complete!")
        print(f"  Total trades: {len(self.trades)}")
        print(f"  Final capital: ${capital:.2f}")
        print(f"  Total return: {(capital/self.initial_capital - 1)*100:.2f}%")
        
        return equity_df
    
    def _close_position(self, position: Dict, exit_price: float, reason: str) -> float:
        """Close a position and return updated capital."""
        # Calculate P&L
        if position['direction'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['num_units']
        else:
            pnl = (position['entry_price'] - exit_price) * position['num_units']
        
        # Apply exit costs
        exit_value = exit_price * position['num_units']
        commission = exit_value * self.commission_rate
        slippage = exit_value * self.slippage_rate
        total_cost = commission + slippage
        
        net_pnl = pnl - position['entry_cost'] - total_cost
        
        # Record trade
        trade = {
            'symbol': position['symbol'],
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'num_units': position['num_units'],
            'entry_time': position['entry_time'],
            'exit_time': pd.Timestamp.now(),
            'pnl': net_pnl,
            'return': net_pnl / position['position_value'],
            'reason': reason
        }
        self.trades.append(trade)
        
        # Return updated capital
        return self.risk_manager.total_capital + net_pnl
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if len(self.equity_curve) == 0:
            return {}
        
        equity = self.equity_curve['equity'].values
        returns = self.equity_curve['returns'].dropna().values
        
        # Basic metrics
        total_return = (equity[-1] / self.initial_capital) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252 * 24)  # Annualized (hourly data)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)
        
        # Drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # Trade statistics
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df)
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) \
                           if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(returns) / (downside_std + 1e-8) * np.sqrt(252 * 24)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 * 24 / len(returns)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': equity[-1]
        }
        
        return metrics
    
    def print_performance_summary(self):
        """Print formatted performance summary."""
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            print("No performance metrics available")
            return
        
        print("\n" + "=" * 60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print("\nRETURNS:")
        print(f"  Total Return:      {metrics['total_return']*100:>8.2f}%")
        print(f"  Annualized Return: {metrics['annualized_return']*100:>8.2f}%")
        
        print("\nRISK-ADJUSTED METRICS:")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']:>8.2f}")
        print(f"  Calmar Ratio:      {metrics['calmar_ratio']:>8.2f}")
        print(f"  Volatility:        {metrics['volatility']*100:>8.2f}%")
        print(f"  Max Drawdown:      {metrics['max_drawdown']*100:>8.2f}%")
        
        print("\nTRADE STATISTICS:")
        print(f"  Total Trades:      {metrics['total_trades']:>8}")
        print(f"  Win Rate:          {metrics['win_rate']*100:>8.2f}%")
        print(f"  Profit Factor:     {metrics['profit_factor']:>8.2f}")
        print(f"  Avg Win:           ${metrics['avg_win']:>8.2f}")
        print(f"  Avg Loss:          ${metrics['avg_loss']:>8.2f}")
        
        print("\nCAPITAL:")
        print(f"  Initial:           ${self.initial_capital:>8.2f}")
        print(f"  Final:             ${metrics['final_capital']:>8.2f}")
        
        print("=" * 60)
    
    def plot_results(self, save_path: str = 'backtest_results.png'):
        """Plot backtest results."""
        if len(self.equity_curve) == 0:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Equity curve
        axes[0].plot(self.equity_curve['timestamp'], self.equity_curve['equity'])
        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0].set_title('Equity Curve', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        equity = self.equity_curve['equity'].values
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax * 100
        axes[1].fill_between(self.equity_curve['timestamp'], drawdown, 0, alpha=0.3, color='red')
        axes[1].set_title('Drawdown', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Trade distribution
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            axes[2].hist(trades_df['pnl'], bins=30, alpha=0.7, edgecolor='black')
            axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.5)
            axes[2].set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('P&L ($)')
            axes[2].set_ylabel('Frequency')
            axes[2].grid(True, alpha=0.3)
            
            # Cumulative P&L
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            axes[3].plot(range(len(trades_df)), trades_df['cumulative_pnl'])
            axes[3].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[3].set_title('Cumulative P&L', fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Trade Number')
            axes[3].set_ylabel('Cumulative P&L ($)')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nBacktest results plot saved to {save_path}")
        plt.close()


# Example usage and testing
if __name__ == "__main__":
    from data_collection import HyperliquidDataCollector
    from feature_engineering import FeatureEngineer
    
    print("=" * 60)
    print("BACKTESTING MODULE TEST")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    collector = HyperliquidDataCollector(use_synthetic=True)
    df = collector.get_ohlcv('BTC', interval='1h', lookback_hours=720)
    df = collector.calculate_variance_metrics(df)
    
    # Engineer features
    print("\n2. Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    train_df, val_df, test_df = engineer.prepare_ml_dataset(df_features)
    
    # Generate simple predictions (for testing)
    print("\n3. Generating test predictions...")
    predictions = np.random.randn(len(test_df)) * 0.5  # Random signals
    
    # Run backtest
    print("\n4. Running backtest...")
    backtest = Backtest(initial_capital=10000)
    equity_curve = backtest.run_backtest(test_df, predictions, symbol='BTC')
    
    # Print results
    backtest.print_performance_summary()
    
    # Plot results
    backtest.plot_results()
    
    print("\n" + "=" * 60)
    print("BACKTESTING TEST COMPLETE")
    print("=" * 60)
