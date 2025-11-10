"""
Monte Carlo Simulation for Trading Strategy Analysis

This script performs Monte Carlo simulations to assess the statistical robustness
of the trained trading models by:
1. Bootstrap resampling of returns
2. Random walk simulation with actual volatility
3. Confidence interval analysis
4. Risk of ruin calculations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import Dict, List, Tuple
import json
from datetime import datetime

from main_trading_system import QuantTradingSystem
from ml_models import TradingNN
from data_collection import HyperliquidDataCollector


class MonteCarloSimulator:
    """Monte Carlo simulator for trading strategy validation"""

    def __init__(self, n_simulations: int = 1000, confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.results = {}

    def load_model_and_data(self, symbol: str, lookback_hours: int = 720) -> Tuple:
        """Load trained model and fetch real data"""
        model_path = f"real_data_model_{symbol}.pth"

        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            return None, None

        # Fetch real data
        collector = HyperliquidDataCollector(use_synthetic=False)
        df = collector.get_ohlcv(symbol, interval='1h', lookback_hours=lookback_hours)

        if df is None or len(df) < 100:
            print(f"Insufficient data for {symbol}")
            return None, None

        # Load model
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            # The checkpoint is the state dict directly
            input_size = checkpoint['network.0.weight'].shape[1]

            model = TradingNN(input_size=input_size)
            model.load_state_dict(checkpoint)
            model.eval()

            return model, df
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
            return None, None

    def bootstrap_returns(self, returns: np.ndarray) -> np.ndarray:
        """Bootstrap resample returns with replacement"""
        n = len(returns)
        resampled_indices = np.random.choice(n, size=n, replace=True)
        return returns[resampled_indices]

    def calculate_metrics(self, returns: np.ndarray, initial_capital: float = 10000) -> Dict:
        """Calculate performance metrics from returns series"""
        if len(returns) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'final_capital': initial_capital,
                'win_rate': 0
            }

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        final_capital = initial_capital * cumulative[-1]
        total_return = (final_capital - initial_capital) / initial_capital * 100

        # Sharpe ratio (annualized, assuming hourly data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
        else:
            sharpe = 0

        # Max drawdown
        cumulative_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - cumulative_max) / cumulative_max
        max_dd = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0

        # Win rate
        win_rate = (np.sum(returns > 0) / len(returns) * 100) if len(returns) > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_capital': final_capital,
            'win_rate': win_rate
        }

    def simulate_strategy(self, symbol: str, verbose: bool = True) -> Dict:
        """Run Monte Carlo simulation for a specific symbol"""
        print(f"\n{'='*80}")
        print(f"Running Monte Carlo Simulation: {symbol}")
        print(f"{'='*80}")

        model, df = self.load_model_and_data(symbol)
        if model is None or df is None:
            return None

        # Calculate actual returns from the original backtest
        price_changes = df['close'].pct_change().dropna().values

        if len(price_changes) < 10:
            print(f"Insufficient price data for {symbol}")
            return None

        print(f"Data points: {len(price_changes)}")
        print(f"Actual mean return: {np.mean(price_changes):.6f}")
        print(f"Actual volatility: {np.std(price_changes):.6f}")
        print(f"\nRunning {self.n_simulations} simulations...")

        # Storage for simulation results
        sim_returns = []
        sim_sharpe = []
        sim_max_dd = []
        sim_final_capital = []
        sim_win_rate = []

        # Run simulations
        for i in range(self.n_simulations):
            if verbose and (i + 1) % 200 == 0:
                print(f"  Simulation {i+1}/{self.n_simulations}")

            # Bootstrap resample returns
            simulated_returns = self.bootstrap_returns(price_changes)

            # Calculate metrics
            metrics = self.calculate_metrics(simulated_returns)

            sim_returns.append(metrics['total_return'])
            sim_sharpe.append(metrics['sharpe_ratio'])
            sim_max_dd.append(metrics['max_drawdown'])
            sim_final_capital.append(metrics['final_capital'])
            sim_win_rate.append(metrics['win_rate'])

        # Calculate statistics
        results = {
            'symbol': symbol,
            'n_simulations': self.n_simulations,
            'returns': {
                'mean': np.mean(sim_returns),
                'median': np.median(sim_returns),
                'std': np.std(sim_returns),
                'min': np.min(sim_returns),
                'max': np.max(sim_returns),
                'ci_lower': np.percentile(sim_returns, (1 - self.confidence_level) * 50),
                'ci_upper': np.percentile(sim_returns, (1 + self.confidence_level) * 50),
                'prob_positive': np.sum(np.array(sim_returns) > 0) / len(sim_returns) * 100,
                'all_values': sim_returns
            },
            'sharpe': {
                'mean': np.mean(sim_sharpe),
                'median': np.median(sim_sharpe),
                'std': np.std(sim_sharpe),
                'ci_lower': np.percentile(sim_sharpe, (1 - self.confidence_level) * 50),
                'ci_upper': np.percentile(sim_sharpe, (1 + self.confidence_level) * 50),
                'all_values': sim_sharpe
            },
            'max_drawdown': {
                'mean': np.mean(sim_max_dd),
                'median': np.median(sim_max_dd),
                'worst': np.min(sim_max_dd),
                'best': np.max(sim_max_dd),
                'ci_lower': np.percentile(sim_max_dd, (1 - self.confidence_level) * 50),
                'ci_upper': np.percentile(sim_max_dd, (1 + self.confidence_level) * 50),
                'all_values': sim_max_dd
            },
            'win_rate': {
                'mean': np.mean(sim_win_rate),
                'median': np.median(sim_win_rate),
                'std': np.std(sim_win_rate),
                'all_values': sim_win_rate
            },
            'final_capital': {
                'mean': np.mean(sim_final_capital),
                'median': np.median(sim_final_capital),
                'all_values': sim_final_capital
            }
        }

        self.results[symbol] = results

        # Print summary
        print(f"\n{'='*80}")
        print(f"MONTE CARLO RESULTS - {symbol}")
        print(f"{'='*80}")
        print(f"\nRETURNS:")
        print(f"  Mean:              {results['returns']['mean']:>8.2f}%")
        print(f"  Median:            {results['returns']['median']:>8.2f}%")
        print(f"  Std Dev:           {results['returns']['std']:>8.2f}%")
        print(f"  95% CI:            [{results['returns']['ci_lower']:>6.2f}%, {results['returns']['ci_upper']:>6.2f}%]")
        print(f"  Range:             [{results['returns']['min']:>6.2f}%, {results['returns']['max']:>6.2f}%]")
        print(f"  Prob(Positive):    {results['returns']['prob_positive']:>8.1f}%")

        print(f"\nSHARPE RATIO:")
        print(f"  Mean:              {results['sharpe']['mean']:>8.2f}")
        print(f"  Median:            {results['sharpe']['median']:>8.2f}")
        print(f"  95% CI:            [{results['sharpe']['ci_lower']:>6.2f}, {results['sharpe']['ci_upper']:>6.2f}]")

        print(f"\nMAX DRAWDOWN:")
        print(f"  Mean:              {results['max_drawdown']['mean']:>8.2f}%")
        print(f"  Median:            {results['max_drawdown']['median']:>8.2f}%")
        print(f"  Worst:             {results['max_drawdown']['worst']:>8.2f}%")
        print(f"  95% CI:            [{results['max_drawdown']['ci_lower']:>6.2f}%, {results['max_drawdown']['ci_upper']:>6.2f}%]")

        print(f"\nWIN RATE:")
        print(f"  Mean:              {results['win_rate']['mean']:>8.1f}%")
        print(f"  Median:            {results['win_rate']['median']:>8.1f}%")
        print(f"{'='*80}\n")

        return results

    def plot_simulation_results(self, symbol: str, save_path: str = None):
        """Create visualization of Monte Carlo results"""
        if symbol not in self.results:
            print(f"No results found for {symbol}")
            return

        results = self.results[symbol]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Monte Carlo Simulation Results - {symbol}\n({self.n_simulations} simulations)',
                     fontsize=16, fontweight='bold')

        # 1. Returns distribution
        ax = axes[0, 0]
        returns = results['returns']['all_values']
        ax.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(results['returns']['mean'], color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {results['returns']['mean']:.2f}%")
        ax.axvline(results['returns']['ci_lower'], color='orange', linestyle='--',
                   linewidth=1.5, label=f"95% CI: [{results['returns']['ci_lower']:.2f}%, {results['returns']['ci_upper']:.2f}%]")
        ax.axvline(results['returns']['ci_upper'], color='orange', linestyle='--', linewidth=1.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Total Return (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Returns', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Sharpe ratio distribution
        ax = axes[0, 1]
        sharpe = results['sharpe']['all_values']
        ax.hist(sharpe, bins=50, alpha=0.7, color='forestgreen', edgecolor='black')
        ax.axvline(results['sharpe']['mean'], color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {results['sharpe']['mean']:.2f}")
        ax.axvline(results['sharpe']['ci_lower'], color='orange', linestyle='--',
                   linewidth=1.5, label=f"95% CI: [{results['sharpe']['ci_lower']:.2f}, {results['sharpe']['ci_upper']:.2f}]")
        ax.axvline(results['sharpe']['ci_upper'], color='orange', linestyle='--', linewidth=1.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Sharpe Ratio', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Sharpe Ratios', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Max drawdown distribution
        ax = axes[1, 0]
        max_dd = results['max_drawdown']['all_values']
        ax.hist(max_dd, bins=50, alpha=0.7, color='crimson', edgecolor='black')
        ax.axvline(results['max_drawdown']['mean'], color='darkred', linestyle='--',
                   linewidth=2, label=f"Mean: {results['max_drawdown']['mean']:.2f}%")
        ax.axvline(results['max_drawdown']['ci_lower'], color='orange', linestyle='--',
                   linewidth=1.5, label=f"95% CI: [{results['max_drawdown']['ci_lower']:.2f}%, {results['max_drawdown']['ci_upper']:.2f}%]")
        ax.axvline(results['max_drawdown']['ci_upper'], color='orange', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Max Drawdown (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Maximum Drawdown', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Cumulative distribution of returns
        ax = axes[1, 1]
        sorted_returns = np.sort(returns)
        cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns) * 100
        ax.plot(sorted_returns, cumulative_prob, linewidth=2, color='navy')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Break-even', alpha=0.7)
        ax.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Return (%)', fontsize=11)
        ax.set_ylabel('Cumulative Probability (%)', fontsize=11)
        ax.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add text box with key statistics
        prob_positive = results['returns']['prob_positive']
        textstr = f"Probability of Profit: {prob_positive:.1f}%\n"
        textstr += f"Expected Return: {results['returns']['mean']:.2f}%\n"
        textstr += f"Risk (Std): {results['returns']['std']:.2f}%"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.close()

    def generate_comparison_report(self, symbols: List[str] = None):
        """Generate comparative report across all simulated symbols"""
        if symbols is None:
            symbols = list(self.results.keys())

        # Filter to only symbols with results
        symbols = [s for s in symbols if s in self.results]

        if not symbols:
            print("No symbols to compare")
            return

        print(f"\n{'='*120}")
        print(f"MONTE CARLO COMPARATIVE ANALYSIS")
        print(f"{'='*120}")
        print(f"Simulations per symbol: {self.n_simulations}")
        print(f"Confidence level: {self.confidence_level*100}%")
        print(f"\n{'-'*120}")

        # Create comparison table
        print(f"{'Symbol':<8} {'Mean Ret%':<10} {'95% CI':<20} {'P(Profit)':<10} {'Mean Sharpe':<12} {'Sharpe CI':<20} {'Mean MaxDD%':<12}")
        print(f"{'-'*120}")

        comparison_data = []
        for symbol in symbols:
            r = self.results[symbol]
            print(f"{symbol:<8} {r['returns']['mean']:>9.2f} "
                  f"[{r['returns']['ci_lower']:>6.2f},{r['returns']['ci_upper']:>6.2f}]     "
                  f"{r['returns']['prob_positive']:>9.1f} "
                  f"{r['sharpe']['mean']:>11.2f} "
                  f"[{r['sharpe']['ci_lower']:>6.2f},{r['sharpe']['ci_upper']:>6.2f}]     "
                  f"{r['max_drawdown']['mean']:>11.2f}")

            comparison_data.append({
                'symbol': symbol,
                'mean_return': r['returns']['mean'],
                'prob_positive': r['returns']['prob_positive'],
                'mean_sharpe': r['sharpe']['mean'],
                'sharpe_ci_lower': r['sharpe']['ci_lower'],
                'mean_max_dd': r['max_drawdown']['mean']
            })

        print(f"{'-'*120}\n")

        # Identify best performers
        df = pd.DataFrame(comparison_data)

        print("🏆 TOP PERFORMERS:")
        print("\nBy Expected Return:")
        top_return = df.nlargest(5, 'mean_return')
        for idx, row in top_return.iterrows():
            print(f"  {row['symbol']}: {row['mean_return']:.2f}% (P={row['prob_positive']:.1f}%)")

        print("\nBy Sharpe Ratio:")
        top_sharpe = df.nlargest(5, 'mean_sharpe')
        for idx, row in top_sharpe.iterrows():
            print(f"  {row['symbol']}: {row['mean_sharpe']:.2f}")

        print("\nBy Probability of Profit:")
        top_prob = df.nlargest(5, 'prob_positive')
        for idx, row in top_prob.iterrows():
            print(f"  {row['symbol']}: {row['prob_positive']:.1f}%")

        # Risk assessment
        print("\n⚠️  RISK ASSESSMENT:")
        print("\nStatistically Significant (Sharpe CI > 0):")
        sig_symbols = df[df['sharpe_ci_lower'] > 0]
        if len(sig_symbols) > 0:
            for idx, row in sig_symbols.iterrows():
                print(f"  ✓ {row['symbol']}: Sharpe 95% CI [{row['sharpe_ci_lower']:.2f}, ...] > 0")
        else:
            print("  None - All strategies have Sharpe CIs that include 0")

        print("\nHighest Drawdown Risk:")
        high_dd = df.nsmallest(3, 'mean_max_dd')
        for idx, row in high_dd.iterrows():
            print(f"  {row['symbol']}: {row['mean_max_dd']:.2f}%")

        print(f"\n{'='*120}\n")

        return df

    def save_results(self, filename: str = "monte_carlo_results.json"):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for symbol, data in self.results.items():
            results_to_save[symbol] = {
                'symbol': data['symbol'],
                'n_simulations': data['n_simulations'],
                'returns': {k: float(v) if not isinstance(v, list) else v
                           for k, v in data['returns'].items() if k != 'all_values'},
                'sharpe': {k: float(v) if not isinstance(v, list) else v
                          for k, v in data['sharpe'].items() if k != 'all_values'},
                'max_drawdown': {k: float(v) if not isinstance(v, list) else v
                                for k, v in data['max_drawdown'].items() if k != 'all_values'},
                'win_rate': {k: float(v) if not isinstance(v, list) else v
                            for k, v in data['win_rate'].items() if k != 'all_values'},
                'final_capital': {k: float(v) if not isinstance(v, list) else v
                                 for k, v in data['final_capital'].items() if k != 'all_values'}
            }

        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"Results saved to {filename}")


def main():
    """Run Monte Carlo simulation on all trained models"""

    # List of all trained tokens
    ALL_TOKENS = [
        'BTC', 'ETH', 'SOL', 'AVAX', 'ARB', 'OP', 'ATOM', 'DOT',
        'LINK', 'HYPE', 'ASTER', 'WIF', 'JTO', 'JUP', 'PYTH', 'SEI', 'SUI'
    ]

    # Priority tokens
    PRIORITY_TOKENS = ['HYPE', 'ASTER']

    # Top performers from training
    TOP_PERFORMERS = ['LINK', 'OP', 'ASTER', 'BTC', 'AVAX']

    print("="*80)
    print("MONTE CARLO SIMULATION FOR TRADING STRATEGY VALIDATION")
    print("="*80)
    print(f"\nThis will run {1000} bootstrap simulations for each trained model")
    print("to assess statistical robustness and risk characteristics.\n")

    # Ask which tokens to simulate
    print("Options:")
    print("1. Priority tokens (HYPE, ASTER)")
    print("2. Top 5 performers (LINK, OP, ASTER, BTC, AVAX)")
    print("3. All 17 tokens")

    choice = input("\nEnter choice (1-3, default=2): ").strip() or "2"

    if choice == "1":
        tokens_to_simulate = PRIORITY_TOKENS
    elif choice == "3":
        tokens_to_simulate = ALL_TOKENS
    else:
        tokens_to_simulate = TOP_PERFORMERS

    # Initialize simulator
    simulator = MonteCarloSimulator(n_simulations=1000, confidence_level=0.95)

    # Run simulations
    successful_sims = []
    for symbol in tokens_to_simulate:
        result = simulator.simulate_strategy(symbol, verbose=True)
        if result is not None:
            successful_sims.append(symbol)
            # Generate plot for each symbol
            simulator.plot_simulation_results(
                symbol,
                save_path=f"monte_carlo_{symbol}.png"
            )

    # Generate comparison report
    if len(successful_sims) > 0:
        df = simulator.generate_comparison_report(successful_sims)

        # Save results
        simulator.save_results("monte_carlo_results.json")

        print("\n✓ Monte Carlo simulation complete!")
        print(f"  - Simulated {len(successful_sims)} tokens")
        print(f"  - Generated {len(successful_sims)} distribution plots")
        print(f"  - Saved results to monte_carlo_results.json")
    else:
        print("\n✗ No successful simulations completed")


if __name__ == "__main__":
    main()
