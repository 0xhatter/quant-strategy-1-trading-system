#!/usr/bin/env python3
"""
Real Data Training - All Available Tokens
Trains ML models using ONLY real Hyperliquid API data (no synthetic fallback)
"""

from main_trading_system import QuantTradingSystem
from data_collection import HyperliquidDataCollector

print("=" * 80)
print(" " * 20 + "REAL DATA TRAINING - ALL TOKENS")
print(" " * 25 + "No Synthetic Data Used")
print("=" * 80)

# Configuration
INITIAL_CAPITAL = 10000
LOOKBACK_HOURS = 720  # 30 days
TRAIN_EPOCHS = 50

# All tokens to attempt
ALL_TOKENS = [
    'BTC', 'ETH', 'SOL', 'AVAX', 'MATIC',
    'ARB', 'OP', 'ATOM', 'DOT', 'LINK',
    'HYPE', 'ASTER', 'WIF', 'BONK', 'PEPE',
    'JTO', 'JUP', 'PYTH', 'SEI', 'SUI'
]

print("\nStep 1: Testing API availability for all tokens...")
print("-" * 80)

# Test which tokens have real data available
collector = HyperliquidDataCollector(use_synthetic=False)
available_tokens = []

import sys
from io import StringIO

for symbol in ALL_TOKENS:
    try:
        print(f"Testing {symbol}...", end=" ")

        # Capture any output to detect synthetic data warning
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        df = collector.get_ohlcv(symbol, interval='1h', lookback_hours=24)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Check if we got real data (not synthetic or API error)
        is_synthetic = "Falling back to synthetic" in output or "API request error" in output

        if len(df) > 20 and 'timestamp' in df.columns and not is_synthetic:
            available_tokens.append(symbol)
            print(f"✓ Real data ({len(df)} candles)")
        elif is_synthetic:
            print(f"✗ Synthetic data (not using)")
        else:
            print(f"✗ No data")
    except Exception as e:
        sys.stdout = old_stdout
        print(f"✗ Error: {str(e)[:50]}")

print("\n" + "=" * 80)
print(f"Real Data Available: {len(available_tokens)}/{len(ALL_TOKENS)} tokens")
print("=" * 80)
print("\nTokens with real Hyperliquid data:")
for i, token in enumerate(available_tokens, 1):
    print(f"  {i:2d}. {token}")

if not available_tokens:
    print("\n✗ No tokens with real data available. Cannot proceed.")
    exit(1)

# Prioritize HYPE and ASTER if available
priority_tokens = []
if 'HYPE' in available_tokens:
    priority_tokens.append('HYPE')
if 'ASTER' in available_tokens:
    priority_tokens.append('ASTER')

print(f"\n🎯 Priority tokens found: {', '.join(priority_tokens) if priority_tokens else 'None'}")

print(f"\nStarting training on all {len(available_tokens)} tokens with real data...")

print("\n" + "=" * 80)
print("Step 2: Training models on all available tokens...")
print("=" * 80)

# Train on each token individually
all_results = {}

for idx, symbol in enumerate(available_tokens, 1):
    print(f"\n{'='*80}")
    print(f"Training {idx}/{len(available_tokens)}: {symbol}")
    if symbol in priority_tokens:
        print(f"⭐ PRIORITY TOKEN ⭐")
    print(f"{'='*80}")

    try:
        # Create a fresh system for each token
        system = QuantTradingSystem(
            initial_capital=INITIAL_CAPITAL,
            assets_to_select=1,  # Train one at a time
            use_synthetic_data=False  # ONLY real data
        )

        # Override to select this specific token
        system.asset_selector.asset_universe = [symbol]

        # Run training pipeline
        results = system.run_full_pipeline(
            lookback_hours=LOOKBACK_HOURS,
            train_epochs=TRAIN_EPOCHS
        )

        # Store results
        if symbol in results:
            all_results[symbol] = results[symbol]
            metrics = results[symbol]['metrics']

            print(f"\n✓ {symbol} Training Complete:")
            print(f"  Return:      {metrics['total_return']*100:>7.2f}%")
            print(f"  Sharpe:      {metrics['sharpe_ratio']:>7.2f}")
            print(f"  Max DD:      {metrics['max_drawdown']*100:>7.2f}%")
            print(f"  Win Rate:    {metrics['win_rate']*100:>7.2f}%")
            print(f"  Trades:      {metrics['total_trades']:>7}")

            # Save model
            system.save_models(path_prefix=f'real_data_model')
        else:
            print(f"⚠ {symbol} - No results generated")

    except Exception as e:
        print(f"\n✗ {symbol} Training Failed: {e}")
        continue

print("\n" + "=" * 80)
print(" " * 25 + "TRAINING SUMMARY")
print("=" * 80)

# Summary table
print(f"\nSuccessfully trained: {len(all_results)}/{len(available_tokens)} tokens\n")
print(f"{'Symbol':<8} {'Return':<10} {'Sharpe':<10} {'MaxDD':<10} {'WinRate':<10} {'Trades':<10}")
print("-" * 80)

for symbol, result in sorted(all_results.items()):
    metrics = result['metrics']
    priority_marker = "⭐" if symbol in priority_tokens else "  "
    print(f"{priority_marker}{symbol:<6} "
          f"{metrics['total_return']*100:>8.2f}% "
          f"{metrics['sharpe_ratio']:>9.2f} "
          f"{metrics['max_drawdown']*100:>8.2f}% "
          f"{metrics['win_rate']*100:>8.2f}% "
          f"{metrics['total_trades']:>9}")

# Highlight best performers
if all_results:
    print("\n" + "=" * 80)
    print("🏆 TOP PERFORMERS (by Sharpe Ratio):")
    print("=" * 80)

    sorted_by_sharpe = sorted(all_results.items(),
                              key=lambda x: x[1]['metrics']['sharpe_ratio'],
                              reverse=True)

    for i, (symbol, result) in enumerate(sorted_by_sharpe[:5], 1):
        metrics = result['metrics']
        priority = "⭐ PRIORITY" if symbol in priority_tokens else ""
        print(f"\n{i}. {symbol} {priority}")
        print(f"   Return:      {metrics['total_return']*100:>7.2f}%")
        print(f"   Sharpe:      {metrics['sharpe_ratio']:>7.2f}")
        print(f"   Win Rate:    {metrics['win_rate']*100:>7.2f}%")
        print(f"   Trades:      {metrics['total_trades']:>7}")

# HYPE and ASTER specific summary
if priority_tokens:
    print("\n" + "=" * 80)
    print("🎯 PRIORITY TOKENS SUMMARY:")
    print("=" * 80)

    for symbol in priority_tokens:
        if symbol in all_results:
            metrics = all_results[symbol]['metrics']
            print(f"\n{symbol}:")
            print(f"  Total Return:        {metrics['total_return']*100:>8.2f}%")
            print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
            print(f"  Max Drawdown:        {metrics['max_drawdown']*100:>8.2f}%")
            print(f"  Win Rate:            {metrics['win_rate']*100:>8.2f}%")
            print(f"  Total Trades:        {metrics['total_trades']:>8}")
            print(f"  Model Saved:         real_data_model_{symbol}.pth")
        else:
            print(f"\n{symbol}: ✗ Training failed or no data")

print("\n" + "=" * 80)
print("📁 Models saved with prefix: 'real_data_model_*.pth'")
print("📊 Charts saved as: 'training_history_*.png' and 'backtest_results_*.png'")
print("\n✓ All training complete using REAL Hyperliquid data only!")
print("=" * 80 + "\n")
