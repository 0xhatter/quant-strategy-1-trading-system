#!/usr/bin/env python3
"""
Train ML model for XPL token using real Hyperliquid data
"""

from main_trading_system import QuantTradingSystem
from data_collection import HyperliquidDataCollector

print("=" * 80)
print("TRAINING XPL WITH REAL HYPERLIQUID DATA")
print("=" * 80)

# Test data availability
collector = HyperliquidDataCollector(use_synthetic=False)
print("\nTesting XPL data availability...")
df = collector.get_ohlcv('XPL', interval='1h', lookback_hours=720)

if df is None or len(df) < 100:
    print("✗ Insufficient data for XPL")
    exit(1)

print(f"✓ Real data available: {len(df)} candles")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Latest price: ${df['close'].iloc[-1]:.4f}")

# Train model
print("\nTraining ML model for XPL...")
system = QuantTradingSystem(
    initial_capital=10000,
    assets_to_select=1,
    use_synthetic_data=False
)

# Override asset selection to just XPL
from asset_selection import AssetSelector
selector = AssetSelector(system.data_collector)
selector.asset_universe = ['XPL']
system.asset_selector = selector

# Run full training pipeline
results = system.run_full_pipeline(train_epochs=50)

if 'XPL' in results:
    result = results['XPL']
    print("\n" + "=" * 80)
    print("XPL TRAINING RESULTS")
    print("=" * 80)
    print(f"  Total Return:      {result['total_return']:>8.2f}%")
    print(f"  Sharpe Ratio:      {result['sharpe_ratio']:>8.2f}")
    print(f"  Max Drawdown:      {result['max_drawdown']:>8.2f}%")
    print(f"  Win Rate:          {result['win_rate']:>8.2f}%")
    print(f"  Total Trades:      {result['total_trades']:>8d}")
    print("=" * 80)
    print("\n✓ Model saved to real_data_model_XPL.pth")
else:
    print("\n✗ Training failed for XPL")
    exit(1)
