"""
Focused Analysis: HYPE, ASTER, XPL
Train models and run Monte Carlo simulation for detailed comparison
"""

import sys
import numpy as np
import pandas as pd
from io import StringIO

from main_trading_system import QuantTradingSystem

def analyze_tokens():
    """Analyze HYPE, ASTER, and XPL"""

    tokens = ['HYPE', 'ASTER', 'XPL']

    print("="*80)
    print("FOCUSED ANALYSIS: HYPE, ASTER, XPL")
    print("="*80)
    print()
    print("Training models with real Hyperliquid data...")
    print()

    results = {}

    from data_collection import HyperliquidDataCollector
    from asset_selection import AssetSelector
    from feature_engineering import FeatureEngineer
    from ml_models import TradingNN, TradingDataset, ModelTrainer, SharpeLoss
    from backtesting import Backtest
    import torch
    from torch.utils.data import DataLoader

    for symbol in tokens:
        print(f"\n{'='*80}")
        print(f"ANALYZING {symbol}")
        print(f"{'='*80}\n")

        # Collect data
        collector = HyperliquidDataCollector(use_synthetic=False)
        print(f"[1/4] Fetching data for {symbol}...")
        df = collector.get_ohlcv(symbol, interval='1h', lookback_hours=720)

        if df is None or len(df) < 100:
            print(f"✗ Insufficient data for {symbol}")
            results[symbol] = None
            continue

        print(f"  ✓ Fetched {len(df)} candles")

        # Feature engineering
        print(f"[2/4] Creating features...")
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df)
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = engineer.prepare_train_test_split(df_features)

        print(f"  ✓ Created {len(feature_cols)} features")
        print(f"  ✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} samples")

        # Train model
        print(f"[3/4] Training model...")
        model = TradingNN(input_size=len(feature_cols))
        trainer = ModelTrainer(model, SharpeLoss())

        train_dataset = TradingDataset(X_train, y_train)
        val_dataset = TradingDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        history = trainer.train(train_loader, val_loader, epochs=50, patience=15)

        # Backtest
        print(f"[4/4] Running backtest...")
        backtest = Backtest(initial_capital=10000)
        backtest_results = backtest.run(model, X_test, y_test, df_features.iloc[-len(y_test):]['close'].values)

        print(f"\n✓ Training Complete:")
        print(f"  Total Return:      {backtest_results['total_return']:>8.2f}%")
        print(f"  Sharpe Ratio:      {backtest_results['sharpe_ratio']:>8.2f}")
        print(f"  Max Drawdown:      {backtest_results['max_drawdown']:>8.2f}%")
        print(f"  Win Rate:          {backtest_results['win_rate']:>8.2f}%")
        print(f"  Total Trades:      {backtest_results['total_trades']:>8d}")

        results[symbol] = backtest_results

        # Save model
        model_path = f"real_data_model_{symbol}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved to:    {model_path}")

    # Summary comparison
    print(f"\n{'='*80}")
    print("BACKTEST COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Token':<8} {'Return%':<10} {'Sharpe':<10} {'MaxDD%':<10} {'WinRate%':<10} {'Trades':<8}")
    print("-"*80)

    for symbol in tokens:
        if results[symbol]:
            r = results[symbol]
            print(f"{symbol:<8} {r['total_return']:>9.2f} {r['sharpe_ratio']:>9.2f} "
                  f"{r['max_drawdown']:>9.2f} {r['win_rate']:>9.2f} {r['total_trades']:>7d}")
        else:
            print(f"{symbol:<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<8}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

    return results


if __name__ == "__main__":
    analyze_tokens()
