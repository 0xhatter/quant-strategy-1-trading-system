"""
Example Workflow
Step-by-step tutorial showing how to use the quantitative trading system.
"""

import numpy as np
import pandas as pd

# Import all modules
from data_collection import HyperliquidDataCollector
from asset_selection import AssetSelector
from feature_engineering import FeatureEngineer
from ml_models import TradingNN, TradingDataset, ModelTrainer, SharpeLoss
from risk_management import RiskManager
from backtesting import Backtest
from main_trading_system import QuantTradingSystem

import torch
from torch.utils.data import DataLoader


def example_1_data_collection():
    """Example 1: Collect and analyze market data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: DATA COLLECTION")
    print("=" * 70)
    
    # Initialize data collector
    collector = HyperliquidDataCollector(use_synthetic=True)
    
    # Fetch OHLCV data for BTC
    print("\nFetching BTC data...")
    df = collector.get_ohlcv('BTC', interval='1h', lookback_hours=720)
    print(f"Retrieved {len(df)} hourly candles")
    
    # Calculate variance and volume metrics
    print("\nCalculating metrics...")
    df = collector.calculate_variance_metrics(df)
    
    # Display sample
    print("\nSample data:")
    print(df[['timestamp', 'close', 'volume', 'variance_24h', 'volatility_24h']].tail())
    
    return df


def example_2_asset_selection():
    """Example 2: Select best assets to trade."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: ASSET SELECTION")
    print("=" * 70)
    
    # Initialize asset selector
    selector = AssetSelector()
    
    # Select top 3 assets
    print("\nSelecting top 3 assets...")
    top_assets = selector.select_top_assets(n=3, lookback_hours=720)
    
    print("\nTop assets:")
    print(top_assets[['symbol', 'composite_score', 'volatility_score', 'volume_score']].head(3))
    
    return top_assets


def example_3_feature_engineering(df):
    """Example 3: Engineer features from raw data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: FEATURE ENGINEERING")
    print("=" * 70)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Create all features
    print("\nCreating features...")
    df_features = engineer.create_all_features(df)
    
    # Prepare ML datasets
    print("\nPreparing train/val/test splits...")
    train_df, val_df, test_df = engineer.prepare_ml_dataset(df_features)
    
    print(f"\nFeature columns ({len(engineer.feature_columns)}):")
    print(engineer.feature_columns[:10], "...")
    
    return train_df, val_df, test_df, engineer


def example_4_model_training(train_df, val_df, test_df, engineer):
    """Example 4: Train ML model with Sharpe loss."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: MODEL TRAINING")
    print("=" * 70)
    
    # Prepare data loaders
    print("\nPreparing data loaders...")
    train_dataset = TradingDataset(
        train_df[engineer.feature_columns].values,
        train_df['target'].values
    )
    val_dataset = TradingDataset(
        val_df[engineer.feature_columns].values,
        val_df['target'].values
    )
    test_dataset = TradingDataset(
        test_df[engineer.feature_columns].values,
        test_df['target'].values
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\nCreating neural network...")
    model = TradingNN(
        input_size=len(engineer.feature_columns),
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.3
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining model with Sharpe loss...")
    trainer = ModelTrainer(model, SharpeLoss(), learning_rate=0.001)
    history = trainer.train(
        train_loader, val_loader,
        epochs=30,
        early_stopping_patience=10
    )
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = trainer.predict(test_loader)
    
    return model, trainer, predictions, test_loader


def example_5_risk_management():
    """Example 5: Calculate position sizes with risk management."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: RISK MANAGEMENT")
    print("=" * 70)
    
    # Initialize risk manager
    risk_manager = RiskManager(
        total_capital=10000,
        max_position_size=0.20,
        max_leverage=10.0,
        max_portfolio_risk=0.15
    )
    
    # Calculate position for a strong buy signal
    print("\nCalculating position for BTC (strong buy signal)...")
    position = risk_manager.calculate_position_size(
        symbol='BTC',
        signal_strength=0.8,
        current_price=45000,
        volatility=0.60,
        win_rate=0.55,
        avg_win=0.03,
        avg_loss=0.02
    )
    
    print("\nPosition details:")
    for key, value in position.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    
    # Check portfolio risk
    is_acceptable, reason = risk_manager.check_portfolio_risk(position)
    print(f"\nRisk check: {is_acceptable} - {reason}")
    
    return risk_manager


def example_6_backtesting(test_df, predictions):
    """Example 6: Run comprehensive backtest."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: BACKTESTING")
    print("=" * 70)
    
    # Initialize backtest
    backtest = Backtest(
        initial_capital=10000,
        commission_rate=0.0004,
        slippage_rate=0.0005
    )
    
    # Run backtest
    print("\nRunning backtest...")
    equity_curve = backtest.run_backtest(
        test_df, predictions,
        symbol='BTC',
        signal_threshold=0.1
    )
    
    # Print performance
    backtest.print_performance_summary()
    
    # Plot results
    backtest.plot_results('backtest_results.png')
    
    return backtest


def example_7_full_pipeline():
    """Example 7: Run complete end-to-end pipeline."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: COMPLETE PIPELINE")
    print("=" * 70)
    
    # Initialize system
    system = QuantTradingSystem(
        initial_capital=10000,
        assets_to_select=3,
        use_synthetic_data=True
    )
    
    # Run full pipeline
    print("\nRunning full pipeline...")
    results = system.run_full_pipeline(
        lookback_hours=720,
        train_epochs=30
    )
    
    # Generate live signals
    print("\n" + "=" * 70)
    print("GENERATING LIVE SIGNALS")
    print("=" * 70)
    
    signals = system.generate_live_signals()
    
    print("\nCurrent trading signals:")
    for symbol, signal_data in signals.items():
        print(f"\n{symbol}:")
        print(f"  Signal:          {signal_data['signal']:>8.4f}")
        print(f"  Recommendation:  {signal_data['recommendation']:>12s}")
        print(f"  Current Price:   ${signal_data['current_price']:>8.2f}")
        print(f"  Position Value:  ${signal_data['position']['position_value']:>8.2f}")
    
    return system, results


def main():
    """Run all examples in sequence."""
    print("\n" + "=" * 70)
    print(" " * 15 + "QUANTITATIVE TRADING SYSTEM")
    print(" " * 20 + "EXAMPLE WORKFLOW")
    print("=" * 70)
    
    print("\nThis workflow demonstrates all components of the trading system.")
    print("Each example builds on the previous one.")
    
    # Example 1: Data Collection
    df = example_1_data_collection()
    
    # Example 2: Asset Selection
    top_assets = example_2_asset_selection()
    
    # Example 3: Feature Engineering
    train_df, val_df, test_df, engineer = example_3_feature_engineering(df)
    
    # Example 4: Model Training
    model, trainer, predictions, test_loader = example_4_model_training(
        train_df, val_df, test_df, engineer
    )
    
    # Example 5: Risk Management
    risk_manager = example_5_risk_management()
    
    # Example 6: Backtesting
    backtest = example_6_backtesting(test_df, predictions)
    
    # Example 7: Full Pipeline
    system, results = example_7_full_pipeline()
    
    print("\n" + "=" * 70)
    print(" " * 20 + "WORKFLOW COMPLETE!")
    print("=" * 70)
    
    print("\nGenerated files:")
    print("  - training_history.png")
    print("  - backtest_results.png")
    print("  - training_history_*.png (for each asset)")
    print("  - backtest_results_*.png (for each asset)")
    
    print("\nNext steps:")
    print("  1. Review the generated plots")
    print("  2. Adjust parameters in each module")
    print("  3. Try different assets or time periods")
    print("  4. Experiment with model architectures")
    print("  5. Implement your own strategies!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
