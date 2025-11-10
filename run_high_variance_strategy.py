#!/usr/bin/env python3
"""
High-Variance Trading Strategy
Focuses on HYPE, ASTER, and other high-variance tokens for maximum trading opportunities.
"""

from main_trading_system import QuantTradingSystem

def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "HIGH-VARIANCE TRADING STRATEGY")
    print(" " * 25 + "Targeting HYPE, ASTER & High-Vol Tokens")
    print("=" * 80)

    # Configuration
    INITIAL_CAPITAL = 10000
    ASSETS_TO_SELECT = 5  # Select top 5 high-variance tokens
    LOOKBACK_HOURS = 720  # 30 days of data
    TRAIN_EPOCHS = 50     # Full training

    print("\nConfiguration:")
    print(f"  Initial Capital:     ${INITIAL_CAPITAL:,}")
    print(f"  Assets to Select:    {ASSETS_TO_SELECT}")
    print(f"  Lookback Period:     {LOOKBACK_HOURS} hours ({LOOKBACK_HOURS//24} days)")
    print(f"  Training Epochs:     {TRAIN_EPOCHS}")
    print(f"  Data Source:         Hyperliquid API (Real-Time)")
    print(f"  Selection Criterion: Variance Score (Highest Volatility)")

    # Initialize system with real Hyperliquid data
    system = QuantTradingSystem(
        initial_capital=INITIAL_CAPITAL,
        assets_to_select=ASSETS_TO_SELECT,
        use_synthetic_data=False  # Use real Hyperliquid API data
    )

    # Override asset selector to prioritize variance
    # The system will automatically select highest-variance tokens
    # from the expanded universe including HYPE, ASTER, WIF, BONK, PEPE, etc.
    print("\nAsset Universe (20 tokens):")
    print("  " + ", ".join(system.asset_selector.asset_universe))

    print("\nStarting analysis and training automatically...")

    # Run full pipeline with variance-focused selection
    print("\n" + "=" * 80)
    print("Starting High-Variance Trading Pipeline...")
    print("=" * 80)

    # Modify asset selection to sort by variance instead of composite score
    original_select = system.asset_selector.select_top_assets

    def variance_select(n, **kwargs):
        """Select assets based on variance score"""
        kwargs['sort_by'] = 'variance_score'
        return original_select(n=n, **kwargs)

    system.asset_selector.select_top_assets = variance_select

    # Run the full pipeline
    results = system.run_full_pipeline(
        lookback_hours=LOOKBACK_HOURS,
        train_epochs=TRAIN_EPOCHS
    )

    # Additional high-variance analysis
    print("\n" + "=" * 80)
    print(" " * 25 + "HIGH-VARIANCE ANALYSIS")
    print("=" * 80)

    for symbol, result in results.items():
        metrics = result['metrics']
        print(f"\n{symbol}:")
        print(f"  Total Return:        {metrics['total_return']*100:>8.2f}%")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']*100:>8.2f}%")
        print(f"  Win Rate:            {metrics['win_rate']*100:>8.2f}%")
        print(f"  Total Trades:        {metrics['total_trades']:>8}")

    # Save models
    print("\n" + "=" * 80)
    print("Saving trained models...")
    system.save_models(path_prefix='high_variance_model')

    print("\n" + "=" * 80)
    print(" " * 20 + "HIGH-VARIANCE STRATEGY COMPLETE")
    print("=" * 80)

    # Generate live signals for high-variance tokens
    print("\n" + "=" * 80)
    print(" " * 25 + "LIVE TRADING SIGNALS")
    print("=" * 80)

    signals = system.generate_live_signals()

    for symbol, signal_data in signals.items():
        print(f"\n{symbol}:")
        print(f"  Signal Strength:     {signal_data['signal']:>8.4f}")
        print(f"  Recommendation:      {signal_data['recommendation']:>12s}")
        print(f"  Current Price:       ${signal_data['current_price']:>8.2f}")
        print(f"  Position Size:       ${signal_data['position']['position_value']:>8.2f}")
        print(f"  Leverage:            {signal_data['position']['leverage']:>8.2f}x")
        print(f"  Stop Loss:           ${signal_data['position']['stop_loss']:>8.2f}")
        print(f"  Take Profit:         ${signal_data['position']['take_profit']:>8.2f}")

    print("\n" + "=" * 80)
    print("\nModels saved with prefix: 'high_variance_model_*.pth'")
    print("Charts saved as: 'training_history_*.png' and 'backtest_results_*.png'")
    print("\nReady for live trading on high-variance tokens!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
