#!/usr/bin/env python3
"""
Live Trading Signal Generator
Provides real-time trade recommendations based on trained models.
"""

from main_trading_system import QuantTradingSystem
import pandas as pd
from datetime import datetime

print("=" * 80)
print(" " * 25 + "LIVE TRADING SIGNALS")
print(" " * 20 + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
print("=" * 80)

# Initialize system with real data
system = QuantTradingSystem(
    initial_capital=10000,
    assets_to_select=5,
    use_synthetic_data=False
)

# Load the trained models (they were just trained)
print("\nLoading trained models...")
try:
    system.load_models(['BTC', 'DOT', 'BONK', 'MATIC', 'PEPE'],
                       path_prefix='high_variance_model')
    print("✓ Models loaded successfully")
except:
    print("⚠ Using freshly trained models from memory")

# Generate live signals
print("\n" + "=" * 80)
print("Fetching current market data and generating signals...")
print("=" * 80)

signals = system.generate_live_signals()

# Display trade recommendations
print("\n" + "=" * 80)
print(" " * 25 + "TRADE RECOMMENDATIONS")
print("=" * 80)

for symbol, signal_data in signals.items():
    signal = signal_data['signal']
    recommendation = signal_data['recommendation']
    price = signal_data['current_price']
    position = signal_data['position']

    print(f"\n{'='*80}")
    print(f"  {symbol}")
    print(f"{'='*80}")

    # Current market info
    print(f"\n  MARKET DATA:")
    print(f"    Current Price:       ${price:,.2f}")
    print(f"    24h Volatility:      {signal_data['volatility']:.2%}")

    # Signal analysis
    print(f"\n  SIGNAL ANALYSIS:")
    print(f"    Signal Strength:     {signal:>8.4f}")
    print(f"    Recommendation:      {recommendation:>12s}")

    # Position sizing
    print(f"\n  POSITION RECOMMENDATION:")
    print(f"    Position Size:       ${position['position_value']:,.2f}")
    print(f"    Leverage:            {position['leverage']:.2f}x")
    print(f"    Stop Loss:           ${position['stop_loss']:,.2f}")
    print(f"    Take Profit:         ${position['take_profit']:,.2f}")

    # Trade action
    print(f"\n  ACTION:")
    if recommendation == "STRONG BUY":
        print(f"    🟢 BUY {symbol} - Strong upward momentum detected")
        print(f"    Entry: ${price:,.2f}")
        print(f"    Target: ${position['take_profit']:,.2f} (+{((position['take_profit']/price)-1)*100:.2f}%)")
        print(f"    Stop: ${position['stop_loss']:,.2f} ({((position['stop_loss']/price)-1)*100:.2f}%)")
    elif recommendation == "BUY":
        print(f"    🟢 BUY {symbol} - Moderate upward signal")
        print(f"    Entry: ${price:,.2f}")
        print(f"    Target: ${position['take_profit']:,.2f} (+{((position['take_profit']/price)-1)*100:.2f}%)")
        print(f"    Stop: ${position['stop_loss']:,.2f} ({((position['stop_loss']/price)-1)*100:.2f}%)")
    elif recommendation == "STRONG SELL":
        print(f"    🔴 SELL/SHORT {symbol} - Strong downward momentum")
        print(f"    Entry: ${price:,.2f}")
        print(f"    Consider closing longs or opening short position")
    elif recommendation == "SELL":
        print(f"    🔴 SELL {symbol} - Moderate downward signal")
        print(f"    Entry: ${price:,.2f}")
        print(f"    Consider reducing position or taking profits")
    else:
        print(f"    ⚪ HOLD {symbol} - No clear signal, wait for better entry")
        print(f"    Current: ${price:,.2f}")
        print(f"    Monitor for signal strength > 0.1 or < -0.1")

print("\n" + "=" * 80)
print(" " * 30 + "SUMMARY")
print("=" * 80)

# Count recommendations
buy_count = sum(1 for s in signals.values() if s['recommendation'] in ['BUY', 'STRONG BUY'])
sell_count = sum(1 for s in signals.values() if s['recommendation'] in ['SELL', 'STRONG SELL'])
hold_count = sum(1 for s in signals.values() if s['recommendation'] == 'HOLD')

print(f"\n  Total Signals Analyzed: {len(signals)}")
print(f"    🟢 BUY Signals:  {buy_count}")
print(f"    🔴 SELL Signals: {sell_count}")
print(f"    ⚪ HOLD Signals: {hold_count}")

# Best opportunities
buy_signals = [(sym, data) for sym, data in signals.items()
               if data['recommendation'] in ['BUY', 'STRONG BUY']]
if buy_signals:
    buy_signals.sort(key=lambda x: x[1]['signal'], reverse=True)
    print(f"\n  🏆 BEST BUY OPPORTUNITY:")
    best = buy_signals[0]
    print(f"    {best[0]} - Signal: {best[1]['signal']:.4f} @ ${best[1]['current_price']:,.2f}")

print("\n" + "=" * 80)
print("\n  ⚠️  RISK DISCLAIMER:")
print("    - These are AI-generated signals based on historical patterns")
print("    - Past performance does not guarantee future results")
print("    - Always use proper risk management and position sizing")
print("    - Never risk more than you can afford to lose")
print("    - Consider market conditions and your own analysis")
print("\n" + "=" * 80)

# Export to file
print("\nExporting signals to 'live_signals.csv'...")
signals_df = pd.DataFrame([
    {
        'Symbol': sym,
        'Timestamp': datetime.now(),
        'Price': data['current_price'],
        'Signal': data['signal'],
        'Recommendation': data['recommendation'],
        'Position_Size': data['position']['position_value'],
        'Stop_Loss': data['position']['stop_loss'],
        'Take_Profit': data['position']['take_profit'],
        'Volatility': data['volatility']
    }
    for sym, data in signals.items()
])
signals_df.to_csv('live_signals.csv', index=False)
print("✓ Signals exported successfully")

print("\n" + "=" * 80)
print(" " * 20 + "Ready for trading! Good luck! 🚀")
print("=" * 80 + "\n")
