#!/usr/bin/env python3
"""
Quick Trading Signals - Using Real Hyperliquid Data
"""

from data_collection import HyperliquidDataCollector
from feature_engineering import FeatureEngineer
from ml_models import TradingNN
import torch
import pandas as pd
from datetime import datetime

print("=" * 80)
print(" " * 25 + "LIVE TRADING SIGNALS")
print(" " * 20 + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)

# Initialize components
collector = HyperliquidDataCollector(use_synthetic=False)
engineer = FeatureEngineer()

# Assets we trained on
assets = ['BTC', 'DOT', 'BONK', 'MATIC', 'PEPE']

# Also check HYPE and ASTER
assets_to_check = ['BTC', 'DOT', 'HYPE', 'ASTER']

print("\nFetching current market data...")
print("-" * 80)

signals_data = []

for symbol in assets_to_check:
    try:
        # Fetch recent data
        df = collector.get_ohlcv(symbol, interval='1h', lookback_hours=200)
        df = collector.calculate_variance_metrics(df)

        # Current price and stats
        current_price = df['close'].iloc[-1]
        volatility_24h = df['volatility_24h'].iloc[-1] if 'volatility_24h' in df.columns else 0.0

        # Recent price change
        price_1h_ago = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_4h_ago = df['close'].iloc[-5] if len(df) > 4 else current_price
        price_24h_ago = df['close'].iloc[-25] if len(df) > 24 else current_price

        change_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100
        change_4h = ((current_price - price_4h_ago) / price_4h_ago) * 100
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100

        # Volume analysis
        current_volume = df['volume'].iloc[-1]
        avg_volume_24h = df['volume'].tail(24).mean()
        volume_ratio = current_volume / avg_volume_24h if avg_volume_24h > 0 else 1.0

        # Momentum indicators
        if len(df) > 14:
            # Simple RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
        else:
            current_rsi = 50

        # Generate recommendation based on momentum
        if change_1h > 1 and change_4h > 2 and current_rsi < 70:
            recommendation = "🟢 STRONG BUY"
            signal_strength = min((change_4h / 10), 1.0)
        elif change_1h > 0.5 and change_4h > 1 and current_rsi < 65:
            recommendation = "🟢 BUY"
            signal_strength = min((change_4h / 20), 0.5)
        elif change_1h < -1 and change_4h < -2 and current_rsi > 30:
            recommendation = "🔴 STRONG SELL"
            signal_strength = max((change_4h / 10), -1.0)
        elif change_1h < -0.5 and change_4h < -1 and current_rsi > 35:
            recommendation = "🔴 SELL"
            signal_strength = max((change_4h / 20), -0.5)
        else:
            recommendation = "⚪ HOLD"
            signal_strength = 0.0

        # Calculate targets
        atr = ((df['high'] - df['low']).tail(14).mean())
        stop_loss = current_price - (2 * atr)
        take_profit = current_price + (3 * atr)

        signals_data.append({
            'symbol': symbol,
            'price': current_price,
            'change_1h': change_1h,
            'change_4h': change_4h,
            'change_24h': change_24h,
            'volatility': volatility_24h,
            'volume_ratio': volume_ratio,
            'rsi': current_rsi,
            'recommendation': recommendation,
            'signal_strength': signal_strength,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr
        })

        print(f"✓ {symbol:6s} - ${current_price:>10,.2f}  {recommendation}")

    except Exception as e:
        print(f"✗ {symbol:6s} - Error: {e}")

print("\n" + "=" * 80)
print(" " * 25 + "DETAILED RECOMMENDATIONS")
print("=" * 80)

for data in signals_data:
    print(f"\n{'='*80}")
    print(f"  {data['symbol']}")
    print(f"{'='*80}")

    print(f"\n  MARKET DATA:")
    print(f"    Current Price:       ${data['price']:>12,.2f}")
    print(f"    1h Change:           {data['change_1h']:>12.2f}%")
    print(f"    4h Change:           {data['change_4h']:>12.2f}%")
    print(f"    24h Change:          {data['change_24h']:>12.2f}%")
    print(f"    24h Volatility:      {data['volatility']:>12.2%}")
    print(f"    Volume vs Avg:       {data['volume_ratio']:>12.2f}x")
    print(f"    RSI(14):             {data['rsi']:>12.2f}")

    print(f"\n  SIGNAL:")
    print(f"    Recommendation:      {data['recommendation']}")
    print(f"    Signal Strength:     {data['signal_strength']:>12.4f}")

    print(f"\n  TRADE SETUP:")
    print(f"    Entry Price:         ${data['price']:>12,.2f}")
    print(f"    Stop Loss:           ${data['stop_loss']:>12,.2f} ({((data['stop_loss']/data['price'])-1)*100:>6.2f}%)")
    print(f"    Take Profit:         ${data['take_profit']:>12,.2f} (+{((data['take_profit']/data['price'])-1)*100:>5.2f}%)")
    print(f"    ATR (14):            ${data['atr']:>12,.2f}")

print("\n" + "=" * 80)
print(" " * 30 + "SUMMARY")
print("=" * 80)

buy_signals = [d for d in signals_data if 'BUY' in d['recommendation']]
sell_signals = [d for d in signals_data if 'SELL' in d['recommendation']]
hold_signals = [d for d in signals_data if 'HOLD' in d['recommendation']]

print(f"\n  Total Signals: {len(signals_data)}")
print(f"    🟢 BUY:  {len(buy_signals)}")
print(f"    🔴 SELL: {len(sell_signals)}")
print(f"    ⚪ HOLD: {len(hold_signals)}")

if buy_signals:
    buy_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
    print(f"\n  🏆 BEST BUY OPPORTUNITIES:")
    for i, sig in enumerate(buy_signals[:3], 1):
        print(f"    {i}. {sig['symbol']:6s} @ ${sig['price']:>10,.2f} - {sig['recommendation']} "
              f"(Signal: {sig['signal_strength']:.3f}, 4h: {sig['change_4h']:+.2f}%)")

if sell_signals:
    sell_signals.sort(key=lambda x: x['signal_strength'])
    print(f"\n  ⚠️  SELL/AVOID:")
    for sig in sell_signals[:3]:
        print(f"    - {sig['symbol']:6s} @ ${sig['price']:>10,.2f} - {sig['recommendation']} "
              f"(4h: {sig['change_4h']:+.2f}%)")

print("\n" + "=" * 80)
print("\n  💡 TRADING TIPS:")
print("    • RSI < 30: Oversold (potential buy)")
print("    • RSI > 70: Overbought (potential sell)")
print("    • Strong uptrend: 1h, 4h, and 24h all positive")
print("    • Volume ratio > 1.5x: Strong momentum")
print("    • Always use stop losses and proper position sizing")

print("\n" + "=" * 80)

# Save to CSV
df_signals = pd.DataFrame(signals_data)
df_signals.to_csv('current_signals.csv', index=False)
print("\n✓ Signals saved to 'current_signals.csv'")

print("\n" + "=" * 80)
print(" " * 25 + "Good luck trading! 🚀")
print("=" * 80 + "\n")
