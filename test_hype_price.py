#!/usr/bin/env python3
"""
Test HYPE price data from Hyperliquid API
"""

from data_collection import HyperliquidDataCollector

print("=" * 70)
print(" " * 20 + "TESTING HYPE PRICE DATA")
print("=" * 70)

# Initialize collector with real API
collector = HyperliquidDataCollector(use_synthetic=False)

# Test fetching HYPE data
print("\nFetching HYPE data from Hyperliquid API...")
print("-" * 70)

try:
    # Fetch 24 hours of 1h candles for HYPE
    df = collector.get_ohlcv('HYPE', interval='1h', lookback_hours=24)

    print(f"\n✓ Successfully fetched {len(df)} candles")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
    print(f"  Latest open: ${df['open'].iloc[-1]:.2f}")
    print(f"  Latest high: ${df['high'].iloc[-1]:.2f}")
    print(f"  Latest low: ${df['low'].iloc[-1]:.2f}")
    print(f"  Latest volume: {df['volume'].iloc[-1]:.2f}")

    print("\nLast 5 candles:")
    print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail().to_string())

    print("\n" + "=" * 70)
    if 35 <= df['close'].iloc[-1] <= 45:
        print(" " * 20 + "HYPE PRICE VERIFIED ✓")
        print(f" " * 15 + f"Current price ~${df['close'].iloc[-1]:.2f} (expected ~$40)")
    else:
        print(" " * 20 + "HYPE PRICE MAY BE INCORRECT ⚠️")
        print(f" " * 15 + f"Got ${df['close'].iloc[-1]:.2f}, expected ~$40")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ Error fetching HYPE data: {e}")
    print("\n" + "=" * 70)
    print(" " * 20 + "HYPE DATA TEST FAILED ✗")
    print("=" * 70)
    raise
