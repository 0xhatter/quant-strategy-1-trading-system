#!/usr/bin/env python3
"""
Quick test script to verify Hyperliquid API connection works.
"""

from data_collection import HyperliquidDataCollector

print("=" * 70)
print(" " * 20 + "HYPERLIQUID API TEST")
print("=" * 70)

# Initialize collector with real API
collector = HyperliquidDataCollector(use_synthetic=False)

# Test fetching BTC data
print("\nTesting API connection with BTC...")
print("-" * 70)

try:
    # Fetch 24 hours of 1h candles (should be 24 candles)
    df = collector.get_ohlcv('BTC', interval='1h', lookback_hours=24)

    print(f"\n✓ Successfully fetched {len(df)} candles")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")

    print("\nSample data (last 5 candles):")
    print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail().to_string())

    print("\n" + "=" * 70)
    print(" " * 20 + "API TEST SUCCESSFUL ✓")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ API test failed: {e}")
    print("\n" + "=" * 70)
    print(" " * 20 + "API TEST FAILED ✗")
    print("=" * 70)
    raise
