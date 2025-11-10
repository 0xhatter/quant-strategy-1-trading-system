#!/usr/bin/env python3
"""
Test ASTER price data from Hyperliquid API
"""

from data_collection import HyperliquidDataCollector

print("=" * 70)
print(" " * 20 + "TESTING ASTER PRICE DATA")
print("=" * 70)

# Initialize collector with real API
collector = HyperliquidDataCollector(use_synthetic=False)

# Test fetching ASTER data
print("\nFetching ASTER data from Hyperliquid API...")
print("-" * 70)

try:
    # Fetch 24 hours of 1h candles for ASTER
    df = collector.get_ohlcv('ASTER', interval='1h', lookback_hours=24)

    print(f"\n✓ Successfully fetched {len(df)} candles")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['close'].min():.4f} to ${df['close'].max():.4f}")
    print(f"  Latest close: ${df['close'].iloc[-1]:.4f}")
    print(f"  Latest open: ${df['open'].iloc[-1]:.4f}")
    print(f"  Latest high: ${df['high'].iloc[-1]:.4f}")
    print(f"  Latest low: ${df['low'].iloc[-1]:.4f}")
    print(f"  Latest volume: {df['volume'].iloc[-1]:.2f}")

    print("\nLast 5 candles:")
    print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail().to_string())

    print("\n" + "=" * 70)
    print(" " * 20 + "ASTER DATA RETRIEVED ✓")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ Error fetching ASTER data: {e}")
    print("\n" + "=" * 70)
    print(" " * 20 + "ASTER DATA TEST FAILED ✗")
    print("=" * 70)
