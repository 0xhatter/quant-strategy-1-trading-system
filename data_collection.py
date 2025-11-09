"""
Data Collection Module
Handles data fetching and metric calculations for the trading system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import requests


class HyperliquidDataCollector:
    """
    Collects and processes market data with variance, volume, and fee metrics.
    Can work with both real Hyperliquid API and synthetic data for testing.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_synthetic: bool = True):
        """
        Initialize the data collector.
        
        Args:
            api_key: API key for Hyperliquid (if using real data)
            use_synthetic: If True, generates synthetic data for testing
        """
        self.api_key = api_key
        self.use_synthetic = use_synthetic
        self.base_url = "https://api.hyperliquid.xyz"
        
    def get_ohlcv(self, symbol: str, interval: str = '1h', 
                  lookback_hours: int = 720) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            interval: Time interval ('1h', '4h', '1d')
            lookback_hours: Number of hours to look back
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if self.use_synthetic:
            return self._generate_synthetic_ohlcv(symbol, interval, lookback_hours)
        else:
            return self._fetch_real_ohlcv(symbol, interval, lookback_hours)
    
    def _generate_synthetic_ohlcv(self, symbol: str, interval: str, 
                                   lookback_hours: int) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        # Determine number of candles based on interval
        interval_hours = self._interval_to_hours(interval)
        n_candles = int(lookback_hours / interval_hours)
        
        # Generate timestamps
        end_time = datetime.now()
        timestamps = [end_time - timedelta(hours=interval_hours * i) 
                     for i in range(n_candles)]
        timestamps.reverse()
        
        # Base price for different symbols
        base_prices = {
            'BTC': 45000,
            'ETH': 2500,
            'SOL': 100,
            'AVAX': 35,
            'MATIC': 0.8
        }
        base_price = base_prices.get(symbol, 1000)
        
        # Generate price data with realistic characteristics
        np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
        
        # Trend component
        trend = np.linspace(0, np.random.randn() * 0.2, n_candles)
        
        # Random walk component
        returns = np.random.randn(n_candles) * 0.02
        log_prices = np.cumsum(returns) + trend
        close_prices = base_price * np.exp(log_prices)
        
        # Generate OHLC from close prices
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, close_prices)):
            # Add intrabar volatility
            volatility = abs(np.random.randn()) * 0.01
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            
            # Open is previous close (or base price for first candle)
            if i == 0:
                open_price = base_price
            else:
                open_price = close_prices[i-1]
            
            # Volume with some randomness
            volume = abs(np.random.randn() * 1000000 + 5000000)
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def _fetch_real_ohlcv(self, symbol: str, interval: str,
                          lookback_hours: int) -> pd.DataFrame:
        """Fetch real OHLCV data from Hyperliquid API."""
        endpoint = f"{self.base_url}/info"

        try:
            # Calculate timestamps in milliseconds
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_hours)

            # Prepare request payload
            payload = {
                'type': 'candleSnapshot',
                'req': {
                    'coin': symbol,
                    'interval': interval,
                    'startTime': int(start_time.timestamp() * 1000),
                    'endTime': int(end_time.timestamp() * 1000)
                }
            }

            # Make API request
            print(f"Fetching {symbol} data from Hyperliquid API...")
            response = requests.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()

            # Parse response
            candles = response.json()

            if not candles or len(candles) == 0:
                print(f"Warning: No data returned for {symbol}")
                print("Falling back to synthetic data...")
                return self._generate_synthetic_ohlcv(symbol, interval, lookback_hours)

            # Convert to DataFrame with correct column mapping
            # API returns: t (timestamp), o (open), h (high), l (low), c (close), v (volume)
            data = []
            for candle in candles:
                data.append({
                    'timestamp': pd.to_datetime(int(candle['t']), unit='ms'),
                    'open': float(candle['o']),
                    'high': float(candle['h']),
                    'low': float(candle['l']),
                    'close': float(candle['c']),
                    'volume': float(candle['v'])
                })

            df = pd.DataFrame(data)
            print(f"Successfully fetched {len(df)} candles for {symbol}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            print("Falling back to synthetic data...")
            return self._generate_synthetic_ohlcv(symbol, interval, lookback_hours)
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error parsing API response: {e}")
            print("Falling back to synthetic data...")
            return self._generate_synthetic_ohlcv(symbol, interval, lookback_hours)
        except Exception as e:
            print(f"Unexpected error fetching real data: {e}")
            print("Falling back to synthetic data...")
            return self._generate_synthetic_ohlcv(symbol, interval, lookback_hours)
    
    def calculate_variance_metrics(self, df: pd.DataFrame, 
                                   windows: List[int] = [24, 48, 168]) -> pd.DataFrame:
        """
        Calculate variance, volume, and fee metrics.
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of lookback windows in periods
            
        Returns:
            DataFrame with additional metric columns
        """
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate variance metrics for different windows
        for window in windows:
            # Price variance
            df[f'variance_{window}h'] = df['returns'].rolling(window).var()
            
            # Realized volatility
            df[f'volatility_{window}h'] = df['returns'].rolling(window).std() * np.sqrt(24)
            
            # Volume metrics
            df[f'volume_ma_{window}h'] = df['volume'].rolling(window).mean()
            df[f'volume_std_{window}h'] = df['volume'].rolling(window).std()
            df[f'volume_ratio_{window}h'] = df['volume'] / df[f'volume_ma_{window}h']
            
            # Price range metrics
            df[f'high_low_range_{window}h'] = (
                (df['high'] - df['low']) / df['close']
            ).rolling(window).mean()
            
        # Estimated trading fees (as percentage of volume)
        df['estimated_fee_rate'] = 0.0002  # 0.02% typical maker fee
        df['estimated_fees'] = df['volume'] * df['close'] * df['estimated_fee_rate']
        
        # Volume-weighted metrics
        df['vwap'] = (df['volume'] * df['close']).rolling(24).sum() / df['volume'].rolling(24).sum()
        df['price_vwap_ratio'] = df['close'] / df['vwap']
        
        return df
    
    def get_multiple_assets(self, symbols: List[str], interval: str = '1h',
                           lookback_hours: int = 720) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple assets.
        
        Args:
            symbols: List of trading symbols
            interval: Time interval
            lookback_hours: Number of hours to look back
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        data = {}
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            df = self.get_ohlcv(symbol, interval, lookback_hours)
            df = self.calculate_variance_metrics(df)
            data[symbol] = df
        return data
    
    @staticmethod
    def _interval_to_hours(interval: str) -> float:
        """Convert interval string to hours."""
        if interval.endswith('h'):
            return float(interval[:-1])
        elif interval.endswith('d'):
            return float(interval[:-1]) * 24
        elif interval.endswith('m'):
            return float(interval[:-1]) / 60
        else:
            return 1.0


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("DATA COLLECTION MODULE TEST")
    print("=" * 60)
    
    # Initialize collector with real Hyperliquid API data
    collector = HyperliquidDataCollector(use_synthetic=False)
    
    # Test single asset
    print("\n1. Fetching BTC data...")
    btc_df = collector.get_ohlcv('BTC', interval='1h', lookback_hours=720)
    print(f"   Retrieved {len(btc_df)} candles")
    print(f"   Date range: {btc_df['timestamp'].min()} to {btc_df['timestamp'].max()}")
    print(f"   Price range: ${btc_df['close'].min():.2f} to ${btc_df['close'].max():.2f}")
    
    # Calculate metrics
    print("\n2. Calculating variance metrics...")
    btc_df = collector.calculate_variance_metrics(btc_df)
    print(f"   Added {len(btc_df.columns) - 6} metric columns")
    print(f"   Columns: {list(btc_df.columns)}")
    
    # Show sample data
    print("\n3. Sample data (last 5 rows):")
    print(btc_df[['timestamp', 'close', 'volume', 'variance_24h', 'volatility_24h']].tail())
    
    # Test multiple assets
    print("\n4. Fetching multiple assets...")
    symbols = ['BTC', 'ETH', 'SOL']
    multi_data = collector.get_multiple_assets(symbols, interval='1h', lookback_hours=168)
    
    for symbol, df in multi_data.items():
        print(f"   {symbol}: {len(df)} candles, "
              f"avg price: ${df['close'].mean():.2f}, "
              f"avg volatility: {df['volatility_24h'].mean():.4f}")
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION TEST COMPLETE")
    print("=" * 60)
