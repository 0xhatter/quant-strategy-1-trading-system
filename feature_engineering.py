"""
Feature Engineering Module
Creates technical indicators and features for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    Creates comprehensive technical features for ML models.
    Generates 100+ features from OHLCV and variance metrics.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features."""
        df = df.copy()
        
        # Returns at different horizons
        for period in [1, 2, 4, 8, 24, 48]:
            df[f'return_{period}h'] = df['close'].pct_change(period)
            df[f'log_return_{period}h'] = np.log(df['close'] / df['close'].shift(period))
        
        # Moving averages
        for window in [6, 12, 24, 48, 168]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
        
        # Exponential moving averages
        for span in [12, 26, 50]:
            df[f'ema_{span}'] = df['close'].ewm(span=span).mean()
            df[f'price_to_ema_{span}'] = df['close'] / df[f'ema_{span}']
        
        # Price momentum
        for window in [12, 24, 48]:
            df[f'momentum_{window}'] = df['close'] - df['close'].shift(window)
            df[f'roc_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features."""
        df = df.copy()
        
        # Historical volatility at different windows
        for window in [12, 24, 48, 168]:
            returns = df['close'].pct_change()
            df[f'hist_vol_{window}'] = returns.rolling(window).std() * np.sqrt(24)
            df[f'vol_of_vol_{window}'] = df[f'hist_vol_{window}'].rolling(window).std()
        
        # Parkinson volatility (uses high-low range)
        for window in [12, 24, 48]:
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (np.log(df['high'] / df['low']) ** 2).rolling(window).mean() / (4 * np.log(2))
            )
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        for window in [14, 24, 48]:
            df[f'atr_{window}'] = true_range.rolling(window).mean()
            df[f'atr_pct_{window}'] = df[f'atr_{window}'] / df['close']
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        df = df.copy()
        
        # Volume moving averages
        for window in [12, 24, 48, 168]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # Volume momentum
        for window in [12, 24]:
            df[f'volume_momentum_{window}'] = df['volume'] - df['volume'].shift(window)
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_sma_24'] = df['obv'].rolling(24).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma_24']
        
        # Volume-weighted features
        df['vwap_ratio'] = df['close'] / df['vwap']
        
        return df
    
    def create_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum indicators."""
        df = df.copy()
        
        # RSI (Relative Strength Index)
        for window in [14, 24, 48]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        for window in [14, 24]:
            low_min = df['low'].rolling(window).min()
            high_max = df['high'].rolling(window).max()
            df[f'stoch_{window}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
            df[f'stoch_{window}_sma'] = df[f'stoch_{window}'].rolling(3).mean()
        
        # Williams %R
        for window in [14, 24]:
            high_max = df['high'].rolling(window).max()
            low_min = df['low'].rolling(window).min()
            df[f'williams_r_{window}'] = -100 * (high_max - df['close']) / (high_max - low_min + 1e-8)
        
        return df
    
    def create_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend indicators."""
        df = df.copy()
        
        # ADX (Average Directional Index)
        for window in [14, 24]:
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            atr = (df['high'] - df['low']).rolling(window).mean()
            pos_di = 100 * pos_dm.rolling(window).mean() / (atr + 1e-8)
            neg_di = 100 * neg_dm.rolling(window).mean() / (atr + 1e-8)
            
            dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-8)
            df[f'adx_{window}'] = dx.rolling(window).mean()
        
        # Bollinger Bands
        for window in [20, 48]:
            sma = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df[f'bb_upper_{window}'] = sma + (2 * std)
            df[f'bb_lower_{window}'] = sma - (2 * std)
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / sma
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (
                df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'] + 1e-8
            )
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        df = df.copy()
        
        returns = df['close'].pct_change()
        
        # Rolling statistics
        for window in [24, 48, 168]:
            df[f'return_mean_{window}'] = returns.rolling(window).mean()
            df[f'return_std_{window}'] = returns.rolling(window).std()
            df[f'return_skew_{window}'] = returns.rolling(window).skew()
            df[f'return_kurt_{window}'] = returns.rolling(window).kurt()
        
        # Autocorrelation
        for lag in [1, 2, 4, 8]:
            df[f'autocorr_lag_{lag}'] = returns.rolling(48).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
            )
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features at once."""
        print("Creating features...")
        
        df = df.copy()
        df = self.create_price_features(df)
        print("  ✓ Price features")
        df = self.create_volatility_features(df)
        print("  ✓ Volatility features")
        df = self.create_volume_features(df)
        print("  ✓ Volume features")
        df = self.create_momentum_indicators(df)
        print("  ✓ Momentum indicators")
        df = self.create_trend_indicators(df)
        print("  ✓ Trend indicators")
        df = self.create_statistical_features(df)
        print("  ✓ Statistical features")
        
        # Create target variable (future return)
        df['target'] = df['close'].pct_change(4).shift(-4)
        
        print(f"\nTotal features created: {len(df.columns)}")
        return df
    
    def prepare_ml_dataset(self, df: pd.DataFrame, train_ratio: float = 0.7,
                          val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare train/val/test datasets."""
        df_clean = df.dropna()
        
        # Get feature columns (exclude OHLCV and target)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
        self.feature_columns = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Split data
        n = len(df_clean)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_clean.iloc[:train_end].copy()
        val_df = df_clean.iloc[train_end:val_end].copy()
        test_df = df_clean.iloc[val_end:].copy()
        
        # Fit scaler on training data
        self.scaler.fit(train_df[self.feature_columns])
        
        # Transform all datasets
        train_df[self.feature_columns] = self.scaler.transform(train_df[self.feature_columns])
        val_df[self.feature_columns] = self.scaler.transform(val_df[self.feature_columns])
        test_df[self.feature_columns] = self.scaler.transform(test_df[self.feature_columns])
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        print(f"  Features: {len(self.feature_columns)}")
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    from data_collection import HyperliquidDataCollector
    
    print("=" * 60)
    print("FEATURE ENGINEERING MODULE TEST")
    print("=" * 60)
    
    # Get sample data
    collector = HyperliquidDataCollector(use_synthetic=True)
    df = collector.get_ohlcv('BTC', interval='1h', lookback_hours=720)
    df = collector.calculate_variance_metrics(df)
    
    # Create features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    # Prepare ML dataset
    train_df, val_df, test_df = engineer.prepare_ml_dataset(df_features)
    
    print("\nSample features (first 5):")
    print(train_df[engineer.feature_columns[:5]].head())
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING TEST COMPLETE")
    print("=" * 60)
