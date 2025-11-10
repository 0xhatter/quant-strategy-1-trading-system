"""
Asset Selection Module
Ranks and selects the best assets to trade based on multiple criteria.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from data_collection import HyperliquidDataCollector


class AssetSelector:
    """
    Selects top trading assets based on variance, volume, and other metrics.
    """
    
    def __init__(self, data_collector: Optional[HyperliquidDataCollector] = None):
        """
        Initialize the asset selector.
        
        Args:
            data_collector: HyperliquidDataCollector instance (creates new if None)
        """
        self.data_collector = data_collector or HyperliquidDataCollector(use_synthetic=True)
        
        # Default universe of assets to consider
        # Includes high-variance tokens like HYPE, ASTER
        self.asset_universe = [
            'BTC', 'ETH', 'SOL', 'AVAX', 'MATIC',
            'ARB', 'OP', 'ATOM', 'DOT', 'LINK',
            'HYPE', 'ASTER', 'WIF', 'BONK', 'PEPE',
            'JTO', 'JUP', 'PYTH', 'SEI', 'SUI'
        ]
    
    def calculate_asset_scores(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Calculate various scoring metrics for an asset.
        
        Args:
            df: DataFrame with OHLCV and calculated metrics
            symbol: Asset symbol
            
        Returns:
            Dictionary of scores
        """
        # Remove NaN rows
        df_clean = df.dropna()
        
        if len(df_clean) < 50:
            return {
                'symbol': symbol,
                'variance_score': 0,
                'volume_score': 0,
                'trend_score': 0,
                'liquidity_score': 0,
                'composite_score': 0
            }
        
        # 1. Variance Score (higher variance = more trading opportunities)
        recent_variance = df_clean['variance_24h'].tail(168).mean()  # Last week
        variance_percentile = df_clean['variance_24h'].rank(pct=True).iloc[-1]
        variance_score = variance_percentile * 100
        
        # 2. Volume Score (higher volume = better liquidity)
        recent_volume = df_clean['volume'].tail(168).mean()
        volume_stability = 1 / (1 + df_clean['volume'].tail(168).std() / recent_volume)
        volume_score = (recent_volume / 1e7) * volume_stability * 100
        volume_score = min(volume_score, 100)  # Cap at 100
        
        # 3. Trend Score (consistent trends are good)
        returns = df_clean['returns'].tail(168)
        trend_strength = abs(returns.mean()) / (returns.std() + 1e-8)
        trend_score = min(trend_strength * 20, 100)
        
        # 4. Liquidity Score (tight spreads, good volume)
        avg_range = df_clean['high_low_range_24h'].tail(168).mean()
        liquidity_score = (1 - min(avg_range, 0.1)) * 100
        
        # 5. Volatility Score (moderate volatility is ideal)
        recent_vol = df_clean['volatility_24h'].tail(168).mean()
        # Prefer volatility between 0.3 and 1.0 (30% to 100% annualized)
        if 0.3 <= recent_vol <= 1.0:
            volatility_score = 100
        elif recent_vol < 0.3:
            volatility_score = (recent_vol / 0.3) * 100
        else:
            volatility_score = max(0, 100 - (recent_vol - 1.0) * 50)
        
        # Composite score (weighted average)
        weights = {
            'variance': 0.25,
            'volume': 0.25,
            'trend': 0.15,
            'liquidity': 0.20,
            'volatility': 0.15
        }
        
        composite_score = (
            weights['variance'] * variance_score +
            weights['volume'] * volume_score +
            weights['trend'] * trend_score +
            weights['liquidity'] * liquidity_score +
            weights['volatility'] * volatility_score
        )
        
        return {
            'symbol': symbol,
            'variance_score': round(variance_score, 2),
            'volume_score': round(volume_score, 2),
            'trend_score': round(trend_score, 2),
            'liquidity_score': round(liquidity_score, 2),
            'volatility_score': round(volatility_score, 2),
            'composite_score': round(composite_score, 2),
            'recent_volatility': round(recent_vol, 4),
            'recent_volume': round(recent_volume, 2)
        }
    
    def select_top_assets(self, n: int = 3, interval: str = '1h',
                         lookback_hours: int = 720,
                         custom_universe: Optional[List[str]] = None,
                         sort_by: str = 'composite_score') -> pd.DataFrame:
        """
        Select top N assets based on scoring.

        Args:
            n: Number of assets to select
            interval: Time interval for data
            lookback_hours: Hours of historical data
            custom_universe: Custom list of assets (uses default if None)
            sort_by: Metric to sort by ('composite_score', 'variance_score', 'volatility_score')

        Returns:
            DataFrame with ranked assets and their scores
        """
        universe = custom_universe or self.asset_universe
        
        print(f"\nAnalyzing {len(universe)} assets...")
        print("-" * 60)
        
        # Fetch data for all assets
        all_data = self.data_collector.get_multiple_assets(
            universe, interval, lookback_hours
        )
        
        # Calculate scores for each asset
        scores = []
        for symbol, df in all_data.items():
            score_dict = self.calculate_asset_scores(df, symbol)
            scores.append(score_dict)
            print(f"{symbol:6s}: Composite Score = {score_dict['composite_score']:.2f}")
        
        # Create DataFrame and sort by specified metric
        scores_df = pd.DataFrame(scores)
        scores_df = scores_df.sort_values(sort_by, ascending=False)

        print("-" * 60)
        print(f"\nTop {n} selected assets (sorted by {sort_by}):")
        top_assets = scores_df.head(n)
        for idx, row in top_assets.iterrows():
            print(f"  {row['symbol']:6s}: {row[sort_by]:.2f} "
                  f"(Var: {row['variance_score']:.1f}, "
                  f"Vol: {row['volatility_score']:.1f}, "
                  f"Liq: {row['liquidity_score']:.1f})")
        
        return scores_df
    
    def get_asset_data_for_selected(self, selected_symbols: List[str],
                                   interval: str = '1h',
                                   lookback_hours: int = 720) -> Dict[str, pd.DataFrame]:
        """
        Get full data for selected assets.
        
        Args:
            selected_symbols: List of symbols to fetch
            interval: Time interval
            lookback_hours: Hours of historical data
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        return self.data_collector.get_multiple_assets(
            selected_symbols, interval, lookback_hours
        )


from typing import Optional


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("ASSET SELECTION MODULE TEST")
    print("=" * 60)
    
    # Initialize selector
    selector = AssetSelector()
    
    # Select top 3 assets
    print("\n1. Selecting top 3 assets from universe...")
    top_assets = selector.select_top_assets(n=3, lookback_hours=720)
    
    # Display detailed scores
    print("\n2. Detailed scores for all assets:")
    print(top_assets.to_string(index=False))
    
    # Get data for selected assets
    print("\n3. Fetching full data for top 3 assets...")
    selected_symbols = top_assets.head(3)['symbol'].tolist()
    asset_data = selector.get_asset_data_for_selected(selected_symbols)
    
    for symbol, df in asset_data.items():
        print(f"\n   {symbol}:")
        print(f"     Data points: {len(df)}")
        print(f"     Avg price: ${df['close'].mean():.2f}")
        print(f"     Avg 24h volatility: {df['volatility_24h'].mean():.4f}")
        print(f"     Avg volume: {df['volume'].mean():.2e}")
    
    print("\n" + "=" * 60)
    print("ASSET SELECTION TEST COMPLETE")
    print("=" * 60)
