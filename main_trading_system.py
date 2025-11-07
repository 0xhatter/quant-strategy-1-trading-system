"""
Main Trading System
Complete end-to-end quantitative trading pipeline orchestrator.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from data_collection import HyperliquidDataCollector
from asset_selection import AssetSelector
from feature_engineering import FeatureEngineer
from ml_models import TradingNN, TradingDataset, ModelTrainer, SharpeLoss
from risk_management import RiskManager
from backtesting import Backtest


class QuantTradingSystem:
    """
    Complete quantitative trading system that orchestrates all components.
    """
    
    def __init__(self, initial_capital: float = 10000,
                 assets_to_select: int = 3,
                 model_type: str = 'neural_network',
                 use_synthetic_data: bool = True):
        """
        Initialize the trading system.
        
        Args:
            initial_capital: Starting capital
            assets_to_select: Number of top assets to trade
            model_type: Type of model ('neural_network', 'ensemble')
            use_synthetic_data: Whether to use synthetic data for testing
        """
        self.initial_capital = initial_capital
        self.assets_to_select = assets_to_select
        self.model_type = model_type
        self.use_synthetic_data = use_synthetic_data
        
        # Initialize components
        self.data_collector = HyperliquidDataCollector(use_synthetic=use_synthetic_data)
        self.asset_selector = AssetSelector(self.data_collector)
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager(total_capital=initial_capital)
        
        self.models = {}
        self.results = {}
    
    def run_full_pipeline(self, lookback_hours: int = 720,
                         train_epochs: int = 50) -> Dict:
        """
        Run the complete trading pipeline.
        
        Args:
            lookback_hours: Hours of historical data to use
            train_epochs: Number of training epochs
            
        Returns:
            Dictionary with results for all assets
        """
        print("\n" + "=" * 70)
        print(" " * 20 + "QUANT TRADING SYSTEM")
        print("=" * 70)
        
        # Step 1: Asset Selection
        print("\n[STEP 1/5] ASSET SELECTION")
        print("-" * 70)
        top_assets_df = self.asset_selector.select_top_assets(
            n=self.assets_to_select,
            lookback_hours=lookback_hours
        )
        selected_symbols = top_assets_df.head(self.assets_to_select)['symbol'].tolist()
        
        # Step 2: Data Collection
        print("\n[STEP 2/5] DATA COLLECTION")
        print("-" * 70)
        asset_data = self.asset_selector.get_asset_data_for_selected(
            selected_symbols,
            lookback_hours=lookback_hours
        )
        
        # Process each asset
        all_results = {}
        
        for symbol in selected_symbols:
            print(f"\n{'='*70}")
            print(f"Processing {symbol}")
            print(f"{'='*70}")
            
            df = asset_data[symbol]
            
            # Step 3: Feature Engineering
            print(f"\n[STEP 3/5] FEATURE ENGINEERING - {symbol}")
            print("-" * 70)
            df_features = self.feature_engineer.create_all_features(df)
            train_df, val_df, test_df = self.feature_engineer.prepare_ml_dataset(df_features)
            
            # Step 4: Model Training
            print(f"\n[STEP 4/5] MODEL TRAINING - {symbol}")
            print("-" * 70)
            
            # Prepare data loaders
            train_dataset = TradingDataset(
                train_df[self.feature_engineer.feature_columns].values,
                train_df['target'].values
            )
            val_dataset = TradingDataset(
                val_df[self.feature_engineer.feature_columns].values,
                val_df['target'].values
            )
            test_dataset = TradingDataset(
                test_df[self.feature_engineer.feature_columns].values,
                test_df['target'].values
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Create and train model
            model = TradingNN(
                input_size=len(self.feature_engineer.feature_columns),
                hidden_sizes=[128, 64, 32],
                dropout_rate=0.3
            )
            
            trainer = ModelTrainer(model, SharpeLoss(), learning_rate=0.001)
            history = trainer.train(
                train_loader, val_loader,
                epochs=train_epochs,
                early_stopping_patience=15
            )
            
            # Save training plot
            trainer.plot_training_history(f'training_history_{symbol}.png')
            
            # Generate predictions
            predictions = trainer.predict(test_loader)
            
            # Step 5: Backtesting
            print(f"\n[STEP 5/5] BACKTESTING - {symbol}")
            print("-" * 70)
            
            backtest = Backtest(
                initial_capital=self.initial_capital,
                risk_manager=RiskManager(total_capital=self.initial_capital)
            )
            
            equity_curve = backtest.run_backtest(
                test_df, predictions, symbol=symbol, signal_threshold=0.1
            )
            
            backtest.print_performance_summary()
            backtest.plot_results(f'backtest_results_{symbol}.png')
            
            # Store results
            metrics = backtest.calculate_performance_metrics()
            all_results[symbol] = {
                'metrics': metrics,
                'equity_curve': equity_curve,
                'predictions': predictions,
                'model': model,
                'trainer': trainer,
                'backtest': backtest
            }
        
        # Summary across all assets
        self._print_portfolio_summary(all_results)
        
        self.results = all_results
        return all_results
    
    def _print_portfolio_summary(self, results: Dict):
        """Print summary across all assets."""
        print("\n" + "=" * 70)
        print(" " * 20 + "PORTFOLIO SUMMARY")
        print("=" * 70)
        
        summary_data = []
        for symbol, result in results.items():
            metrics = result['metrics']
            summary_data.append({
                'Symbol': symbol,
                'Return': f"{metrics['total_return']*100:.2f}%",
                'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                'Max DD': f"{metrics['max_drawdown']*100:.2f}%",
                'Win Rate': f"{metrics['win_rate']*100:.2f}%",
                'Trades': metrics['total_trades']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Portfolio-level metrics
        total_return = np.mean([r['metrics']['total_return'] for r in results.values()])
        avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in results.values()])
        
        print(f"\nPORTFOLIO METRICS:")
        print(f"  Average Return:      {total_return*100:>8.2f}%")
        print(f"  Average Sharpe:      {avg_sharpe:>8.2f}")
        print(f"  Number of Assets:    {len(results):>8}")
        
        print("\n" + "=" * 70)
    
    def generate_live_signals(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Generate live trading signals for current market conditions.
        
        Args:
            symbols: List of symbols (uses selected assets if None)
            
        Returns:
            Dictionary with signals for each symbol
        """
        if symbols is None:
            symbols = list(self.results.keys())
        
        signals = {}
        
        for symbol in symbols:
            if symbol not in self.results:
                print(f"No model available for {symbol}")
                continue
            
            # Fetch latest data
            df = self.data_collector.get_ohlcv(symbol, interval='1h', lookback_hours=200)
            df = self.data_collector.calculate_variance_metrics(df)
            df_features = self.feature_engineer.create_all_features(df)
            
            # Get latest features
            latest_features = df_features[self.feature_engineer.feature_columns].iloc[-1:].values
            latest_features = torch.FloatTensor(latest_features)
            
            # Generate signal
            model = self.results[symbol]['model']
            model.eval()
            with torch.no_grad():
                signal = model(latest_features).item()
            
            # Get current price and volatility
            current_price = df['close'].iloc[-1]
            volatility = df['volatility_24h'].iloc[-1]
            
            # Calculate position
            position = self.risk_manager.calculate_position_size(
                symbol=symbol,
                signal_strength=signal,
                current_price=current_price,
                volatility=volatility
            )
            
            signals[symbol] = {
                'signal': signal,
                'current_price': current_price,
                'volatility': volatility,
                'position': position,
                'recommendation': self._get_recommendation(signal)
            }
        
        return signals
    
    @staticmethod
    def _get_recommendation(signal: float) -> str:
        """Get trading recommendation from signal."""
        if signal > 0.3:
            return "STRONG BUY"
        elif signal > 0.1:
            return "BUY"
        elif signal < -0.3:
            return "STRONG SELL"
        elif signal < -0.1:
            return "SELL"
        else:
            return "HOLD"
    
    def save_models(self, path_prefix: str = 'model'):
        """Save trained models."""
        for symbol, result in self.results.items():
            model_path = f"{path_prefix}_{symbol}.pth"
            torch.save(result['model'].state_dict(), model_path)
            print(f"Model for {symbol} saved to {model_path}")
    
    def load_models(self, symbols: List[str], path_prefix: str = 'model'):
        """Load trained models."""
        for symbol in symbols:
            model_path = f"{path_prefix}_{symbol}.pth"
            model = TradingNN(
                input_size=len(self.feature_engineer.feature_columns),
                hidden_sizes=[128, 64, 32]
            )
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            if symbol not in self.results:
                self.results[symbol] = {}
            self.results[symbol]['model'] = model
            print(f"Model for {symbol} loaded from {model_path}")


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 15 + "MAIN TRADING SYSTEM TEST")
    print("=" * 70)
    
    # Initialize system
    system = QuantTradingSystem(
        initial_capital=10000,
        assets_to_select=3,
        use_synthetic_data=True
    )
    
    # Run full pipeline
    results = system.run_full_pipeline(
        lookback_hours=720,
        train_epochs=30  # Reduced for testing
    )
    
    # Generate live signals
    print("\n" + "=" * 70)
    print(" " * 20 + "LIVE SIGNALS")
    print("=" * 70)
    
    signals = system.generate_live_signals()
    
    for symbol, signal_data in signals.items():
        print(f"\n{symbol}:")
        print(f"  Signal Strength:   {signal_data['signal']:>8.4f}")
        print(f"  Recommendation:    {signal_data['recommendation']:>12s}")
        print(f"  Current Price:     ${signal_data['current_price']:>8.2f}")
        print(f"  Position Size:     ${signal_data['position']['position_value']:>8.2f}")
        print(f"  Leverage:          {signal_data['position']['leverage']:>8.2f}x")
    
    print("\n" + "=" * 70)
    print(" " * 15 + "MAIN TRADING SYSTEM TEST COMPLETE")
    print("=" * 70)
