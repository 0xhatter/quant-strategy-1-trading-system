# Quantitative Trading System

A complete machine learning-based quantitative trading system optimized for Sharpe ratio with comprehensive risk management and backtesting capabilities.

## 🚀 Quick Start

```bash
# Install dependencies
pip install pandas numpy matplotlib torch scikit-learn requests

# Run the example workflow
python example_workflow.py
```

This will generate synthetic market data, engineer 100+ features, train an ML model, run a backtest, and create visualizations.

## 📋 Features

### Core Capabilities
- **Data Collection**: Fetch OHLCV data with variance, volume, and fee metrics
- **Asset Selection**: Rank and select top trading assets based on multiple criteria
- **Feature Engineering**: Generate 100+ technical indicators and features
- **ML Models**: Neural networks with custom Sharpe ratio loss function
- **Risk Management**: Kelly Criterion, volatility-based sizing, position limits
- **Backtesting**: Realistic backtesting with transaction costs and slippage

### Key Highlights
- ✅ Custom Sharpe loss function for risk-adjusted returns
- ✅ Comprehensive feature engineering (price, volume, momentum, volatility)
- ✅ Advanced risk management with position sizing and stop losses
- ✅ Realistic backtesting with commissions and slippage
- ✅ Portfolio-level risk monitoring
- ✅ Automated asset selection and ranking
- ✅ Visualization of training and backtest results

## 📁 Project Structure

```
├── data_collection.py       # Data fetching and metrics calculation
├── asset_selection.py        # Asset ranking and selection
├── feature_engineering.py    # Technical feature creation
├── ml_models.py             # ML models with custom Sharpe loss
├── risk_management.py       # Position sizing and risk control
├── backtesting.py           # Realistic backtesting engine
├── main_trading_system.py   # Complete pipeline orchestrator
├── example_workflow.py      # Step-by-step tutorial
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🎯 Usage Examples

### 1. Data Collection

```python
from data_collection import HyperliquidDataCollector

collector = HyperliquidDataCollector()
df = collector.get_ohlcv('BTC', interval='1h', lookback_hours=720)
df = collector.calculate_variance_metrics(df)
```

### 2. Asset Selection

```python
from asset_selection import AssetSelector

selector = AssetSelector()
top_assets = selector.select_top_assets(n=3)
```

### 3. Feature Engineering

```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_all_features(df)
train, val, test = engineer.prepare_ml_dataset(df_features)
```

### 4. Model Training

```python
from ml_models import TradingNN, ModelTrainer, SharpeLoss

model = TradingNN(input_size=n_features)
trainer = ModelTrainer(model, SharpeLoss())
history = trainer.train(train_loader, val_loader)
```

### 5. Risk Management

```python
from risk_management import RiskManager

risk_manager = RiskManager(total_capital=10000)
position = risk_manager.calculate_position_size(
    symbol='BTC',
    signal_strength=0.8,
    current_price=45000,
    volatility=0.60
)
```

### 6. Backtesting

```python
from backtesting import Backtest

backtest = Backtest(initial_capital=10000)
equity_curve = backtest.run_backtest(test_df, predictions)
metrics = backtest.calculate_performance_metrics()
```

### 7. Complete Pipeline

```python
from main_trading_system import QuantTradingSystem

system = QuantTradingSystem(
    initial_capital=10000,
    assets_to_select=3
)
results = system.run_full_pipeline()
```

## 🔧 Configuration

### Risk Parameters

Adjust in `risk_management.py` or when initializing `RiskManager`:

```python
risk_manager = RiskManager(
    total_capital=10000,
    max_position_size=0.20,      # Max 20% per position
    max_leverage=10.0,            # Max 10x leverage
    max_portfolio_risk=0.15       # Max 15% portfolio risk
)
```

### Model Architecture

Modify in `ml_models.py`:

```python
model = TradingNN(
    input_size=n_features,
    hidden_sizes=[256, 128, 64, 32],  # Deeper network
    dropout_rate=0.4                   # More regularization
)
```

### Backtesting Costs

Adjust in `backtesting.py`:

```python
backtest = Backtest(
    initial_capital=10000,
    commission_rate=0.0004,  # 0.04% commission
    slippage_rate=0.0005     # 0.05% slippage
)
```

## 📊 Performance Metrics

The system calculates comprehensive performance metrics:

### Returns
- Total Return
- Annualized Return

### Risk-Adjusted
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside risk)
- Calmar Ratio (return/max drawdown)

### Risk
- Volatility (annualized)
- Maximum Drawdown
- Portfolio Risk

### Trade Statistics
- Total Trades
- Win Rate
- Profit Factor
- Average Win/Loss

## 🎓 Learning Path

### Beginner (5 minutes)
```bash
python example_workflow.py
```

### Intermediate (30 minutes)
1. Read through `example_workflow.py`
2. Understand each component
3. Modify parameters and observe changes

### Advanced (2+ hours)
1. Study each module in detail
2. Implement custom features
3. Experiment with different models
4. Optimize parameters
5. Add new strategies

## ⚙️ Advanced Features

### Custom Loss Functions

The system uses a custom Sharpe ratio loss function that directly optimizes for risk-adjusted returns:

```python
class SharpeLoss(nn.Module):
    def forward(self, predictions, targets):
        strategy_returns = predictions * targets
        sharpe = mean(strategy_returns) / std(strategy_returns)
        return -sharpe  # Minimize negative Sharpe
```

### Feature Engineering

100+ features including:
- **Price**: Returns, moving averages, momentum
- **Volatility**: Historical vol, Parkinson vol, ATR
- **Volume**: Volume ratios, OBV, volume momentum
- **Momentum**: RSI, MACD, Stochastic, Williams %R
- **Trend**: ADX, Bollinger Bands
- **Statistical**: Skewness, kurtosis, autocorrelation

### Position Sizing

Multiple methods combined:
- **Volatility-based**: Target portfolio volatility
- **Kelly Criterion**: Optimal fraction based on win rate
- **Signal-adjusted**: Scale by signal strength
- **Risk limits**: Maximum position and portfolio risk

## 🚨 Important Notes

### For Testing
- Examples use **synthetic data** for demonstration
- Real market data requires API credentials
- Always paper trade before using real money

### Risk Warnings
- Past performance doesn't guarantee future results
- Markets are unpredictable and can be volatile
- Never risk more than you can afford to lose
- This is educational software, not financial advice

### Best Practices
1. Start with paper trading
2. Test thoroughly on historical data
3. Monitor performance continuously
4. Adjust risk parameters conservatively
5. Keep learning and adapting

## 🔍 Troubleshooting

### Installation Issues
```bash
# If torch installation fails, try:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For M1/M2 Macs:
pip install torch
```

### Model Not Converging
- Increase training epochs
- Adjust learning rate
- Add more data
- Reduce model complexity
- Check feature scaling

### Poor Backtest Performance
- Try different assets
- Adjust signal threshold
- Modify risk parameters
- Use ensemble models
- Increase feature set

## 📈 Example Results

Typical performance on synthetic data:

```
RETURNS:
  Total Return:      25.3%
  
RISK-ADJUSTED METRICS:
  Sharpe Ratio:      2.1
  Max Drawdown:      12.5%
  
TRADE STATISTICS:
  Win Rate:          57.2%
  Profit Factor:     1.9
```

## 🛠️ Extending the System

### Add New Features

```python
def create_custom_features(self, df):
    df['my_indicator'] = ...
    return df
```

### Add New Models

```python
class CustomModel(nn.Module):
    def __init__(self):
        # Your architecture
        pass
```

### Add New Strategies

```python
def custom_signal_generator(predictions, threshold):
    # Your logic
    return signals
```

## 📚 Further Reading

- [Quantitative Trading Strategies](https://www.quantstart.com/)
- [Machine Learning for Trading](https://www.mltrading.io/)
- [Risk Management Techniques](https://www.investopedia.com/risk-management/)
- [Technical Analysis](https://www.investopedia.com/technical-analysis/)

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different strategies
- Add new features and indicators
- Improve model architectures
- Enhance risk management
- Share your findings

## 📄 License

This project is for educational purposes only. Use at your own risk.

## ⚠️ Disclaimer

This software is provided for educational and research purposes only. It is not intended to be used for actual trading without proper testing, validation, and risk management. The authors are not responsible for any financial losses incurred through the use of this software.

Trading cryptocurrencies and other financial instruments carries a high level of risk and may not be suitable for all investors. Past performance is not indicative of future results.

---

**Happy Trading! 🚀**

For questions or issues, please review the code documentation and examples.
