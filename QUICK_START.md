# QUICK START GUIDE

🚀 Get Started in 5 Minutes

## Step 1: Install Dependencies

```bash
pip install pandas numpy matplotlib torch scikit-learn requests
```

## Step 2: Run the Example

```bash
python example_workflow.py
```

This will:
- Generate synthetic market data
- Engineer 100+ features
- Train an ML model with Sharpe loss
- Run a comprehensive backtest
- Generate trading signals
- Create visualizations

## Step 3: Review Results

Check the generated files:
- `training_history.png` - Model training curves
- `backtest_results.png` - Performance visualizations

---

## 📁 All Project Files

### Core System Files:
- `data_collection.py` - Data fetching and metrics calculation
- `asset_selection.py` - Asset ranking and selection
- `feature_engineering.py` - Technical feature creation
- `ml_models.py` - ML models with custom Sharpe loss
- `risk_management.py` - Position sizing and risk control
- `backtesting.py` - Realistic backtesting engine
- `main_trading_system.py` - Complete pipeline orchestrator

### Documentation Files:
- `README.md` - Comprehensive documentation
- `PROJECT_SUMMARY.md` - Project overview
- `QUICK_START.md` - This file
- `example_workflow.py` - Step-by-step tutorial

### Configuration Files:
- `requirements.txt` - Python dependencies

---

## 🎯 What Each File Does

### data_collection.py
```python
from data_collection import HyperliquidDataCollector

collector = HyperliquidDataCollector()
df = collector.get_ohlcv('BTC', interval='1h', lookback_hours=720)
df = collector.calculate_variance_metrics(df)
```
**Output**: DataFrame with OHLCV + variance/volume/fee metrics

### asset_selection.py
```python
from asset_selection import AssetSelector

selector = AssetSelector()
top_assets = selector.select_top_assets(n=3)
```
**Output**: Ranked list of top assets to trade

### feature_engineering.py
```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_all_features(df)
train, val, test = engineer.prepare_ml_dataset(df_features)
```
**Output**: Train/val/test datasets with 100+ features

### ml_models.py
```python
from ml_models import TradingNN, ModelTrainer, SharpeLoss

model = TradingNN(input_size=n_features)
trainer = ModelTrainer(model, SharpeLoss())
history = trainer.train(train_loader, val_loader)
```
**Output**: Trained model optimized for Sharpe ratio

### risk_management.py
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
**Output**: Position size, leverage, stop loss, take profit

### backtesting.py
```python
from backtesting import Backtest

backtest = Backtest(initial_capital=10000)
equity_curve = backtest.run_backtest(test_df, predictions, risk_manager)
metrics = backtest.calculate_performance_metrics()
```
**Output**: Performance metrics, equity curve, trade log

### main_trading_system.py
```python
from main_trading_system import QuantTradingSystem

system = QuantTradingSystem(
    initial_capital=10000,
    assets_to_select=3
)
results = system.run_full_pipeline()
```
**Output**: Complete end-to-end results

---

## 🎓 Learning Path

### Option 1: Quick Overview (5 min)
```bash
python example_workflow.py
```
Just run this and see how everything works.

### Option 2: Step-by-Step (30 min)
1. Open `example_workflow.py`
2. Read through each section
3. Run it section by section
4. Understand what each step does

### Option 3: Deep Dive (2+ hours)
1. Read `README.md` completely
2. Study each module individually
3. Run tests in each file's `__main__` section
4. Modify parameters and observe changes
5. Run full pipeline with `main_trading_system.py`

---

## 🔧 Common Tasks

### Task 1: Test Data Collection
```bash
python data_collection.py
```

### Task 2: Select Best Assets
```bash
python asset_selection.py
```

### Task 3: Train a Model
```bash
python ml_models.py
```

### Task 4: Run a Backtest
```bash
python backtesting.py
```

### Task 5: Complete Pipeline
```bash
python main_trading_system.py
```

---

## ⚙️ Customization Examples

### Change Risk Parameters
Edit `main_trading_system.py`:
```python
risk_manager = RiskManager(
    total_capital=10000,
    max_position_size=0.15,  # Change from 0.20 to 0.15
    max_leverage=5.0,        # Change from 10.0 to 5.0
    max_portfolio_risk=0.10  # Change from 0.15 to 0.10
)
```

### Select Different Assets
Edit `main_trading_system.py`:
```python
system = QuantTradingSystem(
    initial_capital=10000,
    assets_to_select=5,      # Change from 3 to 5
    model_type='ensemble'    # Change to ensemble
)
```

### Modify Model Architecture
Edit `ml_models.py` or your training script:
```python
model = TradingNN(
    input_size=n_features,
    hidden_sizes=[256, 128, 64, 32],  # Deeper network
    dropout_rate=0.4                   # More regularization
)
```

---

## 📊 Understanding Output

### Training History Plot
- **Top chart**: Loss curves (should decrease)
- **Bottom chart**: Sharpe ratio curves (should increase)
- **Goal**: Validation Sharpe > 0 and stable

### Backtest Results Plot
- **Top chart**: Equity curve (should trend up)
- **Second chart**: Drawdown (should be shallow)
- **Bottom charts**: Trade distribution, win rates, cumulative P&L

### Console Output
```
RETURNS:
  Total Return: 25.3%
  
RISK-ADJUSTED METRICS:
  Sharpe Ratio: 2.1  ← Higher is better (target: ≥2.0)
  Max Drawdown: 12.5% ← Lower is better (target: ≤15%)
  
TRADE STATISTICS:
  Win Rate: 57.2%    ← Higher is better (target: ≥55%)
  Profit Factor: 1.9 ← Higher is better (target: ≥1.8)
```

---

## ❗ Troubleshooting

### Issue: "No module named 'torch'"
```bash
pip install torch
```

### Issue: "API connection failed"
- Check internet connection
- For real Hyperliquid: verify API credentials
- For example: it uses synthetic data (no API needed)

### Issue: "Model not converging"
- Try more epochs
- Adjust learning rate
- Check data quality
- Add more regularization

### Issue: "Poor backtest performance"
- Try different assets
- Adjust risk parameters
- Modify feature set
- Use ensemble model

---

## 🎯 Success Criteria

✅ Example workflow runs without errors  
✅ Training completes and generates plots  
✅ Backtest shows positive Sharpe ratio  
✅ Generated files appear in directory  
✅ Console shows reasonable metrics  

---

## 🚨 Important Reminders

1. **Start with Examples**: Always use `example_workflow.py` first
2. **Synthetic Data**: Examples use fake data - real markets differ
3. **Paper Trade**: Test thoroughly before real money
4. **Risk Management**: Never risk more than you can afford to lose
5. **Continuous Learning**: Markets change, stay adaptable

---

## 📚 What to Read Next

1. `README.md` - Full documentation
2. `PROJECT_SUMMARY.md` - High-level overview
3. Module docstrings - Detailed API documentation
4. Example comments - Implementation details

---

## 🤝 Getting Help

If something doesn't work:
1. Check error messages carefully
2. Verify all dependencies installed
3. Try running `example_workflow.py` first
4. Review module documentation
5. Check that data is valid

---

## 🏁 Ready to Start?

```bash
# Install
pip install pandas numpy matplotlib torch scikit-learn requests

# Run
python example_workflow.py

# Enjoy!
```

**Good luck with your quant trading journey! 🚀**

---

**Time to First Result**: ~5 minutes  
**Time to Understanding**: ~30 minutes  
**Time to Mastery**: Keep learning!
