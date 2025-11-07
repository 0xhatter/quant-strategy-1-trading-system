# 🚀 GET STARTED

Welcome to your Quantitative Trading System! Follow these steps to get up and running.

## ⚡ Quick Setup (5 minutes)

### Step 1: Install Dependencies

Choose one of these methods:

#### Option A: Automatic Installation (Recommended)
```bash
./install_dependencies.sh
```

#### Option B: Manual Installation
```bash
pip install pandas numpy matplotlib torch scikit-learn requests
```

#### Option C: Using requirements.txt
```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup

```bash
python3 verify_setup.py
```

You should see:
```
✅ All dependencies are installed!
✅ All project files are present!
✅ All project modules can be imported!
🎉 SETUP VERIFICATION COMPLETE! 🎉
```

### Step 3: Run Your First Example

```bash
python3 example_workflow.py
```

This will:
- ✓ Generate synthetic market data
- ✓ Select top trading assets
- ✓ Engineer 100+ features
- ✓ Train ML model with Sharpe loss
- ✓ Run comprehensive backtest
- ✓ Generate visualizations

**Expected runtime**: 3-5 minutes

---

## 📊 What You'll See

### Console Output

```
======================================================================
                  QUANTITATIVE TRADING SYSTEM
                      EXAMPLE WORKFLOW
======================================================================

EXAMPLE 1: DATA COLLECTION
======================================================================
Fetching BTC data...
Retrieved 720 hourly candles
Calculating metrics...

EXAMPLE 2: ASSET SELECTION
======================================================================
Analyzing 10 assets...
BTC   : Composite Score = 78.45
ETH   : Composite Score = 72.31
...

EXAMPLE 3: FEATURE ENGINEERING
======================================================================
Creating features...
  ✓ Price features
  ✓ Volatility features
  ✓ Volume features
  ✓ Momentum indicators
  ✓ Trend indicators
  ✓ Statistical features

Total features created: 150

EXAMPLE 4: MODEL TRAINING
======================================================================
Training on cpu
Epoch 10/30
  Train Loss: -0.8234, Train Sharpe: 0.8234
  Val Loss:   -0.7891, Val Sharpe:   0.7891
...

EXAMPLE 5: RISK MANAGEMENT
======================================================================
Calculating position for BTC (strong buy signal)...
Position details:
  direction           : long
  position_value      : 1500.0000
  leverage            : 1.5000
  stop_loss           : 44100.0000
  take_profit         : 45900.0000

EXAMPLE 6: BACKTESTING
======================================================================
Running backtest for BTC...
Backtest complete!
  Total trades: 45
  Final capital: $12,530.00
  Total return: 25.30%

BACKTEST PERFORMANCE SUMMARY
======================================================================
RETURNS:
  Total Return:      25.30%
  Annualized Return: 42.15%

RISK-ADJUSTED METRICS:
  Sharpe Ratio:      2.10
  Sortino Ratio:     2.85
  Max Drawdown:      12.50%

TRADE STATISTICS:
  Total Trades:      45
  Win Rate:          57.20%
  Profit Factor:     1.90

EXAMPLE 7: COMPLETE PIPELINE
======================================================================
[Running full pipeline for 3 assets...]
```

### Generated Files

After running, you'll find these files:

1. **training_history.png** - Model training curves
   - Loss over epochs
   - Sharpe ratio over epochs

2. **backtest_results.png** - Backtest performance
   - Equity curve
   - Drawdown chart
   - Trade P&L distribution
   - Cumulative P&L

3. **training_history_BTC.png** - Per-asset training
4. **backtest_results_BTC.png** - Per-asset backtest
5. **best_model.pth** - Trained model weights

---

## 🎯 Next Steps

### 1. Explore Individual Modules

Test each component separately:

```bash
# Test data collection
python3 data_collection.py

# Test asset selection
python3 asset_selection.py

# Test feature engineering
python3 feature_engineering.py

# Test ML models
python3 ml_models.py

# Test risk management
python3 risk_management.py

# Test backtesting
python3 backtesting.py

# Run complete system
python3 main_trading_system.py
```

### 2. Customize Parameters

Edit the files to adjust:

**Risk Settings** (in `main_trading_system.py`):
```python
risk_manager = RiskManager(
    total_capital=10000,      # Your capital
    max_position_size=0.20,   # Max 20% per position
    max_leverage=10.0,        # Max 10x leverage
    max_portfolio_risk=0.15   # Max 15% portfolio risk
)
```

**Model Architecture** (in `ml_models.py`):
```python
model = TradingNN(
    input_size=n_features,
    hidden_sizes=[128, 64, 32],  # Network layers
    dropout_rate=0.3              # Regularization
)
```

**Trading Parameters** (in `backtesting.py`):
```python
backtest = Backtest(
    initial_capital=10000,
    commission_rate=0.0004,  # 0.04% per trade
    slippage_rate=0.0005     # 0.05% slippage
)
```

### 3. Read Documentation

- **README.md** - Comprehensive guide with all features
- **PROJECT_SUMMARY.md** - Technical overview and architecture
- **QUICK_START.md** - Quick reference guide

### 4. Experiment

Try different:
- Assets (BTC, ETH, SOL, etc.)
- Time periods (hours, days)
- Model architectures
- Risk parameters
- Feature combinations

---

## 🔧 Troubleshooting

### Issue: Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
pip install torch
# or
./install_dependencies.sh
```

### Issue: Import Errors

**Error**: `ImportError: cannot import name 'HyperliquidDataCollector'`

**Solution**: Make sure you're in the project directory:
```bash
cd "/Users/nitankkhatter/QUANT STRATEGY 1"
python3 example_workflow.py
```

### Issue: Model Not Training

**Symptoms**: Loss not decreasing, Sharpe ratio stays at 0

**Solutions**:
- Increase training epochs (30 → 50)
- Adjust learning rate (0.001 → 0.0001)
- Check data quality
- Reduce model complexity

### Issue: Poor Backtest Results

**Symptoms**: Negative returns, low Sharpe ratio

**Solutions**:
- Try different assets
- Adjust signal threshold
- Modify risk parameters
- Increase training data
- Use more features

---

## 📚 Learning Resources

### In This Project
- Code comments in each module
- Docstrings for all functions
- Example workflow with explanations
- Comprehensive documentation

### External Resources
- [Quantitative Trading](https://www.quantstart.com/)
- [Machine Learning for Trading](https://www.mltrading.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Technical Analysis](https://www.investopedia.com/technical-analysis/)

---

## ✅ Success Checklist

- [ ] Dependencies installed (`verify_setup.py` passes)
- [ ] Example workflow runs successfully
- [ ] Training plots generated
- [ ] Backtest plots generated
- [ ] Understand console output
- [ ] Reviewed documentation
- [ ] Tested individual modules
- [ ] Customized parameters
- [ ] Ready to experiment!

---

## 🎓 Recommended Learning Path

### Week 1: Basics
- Run example workflow
- Understand each component
- Read all documentation
- Test individual modules

### Week 2: Customization
- Modify risk parameters
- Try different assets
- Adjust model architecture
- Experiment with features

### Week 3: Advanced
- Implement custom features
- Create new strategies
- Optimize parameters
- Analyze results deeply

### Week 4+: Mastery
- Build your own strategies
- Combine multiple models
- Implement advanced techniques
- Continuous improvement

---

## 🚨 Important Reminders

⚠️ **This is educational software**
- Uses synthetic data for examples
- Not financial advice
- Test thoroughly before real trading
- Manage risk carefully

⚠️ **Risk Management**
- Never risk more than you can afford to lose
- Start with paper trading
- Use proper position sizing
- Always use stop losses

⚠️ **Continuous Learning**
- Markets change constantly
- Keep updating your knowledge
- Monitor performance
- Adapt strategies

---

## 🎉 You're Ready!

Your quantitative trading system is set up and ready to use.

**Start exploring**:
```bash
python3 example_workflow.py
```

**Questions?** Review the documentation:
- README.md
- PROJECT_SUMMARY.md
- QUICK_START.md

**Happy Trading! 🚀**

---

*Last updated: 2024*
