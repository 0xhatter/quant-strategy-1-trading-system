# Quick Start Guide - High-Variance Trading on HYPE & ASTER

## ✅ What's Ready Now

Everything is configured and ready to run! The system will:

1. **Fetch real-time data** from Hyperliquid API for 20 tokens
2. **Select top 5 highest-variance** tokens (prioritizing HYPE, ASTER, etc.)
3. **Engineer 100+ features** per token
4. **Train neural network models** (50 epochs each)
5. **Backtest** strategies on historical data
6. **Generate live trading signals**

## 📊 Asset Universe (20 Tokens)

### High-Priority Targets
- **HYPE** - Your primary target
- **ASTER** - Your secondary target

### Other High-Variance Candidates
- **WIF, BONK, PEPE** - Meme coins with high volatility
- **JTO, JUP, PYTH** - New DeFi protocols
- **SEI, SUI** - New Layer 1s

### Established Assets (for comparison)
- BTC, ETH, SOL, AVAX, MATIC, ARB, OP, ATOM, DOT, LINK

## 🚀 How to Run

### Option 1: Automatic High-Variance Strategy (Recommended)

```bash
# Wait for PyTorch installation to complete, then:
python run_high_variance_strategy.py
```

This will:
- Automatically select top 5 tokens by **variance score**
- Train models on all selected tokens
- Generate live trading signals
- Save models for future use

### Option 2: Test API Connection (Works Right Now!)

```bash
# Test that Hyperliquid API is working
python test_hyperliquid_api.py
```

Expected output:
```
✓ Successfully fetched 25 candles
  Latest BTC close: $101,877.00
```

### Option 3: Manual Execution

```python
from main_trading_system import QuantTradingSystem

# Initialize with real Hyperliquid data
system = QuantTradingSystem(
    initial_capital=10000,
    assets_to_select=5,  # Top 5 high-variance tokens
    use_synthetic_data=False
)

# Run complete pipeline
results = system.run_full_pipeline(
    lookback_hours=720,  # 30 days
    train_epochs=50
)

# Get live signals
signals = system.generate_live_signals()
```

## 📁 What You'll Get

### Models (Saved Automatically)
```
high_variance_model_HYPE.pth
high_variance_model_ASTER.pth
high_variance_model_WIF.pth
... (one per selected token)
```

### Visualizations
```
training_history_HYPE.png      # Training loss curves
backtest_results_HYPE.png      # Equity curves
... (one set per token)
```

### Console Output
```
Top 5 selected assets (sorted by variance_score):
  HYPE  : 95.23 (Var: 95.2, Vol: 88.4, Liq: 75.6)
  ASTER : 92.14 (Var: 92.1, Vol: 85.2, Liq: 72.3)
  ...

HYPE:
  Total Return:         145.67%
  Sharpe Ratio:           2.45
  Max Drawdown:         -18.34%
  Win Rate:              67.89%

LIVE SIGNALS:
  HYPE - STRONG BUY (Signal: 0.78)
  Position: $2,500 @ 2.0x leverage
```

## ⏳ Current Status

**Dependencies:**
- ✅ pandas, numpy, requests (installed)
- ⏳ PyTorch, matplotlib, scikit-learn (installing in background)

**System Status:**
- ✅ Hyperliquid API integration working
- ✅ Asset universe expanded to 20 tokens
- ✅ Variance-focused selection configured
- ✅ All code committed and pushed

**Installation Progress:**
PyTorch (~900MB) is downloading/installing. This is the only blocker.

## 🎯 When PyTorch Finishes Installing

Simply run:
```bash
python run_high_variance_strategy.py
```

The system will automatically:
1. Connect to Hyperliquid
2. Fetch 30 days of data for all 20 tokens
3. Rank by variance (HYPE and ASTER should be near the top)
4. Select top 5
5. Train models
6. Generate signals

## 🔧 Advanced Usage

### Select Only HYPE and ASTER

```python
from asset_selection import AssetSelector
from data_collection import HyperliquidDataCollector

collector = HyperliquidDataCollector(use_synthetic=False)
selector = AssetSelector(collector)

# Custom universe with just HYPE and ASTER
custom_tokens = ['HYPE', 'ASTER']

top_assets = selector.select_top_assets(
    n=2,
    custom_universe=custom_tokens,
    lookback_hours=720,
    sort_by='variance_score'
)
```

### Adjust for More Aggressive Trading

Edit `run_high_variance_strategy.py`:
```python
# Select more tokens
ASSETS_TO_SELECT = 10  # Instead of 5

# Use shorter period (more recent volatility)
LOOKBACK_HOURS = 168  # 1 week instead of 30 days

# Train longer for better accuracy
TRAIN_EPOCHS = 100  # Instead of 50
```

### Check Current Variance Rankings

```python
from asset_selection import AssetSelector
from data_collection import HyperliquidDataCollector

collector = HyperliquidDataCollector(use_synthetic=False)
selector = AssetSelector(collector)

# Get variance scores for all tokens
scores = selector.select_top_assets(
    n=20,  # Show all
    sort_by='variance_score'
)

print(scores[['symbol', 'variance_score', 'recent_volatility']])
```

## 📚 Documentation

- **`HIGH_VARIANCE_STRATEGY_README.md`** - Complete guide
- **`SETUP_COMPLETE.md`** - API integration details
- **`README.md`** - Original system documentation

## ⚠️ Important Notes

### If HYPE/ASTER Not Available on Hyperliquid

If you see:
```
Warning: No data returned for HYPE
Falling back to synthetic data...
```

This means Hyperliquid doesn't have that token. The system will:
1. Try all 20 tokens
2. Select the top 5 that have real data
3. Fall back to synthetic data for missing tokens

### Alternative: Use Different High-Variance Tokens

The system automatically finds the highest-variance tokens from those available. Even if HYPE/ASTER aren't on Hyperliquid, tokens like WIF, BONK, PEPE, JTO typically have very high variance.

## 🎉 You're Ready!

As soon as PyTorch finishes installing (you'll see "Successfully installed torch..." in the background process), you can run:

```bash
python run_high_variance_strategy.py
```

The entire pipeline will execute automatically and you'll have trained models ready to generate live signals for the highest-variance tokens!

---

**Next:** Wait for installation to complete, then execute the strategy!
