"""
Volatility Analysis: Close-to-Close Standard Deviation
Calculate daily 1-sigma volatility for all tokens using 12 months of data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from data_collection import HyperliquidDataCollector
from scipy import stats

# All tokens to analyze
ALL_TOKENS = [
    'BTC', 'ETH', 'SOL', 'AVAX', 'ARB', 'OP', 'ATOM', 'DOT',
    'LINK', 'HYPE', 'ASTER', 'WIF', 'JTO', 'JUP', 'PYTH', 'SEI', 'SUI', 'XPL'
]

def calculate_close_to_close_volatility(prices: np.ndarray) -> float:
    """
    Calculate close-to-close volatility (standard deviation of log returns)

    Args:
        prices: Array of closing prices

    Returns:
        Daily volatility (1-sigma)
    """
    # Calculate log returns: ln(P_t / P_{t-1})
    log_returns = np.log(prices[1:] / prices[:-1])

    # Calculate standard deviation (1-sigma)
    volatility = np.std(log_returns, ddof=1)

    return volatility


def fetch_daily_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch daily data for a token"""
    collector = HyperliquidDataCollector(use_synthetic=False)

    # Fetch hourly data and resample to daily
    # Need more hours to get full year: 365 days * 24 hours = 8760 hours
    hours_needed = days * 24
    df = collector.get_ohlcv(symbol, interval='1h', lookback_hours=hours_needed)

    if df is None or len(df) < 24:
        return None

    # Resample to daily data (using close price at end of each day)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Resample to daily, taking the last close of each day
    daily_df = df['close'].resample('D').last().dropna()

    return daily_df


print("="*80)
print("VOLATILITY ANALYSIS: CLOSE-TO-CLOSE STANDARD DEVIATION")
print("="*80)
print(f"\nFetching up to 12 months of daily data for {len(ALL_TOKENS)} tokens...")
print(f"Method: Close-to-Close Estimator (σ = std of log returns)")
print()

results = []

for symbol in ALL_TOKENS:
    print(f"Processing {symbol}...", end=" ")

    try:
        daily_prices = fetch_daily_data(symbol, days=365)

        if daily_prices is None or len(daily_prices) < 30:
            print(f"✗ Insufficient data")
            continue

        # Calculate close-to-close volatility
        prices = daily_prices.values
        volatility = calculate_close_to_close_volatility(prices)

        # Calculate additional statistics
        log_returns = np.log(prices[1:] / prices[:-1])
        mean_return = np.mean(log_returns)
        skewness = stats.skew(log_returns)
        kurtosis = stats.kurtosis(log_returns)

        results.append({
            'symbol': symbol,
            'volatility': volatility,
            'volatility_pct': volatility * 100,  # Convert to percentage
            'annualized_vol': volatility * np.sqrt(252),  # Annualized (252 trading days)
            'annualized_vol_pct': volatility * np.sqrt(252) * 100,
            'mean_return': mean_return,
            'days': len(prices),
            'skewness': skewness,
            'kurtosis': kurtosis
        })

        print(f"✓ σ = {volatility*100:.3f}% ({len(prices)} days)")

    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")

print(f"\n{'='*80}")
print(f"Successfully analyzed {len(results)}/{len(ALL_TOKENS)} tokens")
print(f"{'='*80}\n")

# Convert to DataFrame and sort by volatility (highest to lowest)
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('volatility', ascending=False).reset_index(drop=True)

# Calculate Z-scores of volatilities
df_results['volatility_zscore'] = stats.zscore(df_results['volatility'])

# Display results
print(f"{'='*80}")
print("DAILY VOLATILITY RANKINGS (Close-to-Close Method)")
print(f"{'='*80}\n")
print(f"{'Rank':<6} {'Symbol':<8} {'Daily σ (%)':<12} {'Annual σ (%)':<14} {'Z-Score':<10} {'Days':<8}")
print("-"*80)

for idx, row in df_results.iterrows():
    print(f"{idx+1:<6} {row['symbol']:<8} {row['volatility_pct']:>10.3f}% "
          f"{row['annualized_vol_pct']:>12.2f}% {row['volatility_zscore']:>9.2f} {row['days']:>7d}")

print()

# Statistical summary
print(f"{'='*80}")
print("STATISTICAL SUMMARY")
print(f"{'='*80}\n")
print(f"Mean Daily Volatility:    {df_results['volatility_pct'].mean():.3f}%")
print(f"Median Daily Volatility:  {df_results['volatility_pct'].median():.3f}%")
print(f"Std Dev of Volatilities:  {df_results['volatility_pct'].std():.3f}%")
print(f"Min Volatility:           {df_results['volatility_pct'].min():.3f}% ({df_results.iloc[-1]['symbol']})")
print(f"Max Volatility:           {df_results['volatility_pct'].max():.3f}% ({df_results.iloc[0]['symbol']})")
print()

# Z-score interpretation
print(f"{'='*80}")
print("Z-SCORE INTERPRETATION")
print(f"{'='*80}\n")
print("Z-Score > +2.0:  Extremely high volatility (outlier)")
print("Z-Score > +1.0:  High volatility")
print("Z-Score -1 to +1: Normal volatility range")
print("Z-Score < -1.0:  Low volatility")
print("Z-Score < -2.0:  Extremely low volatility (outlier)")
print()

# Identify outliers
high_vol_outliers = df_results[df_results['volatility_zscore'] > 2.0]
low_vol_outliers = df_results[df_results['volatility_zscore'] < -2.0]

if len(high_vol_outliers) > 0:
    print("High Volatility Outliers (Z > +2.0):")
    for _, row in high_vol_outliers.iterrows():
        print(f"  {row['symbol']}: {row['volatility_pct']:.3f}% (Z={row['volatility_zscore']:.2f})")
    print()

if len(low_vol_outliers) > 0:
    print("Low Volatility Outliers (Z < -2.0):")
    for _, row in low_vol_outliers.iterrows():
        print(f"  {row['symbol']}: {row['volatility_pct']:.3f}% (Z={row['volatility_zscore']:.2f})")
    print()

# Create visualizations
print("Creating visualizations...")

# Figure 1: Bar chart of volatilities
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Daily Volatility
ax = axes[0, 0]
colors = ['red' if z > 2 else 'orange' if z > 1 else 'green' if z < -1 else 'steelblue'
          for z in df_results['volatility_zscore']]
bars = ax.barh(df_results['symbol'], df_results['volatility_pct'], color=colors, edgecolor='black')
ax.set_xlabel('Daily Volatility (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Token', fontsize=12, fontweight='bold')
ax.set_title('Daily Volatility (1-Sigma) - Close-to-Close Method', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()  # Highest at top

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, df_results['volatility_pct'])):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
            va='center', fontsize=9)

# Plot 2: Annualized Volatility
ax = axes[0, 1]
bars = ax.barh(df_results['symbol'], df_results['annualized_vol_pct'], color=colors, edgecolor='black')
ax.set_xlabel('Annualized Volatility (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Token', fontsize=12, fontweight='bold')
ax.set_title('Annualized Volatility (252 Trading Days)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, df_results['annualized_vol_pct'])):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
            va='center', fontsize=9)

# Plot 3: Z-Scores
ax = axes[1, 0]
colors_z = ['red' if z > 2 else 'orange' if z > 1 else 'lightcoral' if z > 0 else 'lightgreen' if z > -1 else 'green'
            for z in df_results['volatility_zscore']]
bars = ax.barh(df_results['symbol'], df_results['volatility_zscore'], color=colors_z, edgecolor='black')
ax.axvline(0, color='black', linewidth=1, linestyle='-')
ax.axvline(1, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='Z=±1')
ax.axvline(-1, color='orange', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(2, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Z=±2')
ax.axvline(-2, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax.set_xlabel('Z-Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Token', fontsize=12, fontweight='bold')
ax.set_title('Volatility Z-Scores (Standardized)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.legend(loc='lower right')
ax.invert_yaxis()

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, df_results['volatility_zscore'])):
    x_pos = val + (0.1 if val > 0 else -0.1)
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
            va='center', ha=ha, fontsize=9)

# Plot 4: Volatility Distribution
ax = axes[1, 1]
ax.hist(df_results['volatility_pct'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(df_results['volatility_pct'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"Mean: {df_results['volatility_pct'].mean():.2f}%")
ax.axvline(df_results['volatility_pct'].median(), color='green', linestyle='--', linewidth=2,
           label=f"Median: {df_results['volatility_pct'].median():.2f}%")
ax.set_xlabel('Daily Volatility (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Daily Volatilities', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('volatility_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: volatility_analysis.png")

# Save detailed results to CSV
df_results.to_csv('volatility_results.csv', index=False)
print("✓ Saved: volatility_results.csv")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}\n")
print("Key Insights:")
print(f"1. Highest Volatility: {df_results.iloc[0]['symbol']} ({df_results.iloc[0]['volatility_pct']:.3f}%)")
print(f"2. Lowest Volatility:  {df_results.iloc[-1]['symbol']} ({df_results.iloc[-1]['volatility_pct']:.3f}%)")
print(f"3. Volatility Range:   {df_results['volatility_pct'].max() - df_results['volatility_pct'].min():.3f}%")
print(f"4. Average Z-Score:    {df_results['volatility_zscore'].mean():.3f} (should be ~0)")
print()
