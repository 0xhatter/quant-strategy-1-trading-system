"""
Risk Management Module
Handles position sizing, leverage, stop losses, and portfolio risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class RiskManager:
    """
    Manages risk for trading positions.
    Implements Kelly Criterion, volatility-based sizing, and risk limits.
    """
    
    def __init__(self, total_capital: float = 10000,
                 max_position_size: float = 0.20,
                 max_leverage: float = 10.0,
                 max_portfolio_risk: float = 0.15,
                 risk_free_rate: float = 0.0):
        """
        Initialize risk manager.
        
        Args:
            total_capital: Total trading capital
            max_position_size: Maximum position size as fraction of capital
            max_leverage: Maximum leverage allowed
            max_portfolio_risk: Maximum portfolio risk (fraction of capital)
            risk_free_rate: Risk-free rate for calculations
        """
        self.total_capital = total_capital
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_portfolio_risk = max_portfolio_risk
        self.risk_free_rate = risk_free_rate
        
        self.current_positions = {}
        self.position_history = []
    
    def calculate_position_size(self, symbol: str, signal_strength: float,
                               current_price: float, volatility: float,
                               win_rate: Optional[float] = None,
                               avg_win: Optional[float] = None,
                               avg_loss: Optional[float] = None) -> Dict:
        """
        Calculate optimal position size using multiple methods.
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength from model (-1 to 1)
            current_price: Current asset price
            volatility: Asset volatility (annualized)
            win_rate: Historical win rate (optional, for Kelly)
            avg_win: Average win size (optional, for Kelly)
            avg_loss: Average loss size (optional, for Kelly)
            
        Returns:
            Dictionary with position details
        """
        # 1. Volatility-based sizing
        target_volatility = 0.15  # Target 15% portfolio volatility
        vol_based_size = (target_volatility / volatility) * self.total_capital
        
        # 2. Kelly Criterion (if statistics available)
        if win_rate and avg_win and avg_loss:
            kelly_fraction = self._calculate_kelly(win_rate, avg_win, avg_loss)
            kelly_size = kelly_fraction * self.total_capital
        else:
            kelly_size = vol_based_size
        
        # 3. Signal-based adjustment
        signal_adjusted_size = abs(signal_strength) * min(vol_based_size, kelly_size)
        
        # 4. Apply maximum position size limit
        max_size = self.max_position_size * self.total_capital
        position_value = min(signal_adjusted_size, max_size)
        
        # Calculate number of units
        num_units = position_value / current_price
        
        # Determine leverage needed
        leverage = min(position_value / self.total_capital, self.max_leverage)
        
        # Calculate stop loss and take profit
        stop_loss_pct = 2 * volatility / np.sqrt(252)  # 2-day volatility
        take_profit_pct = 3 * volatility / np.sqrt(252)  # 3-day volatility
        
        if signal_strength > 0:  # Long position
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:  # Short position
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        
        position = {
            'symbol': symbol,
            'direction': 'long' if signal_strength > 0 else 'short',
            'signal_strength': signal_strength,
            'position_value': position_value,
            'num_units': num_units,
            'leverage': leverage,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': abs(current_price - stop_loss) * num_units,
            'volatility': volatility
        }
        
        return position
    
    def _calculate_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion fraction.
        
        Args:
            win_rate: Probability of winning
            avg_win: Average win size (as fraction)
            avg_loss: Average loss size (as fraction)
            
        Returns:
            Kelly fraction (capped at 0.25 for safety)
        """
        if avg_loss == 0:
            return 0.0
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        
        # Use half-Kelly for safety
        kelly = kelly * 0.5
        
        # Cap at 25% of capital
        return max(0, min(kelly, 0.25))
    
    def check_portfolio_risk(self, new_position: Dict) -> Tuple[bool, str]:
        """
        Check if adding a new position exceeds portfolio risk limits.
        
        Args:
            new_position: Position dictionary from calculate_position_size
            
        Returns:
            Tuple of (is_acceptable, reason)
        """
        # Calculate current portfolio risk
        current_risk = sum(
            pos.get('risk_amount', 0) for pos in self.current_positions.values()
        )
        
        # Add new position risk
        total_risk = current_risk + new_position['risk_amount']
        portfolio_risk_pct = total_risk / self.total_capital
        
        if portfolio_risk_pct > self.max_portfolio_risk:
            return False, f"Portfolio risk {portfolio_risk_pct:.2%} exceeds limit {self.max_portfolio_risk:.2%}"
        
        # Check position concentration
        position_pct = new_position['position_value'] / self.total_capital
        if position_pct > self.max_position_size:
            return False, f"Position size {position_pct:.2%} exceeds limit {self.max_position_size:.2%}"
        
        # Check leverage
        if new_position['leverage'] > self.max_leverage:
            return False, f"Leverage {new_position['leverage']:.1f}x exceeds limit {self.max_leverage:.1f}x"
        
        return True, "Position acceptable"
    
    def add_position(self, position: Dict) -> bool:
        """
        Add a position to the portfolio.
        
        Args:
            position: Position dictionary
            
        Returns:
            True if position was added, False otherwise
        """
        is_acceptable, reason = self.check_portfolio_risk(position)
        
        if is_acceptable:
            self.current_positions[position['symbol']] = position
            self.position_history.append({
                'timestamp': pd.Timestamp.now(),
                'action': 'open',
                **position
            })
            return True
        else:
            print(f"Position rejected: {reason}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, reason: str = 'signal'):
        """
        Close a position.
        
        Args:
            symbol: Symbol to close
            exit_price: Exit price
            reason: Reason for closing ('signal', 'stop_loss', 'take_profit')
        """
        if symbol in self.current_positions:
            position = self.current_positions[symbol]
            
            # Calculate P&L
            if position['direction'] == 'long':
                pnl = (exit_price - position['entry_price']) * position['num_units']
            else:
                pnl = (position['entry_price'] - exit_price) * position['num_units']
            
            # Update capital
            self.total_capital += pnl
            
            # Record closure
            self.position_history.append({
                'timestamp': pd.Timestamp.now(),
                'action': 'close',
                'symbol': symbol,
                'exit_price': exit_price,
                'pnl': pnl,
                'reason': reason
            })
            
            # Remove from current positions
            del self.current_positions[symbol]
    
    def update_stops(self, symbol: str, current_price: float, 
                    trailing_stop_pct: float = 0.02):
        """
        Update trailing stops for a position.
        
        Args:
            symbol: Symbol to update
            current_price: Current price
            trailing_stop_pct: Trailing stop percentage
        """
        if symbol not in self.current_positions:
            return
        
        position = self.current_positions[symbol]
        
        if position['direction'] == 'long':
            # Update stop loss if price moved up
            new_stop = current_price * (1 - trailing_stop_pct)
            if new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
        else:
            # Update stop loss if price moved down
            new_stop = current_price * (1 + trailing_stop_pct)
            if new_stop < position['stop_loss']:
                position['stop_loss'] = new_stop
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        total_position_value = sum(
            pos['position_value'] for pos in self.current_positions.values()
        )
        total_risk = sum(
            pos['risk_amount'] for pos in self.current_positions.values()
        )
        
        return {
            'total_capital': self.total_capital,
            'num_positions': len(self.current_positions),
            'total_position_value': total_position_value,
            'total_risk': total_risk,
            'portfolio_risk_pct': total_risk / self.total_capital,
            'capital_utilization': total_position_value / self.total_capital
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("RISK MANAGEMENT MODULE TEST")
    print("=" * 60)
    
    # Initialize risk manager
    risk_manager = RiskManager(
        total_capital=10000,
        max_position_size=0.20,
        max_leverage=10.0,
        max_portfolio_risk=0.15
    )
    
    # Test position sizing
    print("\n1. Calculating position size for BTC...")
    position = risk_manager.calculate_position_size(
        symbol='BTC',
        signal_strength=0.8,
        current_price=45000,
        volatility=0.60,
        win_rate=0.55,
        avg_win=0.03,
        avg_loss=0.02
    )
    
    for key, value in position.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Test risk check
    print("\n2. Checking portfolio risk...")
    is_acceptable, reason = risk_manager.check_portfolio_risk(position)
    print(f"   Acceptable: {is_acceptable}")
    print(f"   Reason: {reason}")
    
    # Add position
    print("\n3. Adding position...")
    added = risk_manager.add_position(position)
    print(f"   Position added: {added}")
    
    # Portfolio summary
    print("\n4. Portfolio summary:")
    summary = risk_manager.get_portfolio_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("RISK MANAGEMENT TEST COMPLETE")
    print("=" * 60)
