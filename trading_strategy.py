"""
Trading Strategies - Complete Version with Multiple Thresholds and Realistic Trading Rules
Fixed Maximum Drawdown calculation and enhanced performance metrics with holding period constraints
"""
import numpy as np
import pandas as pd

class ForexPortfolioManager:
    """Enhanced portfolio manager with detailed trade tracking, realistic constraints, and leverage support."""
    def __init__(self, config, strategy_threshold=None):
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
        self.capital = self.initial_capital
        self.trade_history = []
        self.capital_history = [self.initial_capital]
        
        # Leverage mapping based on strategy threshold
        self.leverage_mapping = {
            'Conservative': 2.0,  # High leverage for high confidence signals
            'Moderate': 1.0,      # Standard leverage
            'Aggressive': 0.5     # Low leverage for uncertain signals
        }
        
        # Set leverage based on strategy threshold
        self.current_leverage = self.leverage_mapping.get(strategy_threshold, 1.0)
        
        # Base lot size (will be adjusted by leverage)
        self.base_lot_size = 0.1

    def execute_trade(self, trade_type, entry_price, exit_price, currency_pair, hold_hours=0, close_reason="signal"):
        """Enhanced trade execution with detailed logging and leverage application."""
        pip_value = self.config.PIP_VALUES.get(currency_pair, 10.0)
        pip_multiplier = 100 if 'JPY' in currency_pair else 10000
        pips_gained = (exit_price - entry_price) * pip_multiplier if trade_type == 'long' else (entry_price - exit_price) * pip_multiplier
        
        # Apply leverage to lot size
        effective_lot_size = self.base_lot_size * self.current_leverage
        pnl = pips_gained * pip_value * effective_lot_size
        
        self.capital += pnl
        self.capital_history.append(self.capital)
        
        # Detailed trade record including leverage info
        self.trade_history.append({
            'pnl': pnl,
            'trade_type': trade_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pips': pips_gained,
            'hold_hours': hold_hours,
            'close_reason': close_reason,
            'leverage_used': self.current_leverage,
            'effective_lot_size': effective_lot_size
        })

    def update_capital_for_buy_and_hold(self, current_price, initial_price, currency_pair):
        """Special method for Buy & Hold strategy to track continuous performance with leverage."""
        pip_value = self.config.PIP_VALUES.get(currency_pair, 10.0)
        pip_multiplier = 100 if 'JPY' in currency_pair else 10000
        effective_lot_size = self.base_lot_size * self.current_leverage
        
        # Calculate unrealized P&L with leverage
        pips_gained = (current_price - initial_price) * pip_multiplier
        unrealized_pnl = pips_gained * pip_value * effective_lot_size
        current_capital = self.initial_capital + unrealized_pnl
        
        self.capital_history.append(current_capital)

    def get_performance_metrics(self):
        """Enhanced performance calculation including proper Maximum Drawdown and trading statistics."""
        if not self.capital_history or len(self.capital_history) < 2:
            return {
                'total_return_pct': 0, 'sharpe_ratio': 0, 'win_rate': 0, 
                'total_trades': 0, 'max_drawdown_pct': 0,
                'avg_hold_hours': 0, 'stop_loss_rate': 0, 'take_profit_rate': 0
            }
        
        # Calculate total return
        total_return_pct = ((self.capital_history[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate returns series for Sharpe ratio
        capital_series = pd.Series(self.capital_history)
        returns = capital_series.pct_change().dropna()
        
        # Enhanced Sharpe calculation
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # Annualized
        else:
            # For Buy & Hold or strategies with minimal variance, use simpler calculation
            if len(self.capital_history) >= 2:
                total_period_return = (self.capital_history[-1] - self.capital_history[0]) / self.capital_history[0]
                # Assume monthly evaluation period for annualization
                periods_per_year = 12
                annualized_return = total_period_return * periods_per_year
                # Use a conservative estimate for volatility if we can't calculate it
                estimated_volatility = abs(total_period_return) * 2  # Conservative estimate
                sharpe = annualized_return / estimated_volatility if estimated_volatility > 0 else 0
            else:
                sharpe = 0
        
        # Win rate calculation
        if self.trade_history:
            win_rate = sum(1 for t in self.trade_history if t['pnl'] > 0) / len(self.trade_history)
        else:
            # For Buy & Hold, win rate is 1 if profitable, 0 if not
            win_rate = 1.0 if total_return_pct > 0 else 0.0
        
        # Fixed Maximum Drawdown calculation
        peak = capital_series.expanding().max()
        drawdown = (capital_series - peak) / peak * 100
        max_drawdown_pct = abs(drawdown.min()) if len(drawdown) > 0 and not drawdown.isna().all() else 0
        
        # Enhanced trading statistics including leverage info
        if self.trade_history:
            avg_hold_hours = np.mean([t.get('hold_hours', 0) for t in self.trade_history])
            stop_loss_rate = sum(1 for t in self.trade_history if t.get('close_reason') == 'stop_loss') / len(self.trade_history)
            take_profit_rate = sum(1 for t in self.trade_history if t.get('close_reason') == 'take_profit') / len(self.trade_history)
            avg_leverage = np.mean([t.get('leverage_used', 1.0) for t in self.trade_history])
        else:
            avg_hold_hours = 0
            stop_loss_rate = 0
            take_profit_rate = 0
            avg_leverage = self.current_leverage
        
        return {
            'total_return_pct': total_return_pct, 
            'sharpe_ratio': sharpe, 
            'win_rate': win_rate, 
            'total_trades': len(self.trade_history),
            'max_drawdown_pct': max_drawdown_pct,
            'avg_hold_hours': avg_hold_hours,
            'stop_loss_rate': stop_loss_rate,
            'take_profit_rate': take_profit_rate,
            'avg_leverage': avg_leverage
        }

class TradingSimulator:
    """Enhanced simulator with realistic trading constraints, proper Buy & Hold handling, and leverage support."""
    def __init__(self, config, prices, timestamps=None, strategy_threshold=None):
        self.config = config
        self.prices = prices.values if isinstance(prices, pd.Series) else prices
        self.timestamps = timestamps if timestamps is not None else range(len(self.prices))
        self.strategy_threshold = strategy_threshold
        self.portfolio = ForexPortfolioManager(config, strategy_threshold)
        
        # Trading constraints (configurable via config or use defaults)
        self.MIN_HOLD_HOURS = getattr(config, 'MIN_HOLD_HOURS', 1)
        self.MAX_HOLD_HOURS = getattr(config, 'MAX_HOLD_HOURS', 3)
        self.STOP_LOSS_PCT = getattr(config, 'STOP_LOSS_PCT', 2.0)  # 2%
        self.TAKE_PROFIT_AFTER_HOURS = getattr(config, 'TAKE_PROFIT_AFTER_HOURS', 1)

    def run(self, signals):
        """Enhanced simulation with realistic trading rules and proper Buy & Hold handling."""
        # Check if this is a Buy & Hold strategy
        if self._is_buy_and_hold_strategy(signals):
            return self._run_buy_and_hold(signals)
        else:
            return self._run_realistic_strategy(signals)
    
    def _is_buy_and_hold_strategy(self, signals):
        """Detect if this is a Buy & Hold strategy."""
        # Buy & Hold typically has signal=1 at start and 0 elsewhere
        non_zero_signals = np.count_nonzero(signals)
        return non_zero_signals <= 2 and len(signals) > 0 and signals[0] == 1
    
    def _run_buy_and_hold(self, signals):
        """Special handling for Buy & Hold strategy with continuous performance tracking."""
        if len(self.prices) == 0:
            return self.portfolio.get_performance_metrics()
        
        initial_price = self.prices[0]
        
        # Track performance continuously throughout the period
        for i, current_price in enumerate(self.prices):
            if i > 0:  # Skip first price as it's the entry point
                self.portfolio.update_capital_for_buy_and_hold(
                    current_price, initial_price, self.config.TARGET_PAIR
                )
        
        # Create a single trade record for bookkeeping with leverage
        if len(self.prices) > 1:
            final_price = self.prices[-1]
            pip_value = self.config.PIP_VALUES.get(self.config.TARGET_PAIR, 10.0)
            pip_multiplier = 100 if 'JPY' in self.config.TARGET_PAIR else 10000
            effective_lot_size = self.portfolio.base_lot_size * self.portfolio.current_leverage
            
            pips_gained = (final_price - initial_price) * pip_multiplier
            pnl = pips_gained * pip_value * effective_lot_size
            self.portfolio.trade_history.append({
                'pnl': pnl,
                'trade_type': 'long',
                'entry_price': initial_price,
                'exit_price': final_price,
                'pips': pips_gained,
                'hold_hours': len(self.prices) - 1,
                'close_reason': 'strategy_end',
                'leverage_used': self.portfolio.current_leverage,
                'effective_lot_size': effective_lot_size
            })
        
        return self.portfolio.get_performance_metrics()
    
    def _run_realistic_strategy(self, signals):
        """Realistic strategy simulation with proper holding period constraints."""
        position = 0
        entry_price = 0
        entry_time = 0
        
        for i in range(len(self.prices)):
            current_price = self.prices[i]
            
            # Check if we have an open position
            if position != 0:
                hold_hours = i - entry_time
                
                # Calculate current P&L percentage
                if position == 1:  # Long position
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # Short position
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
                # Check exit conditions (in order of priority)
                should_exit = False
                exit_reason = ""
                
                # 1. Stop Loss (immediate, highest priority)
                if pnl_pct <= -self.STOP_LOSS_PCT:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # 2. Take Profit after minimum hold period
                elif hold_hours >= self.TAKE_PROFIT_AFTER_HOURS and pnl_pct > 0:
                    should_exit = True
                    exit_reason = "take_profit"
                
                # 3. Maximum hold period reached
                elif hold_hours >= self.MAX_HOLD_HOURS:
                    should_exit = True
                    exit_reason = "max_hold"
                
                # 4. Signal change (only after minimum hold period)
                elif hold_hours >= self.MIN_HOLD_HOURS and signals[i] != position:
                    should_exit = True
                    exit_reason = "signal"
                
                # Execute exit if conditions met
                if should_exit:
                    self.portfolio.execute_trade(
                        'long' if position == 1 else 'short',
                        entry_price,
                        current_price,
                        self.config.TARGET_PAIR,
                        hold_hours=hold_hours,
                        close_reason=exit_reason
                    )
                    position = 0
            
            # Check for new position entry (only if no current position)
            if position == 0 and signals[i] != 0:
                position = signals[i]
                entry_price = current_price
                entry_time = i
        
        # Close any remaining position at the end
        if position != 0:
            hold_hours = len(self.prices) - 1 - entry_time
            self.portfolio.execute_trade(
                'long' if position == 1 else 'short',
                entry_price,
                self.prices[-1],
                self.config.TARGET_PAIR,
                hold_hours=hold_hours,
                close_reason="strategy_end"
            )
        
        return self.portfolio.get_performance_metrics()

# --- Enhanced Signal Generation Functions ---

def get_cnn_lstm_signals_multiple_thresholds(config, predictions):
    """Generates signals from model predictions for all thresholds with correct buy/sell zones."""
    results = {}
    
    for threshold_name, thresholds in config.THRESHOLDS.items():
        signals = np.zeros_like(predictions)
        buy_threshold = thresholds['buy']
        sell_threshold = thresholds['sell']
        
        # Correct signal generation
        signals[predictions >= buy_threshold] = 1   # Buy signal
        signals[predictions <= sell_threshold] = -1  # Sell signal
        # Values between sell_threshold and buy_threshold remain 0 (Hold)
        
        results[threshold_name] = signals
    
    return results

def get_cnn_lstm_signals(config, predictions):
    """Generates signals from model predictions using Aggressive threshold for backward compatibility."""
    signals = np.zeros_like(predictions)
    thresholds = config.THRESHOLDS['Aggressive']
    buy_threshold = thresholds['buy']
    sell_threshold = thresholds['sell']
    
    signals[predictions >= buy_threshold] = 1
    signals[predictions <= sell_threshold] = -1
    return signals

def get_rsi_signals(config, rsi_series):
    """Generates signals from an RSI series."""
    signals = np.zeros_like(rsi_series)
    signals[rsi_series <= config.RSI_OVERSOLD] = 1
    signals[rsi_series >= config.RSI_OVERBOUGHT] = -1
    return signals

def get_macd_signals(config, macd_series, signal_series):
    """Generates signals from MACD crossover."""
    signals = np.zeros_like(macd_series)
    signals[macd_series > signal_series] = 1
    signals[macd_series < signal_series] = -1
    return signals

def get_buy_and_hold_signals(prices):
    """Enhanced Buy & Hold signal generation."""
    signals = np.zeros_like(prices)
    if len(signals) > 0:
        signals[0] = 1  # Buy at the start and hold
    return signals