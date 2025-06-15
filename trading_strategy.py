"""
Trading Strategy for Multi-Currency CNN-LSTM Forex Prediction
Enhanced with Proper Forex Position Sizing and Confidence-based Lot Sizing
Fixed all DatetimeIndex issues and Sharpe Ratio calculation
"""

import numpy as np
import pandas as pd
from data_processor import DataProcessor

# Helper functions to handle both Series and DatetimeIndex
def get_timestamp_value(timestamps, index):
    """Safely get timestamp value regardless of type"""
    if hasattr(timestamps, 'iloc'):
        return timestamps.iloc[index]
    else:
        return timestamps[index]

def get_last_timestamp(timestamps):
    """Safely get last timestamp regardless of type"""
    if hasattr(timestamps, 'iloc'):
        return timestamps.iloc[-1]
    else:
        return timestamps[-1]

class ForexPortfolioManager:
    """Portfolio Manager specifically for Forex trading"""

    def __init__(self, config):
        self.config = config
        self.capital = config.INITIAL_CAPITAL
        self.initial_capital = config.INITIAL_CAPITAL
        self.positions = []
        self.trade_history = []
        self.capital_history = [config.INITIAL_CAPITAL]
        self.daily_returns = []  # For Sharpe ratio calculation
        self.timestamps = []

    def calculate_position_size(self, confidence_level='moderate', currency_pair='EURUSD'):
        """
        Calculate position size based on confidence level from config

        Args:
            confidence_level: 'conservative', 'moderate', or 'aggressive'
            currency_pair: Currency pair being traded

        Returns:
            dict with lot_size, units, max_loss, pip_value
        """
        lot_size = self.config.LOT_SIZES.get(confidence_level, 1.0)
        pip_value = self.config.PIP_VALUES.get(currency_pair, 10.0)
        max_loss = lot_size * self.config.STOP_LOSS_PIPS * pip_value
        max_allowed_loss = self.capital * self.config.RISK_PER_TRADE_PCT

        if max_loss > max_allowed_loss:
            adjusted_lot_size = max_allowed_loss / (self.config.STOP_LOSS_PIPS * pip_value)
            lot_size = min(lot_size, adjusted_lot_size)
            max_loss = lot_size * self.config.STOP_LOSS_PIPS * pip_value

        units = lot_size * 100000
        position_value = units * 1.0
        required_margin = position_value / self.config.LEVERAGE

        return {
            'lot_size': round(lot_size, 2),
            'units': int(units),
            'pip_value': pip_value,
            'max_loss': max_loss,
            'required_margin': required_margin,
            'confidence_level': confidence_level
        }

    def execute_forex_trade(self, trade_type, entry_price, exit_price,
                          timestamp, currency_pair='EURUSD',
                          confidence_level='moderate'):
        """Execute a forex trade with proper position sizing"""
        position_info = self.calculate_position_size(confidence_level, currency_pair)

        if currency_pair == 'USDJPY':
            pip_multiplier = 100
        else:
            pip_multiplier = 10000

        if trade_type == 'long':
            pips_gained = (exit_price - entry_price) * pip_multiplier
        else:  # short
            pips_gained = (entry_price - exit_price) * pip_multiplier

        pnl = pips_gained * position_info['pip_value'] * position_info['lot_size']
        self.capital += pnl
        self.capital_history.append(self.capital)

        trade_record = {
            'type': trade_type,
            'currency_pair': currency_pair,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'lot_size': position_info['lot_size'],
            'confidence_level': confidence_level,
            'units': position_info['units'],
            'pips_gained': round(pips_gained, 1),
            'pnl': pnl,
            'pnl_pct': (pnl / self.initial_capital) * 100,
            'capital_after': self.capital,
            'timestamp': timestamp
        }
        self.trade_history.append(trade_record)
        return trade_record

    def calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from capital history"""
        if len(self.capital_history) < 2:
            return 0.0

        capital_array = np.array(self.capital_history)
        returns = np.diff(capital_array) / capital_array[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        return sharpe_ratio

    def get_performance_metrics(self):
        """Calculate portfolio performance metrics"""
        if not self.trade_history:
            return {
                'total_return': 0, 'total_return_pct': 0, 'final_capital': self.initial_capital,
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
                'avg_win_pips': 0, 'avg_loss_pips': 0, 'profit_factor': 0,
                'max_capital': self.initial_capital, 'min_capital': self.initial_capital,
                'max_drawdown': 0, 'max_drawdown_pct': 0, 'total_pips': 0, 'sharpe_ratio': 0
            }

        total_return = self.capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        pnls = [t['pnl'] for t in self.trade_history]
        pips = [t['pips_gained'] for t in self.trade_history]
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl'] <= 0]
        winning_pips = [t['pips_gained'] for t in winning_trades]
        losing_pips = [t['pips_gained'] for t in losing_trades]

        capital_array = np.array(self.capital_history)
        running_max = np.maximum.accumulate(capital_array)
        drawdowns = running_max - capital_array
        max_drawdown = np.max(drawdowns)
        max_drawdown_pct = (max_drawdown / running_max[np.argmax(drawdowns)]) * 100 if len(drawdowns) > 0 else 0

        gross_profit = sum([p for p in pnls if p > 0])
        gross_loss = abs(sum([p for p in pnls if p <= 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        sharpe_ratio = self.calculate_sharpe_ratio()

        return {
            'total_return': total_return, 'total_return_pct': total_return_pct, 'final_capital': self.capital,
            'total_trades': len(self.trade_history), 'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades), 'win_rate': len(winning_trades) / len(self.trade_history) if self.trade_history else 0,
            'avg_win_pips': np.mean(winning_pips) if winning_pips else 0, 'avg_loss_pips': np.mean(losing_pips) if losing_pips else 0,
            'profit_factor': profit_factor, 'max_capital': np.max(capital_array), 'min_capital': np.min(capital_array),
            'max_drawdown': max_drawdown, 'max_drawdown_pct': max_drawdown_pct, 'total_pips': sum(pips),
            'sharpe_ratio': sharpe_ratio
        }

class FixedHoldingTradingStrategy:
    """Fixed Holding Period Trading Strategy with Forex Position Management"""
    def __init__(self, config):
        self.config = config

    def apply_strategy(self, predictions, prices, timestamps, threshold_type='moderate',
                      portfolio_manager=None, currency_pair='EURUSD', verbose=True):
        if verbose:
            print(f"📈 Applying {threshold_type} trading strategy for {currency_pair}...")
            print(f"   Lot size: {self.config.LOT_SIZES[threshold_type]} lots")

        if portfolio_manager is None:
            portfolio_manager = ForexPortfolioManager(self.config)

        thresholds = self.config.THRESHOLDS[threshold_type]
        trades, positions, signals = [], np.zeros(len(predictions)), np.zeros(len(predictions))
        current_position, entry_time, entry_price = 0, None, None

        for i, (pred, price, timestamp) in enumerate(zip(predictions, prices, timestamps)):
            if current_position != 0:
                hours_held = (timestamp - entry_time).total_seconds() / 3600
                pip_multiplier = 100 if currency_pair == 'USDJPY' else 10000
                current_pips = (price - entry_price) * pip_multiplier if current_position == 1 else (entry_price - price) * pip_multiplier

                should_close, close_reason = False, ''
                if current_pips <= -self.config.STOP_LOSS_PIPS:
                    should_close, close_reason = True, 'stop_loss'
                elif current_pips >= self.config.TAKE_PROFIT_PIPS:
                    should_close, close_reason = True, 'take_profit'
                elif hours_held >= self.config.MIN_HOLDING_HOURS:
                    if hours_held >= self.config.MAX_HOLDING_HOURS:
                        should_close, close_reason = True, 'time_limit'
                    elif current_pips > 0:
                        should_close, close_reason = True, 'profit_taking'

                if should_close:
                    trade_record = portfolio_manager.execute_forex_trade(
                        'long' if current_position == 1 else 'short', entry_price, price, timestamp,
                        currency_pair, threshold_type)
                    trades.append({
                        'type': 'long' if current_position == 1 else 'short', 'entry_time': entry_time,
                        'exit_time': timestamp, 'entry_price': entry_price, 'exit_price': price,
                        'pips_gained': trade_record['pips_gained'], 'lot_size': trade_record['lot_size'],
                        'pnl': trade_record['pnl'], 'pnl_pct': trade_record['pnl_pct'],
                        'holding_hours': hours_held, 'reason': close_reason, 'confidence_level': threshold_type
                    })
                    current_position, entry_time, entry_price = 0, None, None
                positions[i] = current_position
                continue

            if current_position == 0:
                if pred >= thresholds['buy']:
                    signals[i], current_position, entry_price, entry_time = 1, 1, price, timestamp
                elif pred <= thresholds['sell']:
                    signals[i], current_position, entry_price, entry_time = -1, -1, price, timestamp
            positions[i] = current_position

        if current_position != 0:
            final_timestamp, final_price = get_last_timestamp(timestamps), prices.iloc[-1]
            trade_record = portfolio_manager.execute_forex_trade(
                'long' if current_position == 1 else 'short', entry_price, final_price,
                final_timestamp, currency_pair, threshold_type)
            trades.append({
                'type': 'long' if current_position == 1 else 'short', 'entry_time': entry_time,
                'exit_time': final_timestamp, 'entry_price': entry_price, 'exit_price': final_price,
                'pips_gained': trade_record['pips_gained'], 'lot_size': trade_record['lot_size'],
                'pnl': trade_record['pnl'], 'pnl_pct': trade_record['pnl_pct'],
                'holding_hours': (final_timestamp - entry_time).total_seconds() / 3600,
                'reason': 'end_of_data', 'confidence_level': threshold_type
            })

        portfolio_metrics = portfolio_manager.get_performance_metrics()
        if verbose:
            print(f"   ✅ Strategy completed:\n      Total trades: {len(trades)}\n      Total pips: {portfolio_metrics['total_pips']:.1f}\n      Final capital: ${portfolio_metrics['final_capital']:,.2f}\n      Total return: {portfolio_metrics['total_return_pct']:.2f}%\n      Win rate: {portfolio_metrics['win_rate']:.4f}\n      Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.2f}\n      Max drawdown: {portfolio_metrics['max_drawdown_pct']:.2f}%")

        buy_signals = np.sum(predictions >= thresholds['buy'])
        sell_signals = np.sum(predictions <= thresholds['sell'])
        hold_signals = len(predictions) - buy_signals - sell_signals

        return {
            'trades': trades, 'signals': signals, 'positions': positions, 'performance': portfolio_metrics,
            'threshold_type': threshold_type, 'thresholds': thresholds, 'lot_size': self.config.LOT_SIZES[threshold_type],
            'portfolio_manager': portfolio_manager, 'capital_history': portfolio_manager.capital_history,
            'zone_stats': {
                'buy_signals': buy_signals, 'sell_signals': sell_signals, 'hold_signals': hold_signals,
                'buy_pct': buy_signals / len(predictions) * 100, 'sell_pct': sell_signals / len(predictions) * 100,
                'hold_pct': hold_signals / len(predictions) * 100
            }
        }

    def compare_strategies(self, predictions, prices, timestamps, currency_pair='EURUSD', verbose=True):
        if verbose:
            print(f"\n📊 Comparing all trading strategies for {currency_pair}...")
        results = {}
        for strategy_type in ['conservative', 'moderate', 'aggressive']:
            portfolio_manager = ForexPortfolioManager(self.config)
            result = self.apply_strategy(
                predictions, prices, timestamps, strategy_type, portfolio_manager, currency_pair, verbose=False)
            results[f'{strategy_type.title()}'] = result

        if verbose:
            print("\n📊 Strategy Comparison Summary:\n" + "-" * 120)
            print(f"{'Strategy':<15} {'Lot Size':<10} {'Trades':<8} {'Pips':<10} {'Return%':<10} {'Win Rate':<10} {'Sharpe':<10} {'Max DD%':<10}")
            print("-" * 120)
            for name, result in results.items():
                perf = result['performance']
                print(f"{name:<15} {result['lot_size']:<10.1f} {perf['total_trades']:<8} "
                      f"{perf['total_pips']:<10.1f} {perf['total_return_pct']:<10.2f} "
                      f"{perf['win_rate']:<10.4f} {perf['sharpe_ratio']:<10.2f} "
                      f"{perf['max_drawdown_pct']:<10.2f}")
            print("\n📊 Zone Statistics (% of signals):\n" + "-" * 80)
            print(f"{'Strategy':<15} {'Buy Zone':<15} {'Hold Zone':<15} {'Sell Zone':<15}")
            print("-" * 80)
            for name, result in results.items():
                stats = result['zone_stats']
                print(f"{name:<15} {stats['buy_pct']:<15.1f} {stats['hold_pct']:<15.1f} {stats['sell_pct']:<15.1f}")
        return results

    def apply_multi_model_to_all_pairs(self, model_builder, processed_data, data_splits,
                                      eval_set='test', verbose=True):
        if verbose:
            print("\n🌍 Applying Multi-Currency Model to Each Currency Pair\n" + "="*70)
        all_pairs_results = {}
        for currency_pair in self.config.CURRENCY_PAIRS:
            if verbose:
                print(f"\n💱 Trading {currency_pair} with Multi-Model:")
            X_eval, _, eval_timestamps = data_splits[eval_set]
            predictions = model_builder.predict(X_eval)
            processor = DataProcessor(self.config)
            eval_price_data = processor.get_price_data(processed_data, eval_timestamps, currency_pair)
            eval_prices = eval_price_data['Close_Price']
            pair_results = self.compare_strategies(
                predictions, eval_prices, eval_timestamps, currency_pair, verbose=verbose)
            all_pairs_results[currency_pair] = pair_results
        return all_pairs_results

class TechnicalIndicatorStrategies:
    """Trading strategies based on technical indicators"""
    def __init__(self, config):
        self.config = config

    def rsi_strategy(self, prices, timestamps, technical_indicators,
                    currency_pair='EURUSD', portfolio_manager=None, verbose=True):
        if verbose:
            print(f"📊 Applying RSI-based trading strategy for {currency_pair}...")
        if portfolio_manager is None:
            portfolio_manager = ForexPortfolioManager(self.config)

        rsi_values = technical_indicators['RSI']
        trades, positions = [], np.zeros(len(prices))
        current_position, entry_price, entry_time = 0, None, None
        confidence_level = 'moderate'

        for i, (price, timestamp, rsi) in enumerate(zip(prices, timestamps, rsi_values)):
            if (current_position == 1 and rsi >= self.config.RSI_OVERBOUGHT) or \
               (current_position == -1 and rsi <= self.config.RSI_OVERSOLD):
                reason = 'rsi_overbought' if current_position == 1 else 'rsi_oversold'
                trade_record = portfolio_manager.execute_forex_trade(
                    'long' if current_position == 1 else 'short', entry_price, price, timestamp,
                    currency_pair, confidence_level)
                trades.append({
                    'type': 'long' if current_position == 1 else 'short', 'entry_time': entry_time,
                    'exit_time': timestamp, 'entry_price': entry_price, 'exit_price': price,
                    'pips_gained': trade_record['pips_gained'], 'pnl': trade_record['pnl'],
                    'pnl_pct': trade_record['pnl_pct'], 'reason': reason
                })
                current_position = 0

            if current_position == 0:
                if rsi <= self.config.RSI_OVERSOLD:
                    current_position, entry_price, entry_time = 1, price, timestamp
                elif rsi >= self.config.RSI_OVERBOUGHT:
                    current_position, entry_price, entry_time = -1, price, timestamp
            positions[i] = current_position

        if current_position != 0:
            final_timestamp, final_price = get_last_timestamp(timestamps), prices.iloc[-1]
            trade_record = portfolio_manager.execute_forex_trade(
                'long' if current_position == 1 else 'short', entry_price, final_price,
                final_timestamp, currency_pair, confidence_level)
            trades.append({
                'type': 'long' if current_position == 1 else 'short', 'entry_time': entry_time,
                'exit_time': final_timestamp, 'entry_price': entry_price, 'exit_price': final_price,
                'pips_gained': trade_record['pips_gained'], 'pnl': trade_record['pnl'],
                'pnl_pct': trade_record['pnl_pct'], 'reason': 'end_of_data'
            })

        portfolio_metrics = portfolio_manager.get_performance_metrics()
        if verbose:
            print(f"   ✅ RSI Strategy completed:")
            print(f"      Total trades: {len(trades)}\n      Total pips: {portfolio_metrics['total_pips']:.1f}\n      Final capital: ${portfolio_metrics['final_capital']:,.2f}\n      Total return: {portfolio_metrics['total_return_pct']:.2f}%\n      Win rate: {portfolio_metrics['win_rate']:.4f}\n      Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.2f}")

        return {
            'strategy_name': 'RSI-based Trading', 'trades': trades, 'positions': positions,
            'performance': portfolio_metrics, 'portfolio_manager': portfolio_manager,
            'lot_size': self.config.LOT_SIZES[confidence_level]
        }

    def macd_strategy(self, prices, timestamps, technical_indicators,
                     currency_pair='EURUSD', portfolio_manager=None, verbose=True):
        if verbose:
            print(f"📊 Applying MACD-based trading strategy for {currency_pair}...")
        if portfolio_manager is None:
            portfolio_manager = ForexPortfolioManager(self.config)

        macd, macd_signal = technical_indicators['MACD'], technical_indicators['MACD_Signal']
        trades, positions = [], np.zeros(len(prices))
        current_position, entry_price, entry_time = 0, None, None
        confidence_level = 'moderate'
        start_idx = max(self.config.MACD_SLOW, 50)

        for i in range(start_idx, len(prices)):
            price, timestamp = prices.iloc[i], get_timestamp_value(timestamps, i)
            if i > 0:
                prev_macd, curr_macd = macd.iloc[i-1], macd.iloc[i]
                prev_signal, curr_signal = macd_signal.iloc[i-1], macd_signal.iloc[i]
                bullish_cross = prev_macd <= prev_signal and curr_macd > curr_signal
                bearish_cross = prev_macd >= prev_signal and curr_macd < curr_signal

                if (current_position == 1 and bearish_cross) or (current_position == -1 and bullish_cross):
                    reason = 'macd_bearish_cross' if current_position == 1 else 'macd_bullish_cross'
                    trade_record = portfolio_manager.execute_forex_trade(
                        'long' if current_position == 1 else 'short', entry_price, price, timestamp,
                        currency_pair, confidence_level)
                    trades.append({
                        'type': 'long' if current_position == 1 else 'short', 'entry_time': entry_time,
                        'exit_time': timestamp, 'entry_price': entry_price, 'exit_price': price,
                        'pips_gained': trade_record['pips_gained'], 'pnl': trade_record['pnl'],
                        'pnl_pct': trade_record['pnl_pct'], 'reason': reason
                    })
                    current_position = 0

                if current_position == 0:
                    if bullish_cross:
                        current_position, entry_price, entry_time = 1, price, timestamp
                    elif bearish_cross:
                        current_position, entry_price, entry_time = -1, price, timestamp
            positions[i] = current_position

        if current_position != 0:
            final_timestamp, final_price = get_last_timestamp(timestamps), prices.iloc[-1]
            trade_record = portfolio_manager.execute_forex_trade(
                'long' if current_position == 1 else 'short', entry_price, final_price,
                final_timestamp, currency_pair, confidence_level)
            trades.append({
                'type': 'long' if current_position == 1 else 'short', 'entry_time': entry_time,
                'exit_time': final_timestamp, 'entry_price': entry_price, 'exit_price': final_price,
                'pips_gained': trade_record['pips_gained'], 'pnl': trade_record['pnl'],
                'pnl_pct': trade_record['pnl_pct'], 'reason': 'end_of_data'
            })

        portfolio_metrics = portfolio_manager.get_performance_metrics()
        if verbose:
            print(f"   ✅ MACD Strategy completed:")
            print(f"      Total trades: {len(trades)}\n      Total pips: {portfolio_metrics['total_pips']:.1f}\n      Final capital: ${portfolio_metrics['final_capital']:,.2f}\n      Total return: {portfolio_metrics['total_return_pct']:.2f}%\n      Win rate: {portfolio_metrics['win_rate']:.4f}\n      Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.2f}")

        return {
            'strategy_name': 'MACD-based Trading', 'trades': trades, 'positions': positions,
            'performance': portfolio_metrics, 'portfolio_manager': portfolio_manager,
            'lot_size': self.config.LOT_SIZES[confidence_level]
        }

class SimpleBaselineStrategies:
    """Simple baseline strategies for comparison"""
    def __init__(self, config):
        self.config = config

    def buy_and_hold(self, prices, timestamps, currency_pair='EURUSD', portfolio_manager=None):
        """Simple buy and hold strategy with forex position sizing"""
        if portfolio_manager is None:
            portfolio_manager = ForexPortfolioManager(self.config)

        entry_price, exit_price = prices.iloc[0], prices.iloc[-1]
        exit_timestamp = get_last_timestamp(timestamps)
        confidence_level = 'conservative'

        portfolio_manager.execute_forex_trade(
            'long', entry_price, exit_price, exit_timestamp, currency_pair, confidence_level)

        return {
            'strategy_name': 'Buy and Hold',
            'performance': portfolio_manager.get_performance_metrics(),
            'portfolio_manager': portfolio_manager,
            'lot_size': self.config.LOT_SIZES[confidence_level]
        }

    def random_strategy(self, prices, timestamps, currency_pair='EURUSD',
                       num_trades=50, seed=42):
        """Random trading strategy for benchmark"""
        np.random.seed(seed)
        portfolio_manager = ForexPortfolioManager(self.config)
        confidence_level = 'moderate'
        n_prices = len(prices)
        trades = []

        for _ in range(num_trades):
            if n_prices < 10: break
            entry_idx = np.random.randint(0, n_prices - 5)
            exit_idx = np.random.randint(entry_idx + 1, min(entry_idx + 24, n_prices))
            entry_price, exit_price = prices.iloc[entry_idx], prices.iloc[exit_idx]
            exit_time = get_timestamp_value(timestamps, exit_idx)
            position = np.random.choice(['long', 'short'])

            trade_record = portfolio_manager.execute_forex_trade(
                position, entry_price, exit_price, exit_time, currency_pair, confidence_level)
            trades.append(trade_record)

        return {
            'strategy_name': 'Random',
            'performance': portfolio_manager.get_performance_metrics(),
            'portfolio_manager': portfolio_manager,
            'lot_size': self.config.LOT_SIZES[confidence_level]
        }