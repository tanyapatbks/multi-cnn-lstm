"""
Complete Main execution file for the Multi-Currency CNN-LSTM Forex Prediction System.
Enhanced with multiple threshold support, comprehensive analysis, and leverage support.
"""
import argparse
import warnings
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import matplotlib

# Set backend before importing pyplot for unattended runs
parser_check = argparse.ArgumentParser(add_help=False)
parser_check.add_argument('--unattended', action='store_true')
args_check, _ = parser_check.parse_known_args()
if args_check.unattended:
    matplotlib.use('Agg')

from config import Config
from data_processor import DataProcessor, SequencePreparator
from cnn_lstm_model import CNNLSTMModel
from trading_strategy import (TradingSimulator, get_cnn_lstm_signals_multiple_thresholds, 
                             get_rsi_signals, get_macd_signals, get_buy_and_hold_signals)
from visualization import ForexVisualizer

warnings.filterwarnings('ignore')

def run_complete_loop(config_base: Config, use_test_set=False, unattended_mode=False):
    """
    Enhanced complete loop with multiple threshold support and leverage for comprehensive evaluation.
    """
    all_loop_results = {}
    
    # --- Part 1: Prepare Data ONCE for the entire loop ---
    print("\n📚 STEP 1: PREPARING ALL DATA FOR THIS LOOP...")
    master_processor = DataProcessor(config_base)
    
    raw_data = master_processor.load_currency_data()
    if raw_data is None: 
        print("❌ Failed to load raw data. Exiting.")
        return
    processed_data = master_processor.preprocess_data(raw_data)

    # --- Part 2: Train all CNN-LSTM Model Variations with Multiple Thresholds and Leverage ---
    print(f"\n{'='*80}\n🚀 PART 1: TRAINING CNN-LSTM MODELS WITH LEVERAGE SUPPORT\n{'='*80}")
    print("💰 Leverage Configuration:")
    print("   Conservative: 2.0x (High confidence signals)")
    print("   Moderate:     1.0x (Standard leverage)")
    print("   Aggressive:   0.5x (Uncertain signals)")
    print("="*80)
    
    model_variations = [
        {'model_type': 'multi', 'target_pair': 'EURUSD'},
        {'model_type': 'multi', 'target_pair': 'GBPUSD'},
        {'model_type': 'multi', 'target_pair': 'USDJPY'},
        {'model_type': 'EURUSD', 'target_pair': 'EURUSD'},
        {'model_type': 'GBPUSD', 'target_pair': 'GBPUSD'},
        {'model_type': 'USDJPY', 'target_pair': 'USDJPY'},
    ]

    for variation in model_variations:
        results = train_and_evaluate_model_multiple_thresholds(variation, config_base, processed_data, use_test_set, unattended_mode)
        if results:
            all_loop_results.update(results)
    
    # --- Part 3: Evaluate Baseline Strategies ---
    print(f"\n{'='*80}\n📊 PART 2: CALCULATING BASELINE STRATEGIES WITH MODERATE LEVERAGE\n{'='*80}")
    for currency_pair in ['EURUSD', 'GBPUSD', 'USDJPY']:
        print(f"--- Calculating baselines for {currency_pair} (Leverage: 1.0x) ---")
        baseline_results = evaluate_baselines(config_base, processed_data, currency_pair, use_test_set)
        all_loop_results.update(baseline_results)

    # --- Part 4: Generate Final Visualizations for this Loop ---
    print(f"\n{'='*80}\n🎨 PART 3: GENERATING ENHANCED VISUALIZATIONS\n{'='*80}")
    
    report_path = f'results/final_report_for_val_{config_base.VAL_START}_to_{config_base.VAL_END}/'
    os.makedirs(report_path, exist_ok=True)
    
    report_config = Config(); report_config.RESULTS_PATH = report_path
    visualizer = ForexVisualizer(report_config, unattended_mode=unattended_mode)
    
    # Enhanced visualization with leverage information
    visualizer.create_loop_summary_csv(all_loop_results)
    for currency_pair in ['EURUSD', 'GBPUSD', 'USDJPY']:
        visualizer.plot_loop_comparison_graph(all_loop_results, currency_pair)
    
    # Additional threshold comparison summary
    visualizer.plot_threshold_comparison_summary(all_loop_results)
    
    print(f"\n🎉 ENHANCED LOOP WITH LEVERAGE COMPLETED!")
    print(f"📄 Results saved in: {report_path}")
    print(f"📊 Total strategies evaluated: {len(all_loop_results)}")
    print(f"💰 Leverage impact included in all CNN-LSTM strategies")

def analyze_model_predictions(predictions, model_type, target_pair, config):
    """
    Analyze and display detailed prediction statistics and signal distribution.
    
    Args:
        predictions: numpy array of model predictions (0-1 values)
        model_type: str, 'multi' or single currency
        target_pair: str, target currency pair
        config: Config object with threshold settings
    """
    
    # Calculate prediction statistics
    pred_min = predictions.min()
    pred_max = predictions.max()
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    
    print(f"\n  📊 Model Prediction Analysis:")
    print(f"      • Prediction range: [{pred_min:.3f}, {pred_max:.3f}]")
    print(f"      • Mean: {pred_mean:.3f}, Std: {pred_std:.3f}")
    
    # Calculate signal distribution for all thresholds
    print(f"      • Trading signals generated:")
    
    for threshold_name, thresholds in config.THRESHOLDS.items():
        buy_threshold = thresholds['buy']
        sell_threshold = thresholds['sell']
        
        # Count signals
        buy_signals = np.sum(predictions >= buy_threshold)
        sell_signals = np.sum(predictions <= sell_threshold)
        hold_signals = len(predictions) - buy_signals - sell_signals
        
        total_predictions = len(predictions)
        buy_pct = (buy_signals / total_predictions) * 100
        sell_pct = (sell_signals / total_predictions) * 100
        hold_pct = (hold_signals / total_predictions) * 100
        
        leverage = config.LEVERAGE_SETTINGS[threshold_name]
        
        print(f"        {threshold_name} (Leverage {leverage}x):")
        print(f"          - HOLD: {hold_signals} ({hold_pct:.1f}%)")
        print(f"          - SELL: {sell_signals} ({sell_pct:.1f}%)")
        print(f"          - BUY:  {buy_signals} ({buy_pct:.1f}%)")
    
    return {
        'pred_range': [pred_min, pred_max],
        'pred_stats': {'mean': pred_mean, 'std': pred_std},
        'signal_counts': {
            threshold_name: {
                'buy': np.sum(predictions >= thresholds['buy']),
                'sell': np.sum(predictions <= thresholds['sell']),
                'hold': len(predictions) - np.sum(predictions >= thresholds['buy']) - np.sum(predictions <= thresholds['sell'])
            }
            for threshold_name, thresholds in config.THRESHOLDS.items()
        }
    }

def train_and_evaluate_model_multiple_thresholds(variation, config_base, processed_data, use_test_set, unattended_mode):
    """Enhanced function with prediction analysis, multiple threshold evaluation and leverage support."""
    
    model_type, target_pair = variation['model_type'], variation['target_pair']
    run_name = f"{model_type}_target_{target_pair}" if model_type == 'multi' else model_type
    print(f"\n--- Processing: {run_name.upper()} ---")

    run_config = Config(model_type=model_type, target_pair=target_pair)
    run_config.RESULTS_PATH = f'results/model_runs/{run_name}/'
    run_config.MODELS_PATH = f'models/{run_name}/'
    run_config._create_directories()
        
    dp = DataProcessor(run_config)
    model_input = dp.get_model_input_data(processed_data)
    (train_data, eval_data) = SequencePreparator(run_config).create_sequences_and_splits(model_input, processed_data, use_test_set)
    
    (X_train, y_train), (X_eval, y_eval, ts_eval) = train_data, eval_data
    if len(X_eval) == 0: 
        print(f"⚠️ No evaluation data for {run_name}")
        return {}
    
    # Train model
    model = CNNLSTMModel(run_config)
    history = model.train((X_train, y_train), (X_eval, y_eval))
    
    # Plot training curves
    visualizer = ForexVisualizer(run_config, unattended_mode=unattended_mode)
    visualizer.plot_training_curves(history)

    # Get predictions
    predictions = model.model.predict(X_eval).flatten()
    eval_prices = processed_data[run_config.TARGET_PAIR]['Close'].reindex(ts_eval)
    
    # ✅ NEW: Analyze predictions and show signal distribution
    analyze_model_predictions(predictions, model_type, target_pair, run_config)
    
    # Generate signals for all thresholds
    threshold_signals = get_cnn_lstm_signals_multiple_thresholds(run_config, predictions)
    
    results = {}
    key_prefix = 'Multi-CNN-LSTM' if run_config.MODEL_TYPE == 'multi' else 'Single-CNN-LSTM'
    
    # Evaluate each threshold separately WITH LEVERAGE
    print(f"  🎯 Evaluating {len(threshold_signals)} thresholds with leverage:")
    for threshold_name, signals in threshold_signals.items():
        # ✅ เพิ่ม strategy_threshold parameter สำหรับ leverage
        simulator = TradingSimulator(run_config, eval_prices, strategy_threshold=threshold_name)
        performance = simulator.run(signals)
        
        # Create unique key for each threshold
        strategy_key = f"{key_prefix}-{threshold_name} ({run_config.TARGET_PAIR})"
        results[strategy_key] = performance
        
        # ✅ แสดงข้อมูล leverage ในการรายงาน
        leverage = simulator.portfolio.current_leverage
        print(f"    └─ {threshold_name} (Leverage {leverage}x): Return={performance['total_return_pct']:.2f}%, "
              f"Sharpe={performance['sharpe_ratio']:.2f}, Trades={performance['total_trades']}")
    
    return results

def evaluate_baselines(config, processed_data, currency_pair, use_test_set):
    """Calculates performance for baseline strategies with moderate leverage (1.0x)."""
    eval_set_start = config.TEST_START if use_test_set else config.VAL_START
    eval_set_end = config.TEST_END if use_test_set else config.VAL_END
    
    processor = DataProcessor(config)
    price_data = processor.get_price_data_for_period(processed_data, currency_pair, eval_set_start, eval_set_end)
    if price_data.empty: 
        print(f"⚠️ No price data for {currency_pair}")
        return {}

    eval_prices = price_data['Close']
    
    run_config = Config(model_type=currency_pair, target_pair=currency_pair)
    
    # Calculate baseline strategies with moderate leverage (1.0x)
    baseline_results = {}
    
    try:
        # ✅ เพิ่ม strategy_threshold='Moderate' สำหรับ leverage 1.0x
        sim = TradingSimulator(run_config, eval_prices, strategy_threshold='Moderate')
        performance = sim.run(get_buy_and_hold_signals(eval_prices))
        baseline_results[f'Buy & Hold ({currency_pair})'] = performance
        print(f"  ✅ Buy & Hold: Return={performance['total_return_pct']:.2f}%, Leverage={performance.get('avg_leverage', 1.0):.1f}x")
    except Exception as e:
        print(f"⚠️ Error calculating Buy & Hold for {currency_pair}: {e}")
    
    try:
        # ✅ เพิ่ม strategy_threshold='Moderate' สำหรับ leverage 1.0x
        sim = TradingSimulator(run_config, eval_prices, strategy_threshold='Moderate')
        performance = sim.run(get_rsi_signals(run_config, price_data['RSI']))
        baseline_results[f'RSI-based Trading ({currency_pair})'] = performance
        print(f"  ✅ RSI Strategy: Return={performance['total_return_pct']:.2f}%, Leverage={performance.get('avg_leverage', 1.0):.1f}x")
    except Exception as e:
        print(f"⚠️ Error calculating RSI strategy for {currency_pair}: {e}")
    
    try:
        # ✅ เพิ่ม strategy_threshold='Moderate' สำหรับ leverage 1.0x
        sim = TradingSimulator(run_config, eval_prices, strategy_threshold='Moderate')
        performance = sim.run(get_macd_signals(run_config, price_data['MACD'], price_data['MACD_Signal']))
        baseline_results[f'MACD-based Trading ({currency_pair})'] = performance
        print(f"  ✅ MACD Strategy: Return={performance['total_return_pct']:.2f}%, Leverage={performance.get('avg_leverage', 1.0):.1f}x")
    except Exception as e:
        print(f"⚠️ Error calculating MACD strategy for {currency_pair}: {e}")
    
    print(f"  ✅ Calculated {len(baseline_results)} baseline strategies for {currency_pair}")
    return baseline_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run enhanced Forex CNN-LSTM experiment with leverage support.')
    parser.add_argument('--test', action='store_true', help='Use test set instead of validation set for evaluation.')
    parser.add_argument('--unattended', action='store_true', help='Run in non-interactive mode (saves plots without showing).')
    args = parser.parse_args()

    base_config = Config()
    
    print("="*80)
    print("🚀 ENHANCED FOREX CNN-LSTM PREDICTION SYSTEM WITH LEVERAGE")
    print("="*80)
    eval_period_name = "TEST" if args.test else "VALIDATION"
    eval_start = base_config.TEST_START if args.test else base_config.VAL_START
    eval_end = base_config.TEST_END if args.test else base_config.VAL_END
    print(f"📅 Evaluation Period: {eval_start} to {eval_end} ({eval_period_name})")
    print(f"🎯 Trading Thresholds: {list(base_config.THRESHOLDS.keys())}")
    print(f"💰 Leverage Configuration:")
    print(f"   • Conservative: 2.0x leverage (High confidence)")
    print(f"   • Moderate:     1.0x leverage (Standard)")
    print(f"   • Aggressive:   0.5x leverage (Low confidence)")
    print(f"💱 Currency Pairs: {base_config.ALL_CURRENCY_PAIRS}")
    print(f"🤖 Model Variations: Multi-Currency + Single-Currency for each pair")
    print("="*80)

    try:
        run_complete_loop(base_config, use_test_set=args.test, unattended_mode=args.unattended)
        print(f"\n✅ EXPERIMENT WITH LEVERAGE COMPLETED SUCCESSFULLY!")
        print(f"📊 Check results in: results/final_report_for_val_{base_config.VAL_START}_to_{base_config.VAL_END}/")
        print(f"💰 All CNN-LSTM strategies now include realistic leverage effects")
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {str(e)}")
        import traceback
        traceback.print_exc()