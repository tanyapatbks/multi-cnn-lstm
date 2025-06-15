"""
Multi-Currency CNN-LSTM Forex Prediction System
Enhanced version with Single Currency Model Support
"""
import argparse
import warnings
import numpy as np
import tensorflow as tf

# ==================================================================
# <<< เพิ่มส่วนนี้: ตรวจจับ Unattended Mode ก่อน Import Matplotlib >>>
# ==================================================================
# สร้าง Parser ชั่วคราวเพื่อเช็ค Flag --unattended ก่อน
parser_check = argparse.ArgumentParser(add_help=False)
parser_check.add_argument('--unattended', action='store_true')
args_check, _ = parser_check.parse_known_args()

# ถ้าใช้ --unattended ให้เปลี่ยน Backend ของ Matplotlib เพื่อไม่ให้แสดงกราฟ
if args_check.unattended:
    import matplotlib
    matplotlib.use('Agg')
    print("🎨 Running in Unattended Mode: Plots will be saved to files without being displayed.")
# ==================================================================

import pandas as pd
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Import our modules
from config import Config
from data_processor import DataProcessor, SequencePreparator
from cnn_lstm_model import CNNLSTMModel
from trading_strategy import FixedHoldingTradingStrategy, TechnicalIndicatorStrategies, SimpleBaselineStrategies
from checkpoint import CheckpointManager, ResultsManager
from visualization import ForexVisualizer

def train_single_model(model_type='multi', start_from_step=None, use_test_set=False, unattended_mode=False):
    """
    Train a single model (either multi-currency or single currency)
    """
    print(f"\n🚀 Training {model_type.upper()} Model")
    print("="*70)
    
    config = Config(model_type=model_type)
    config.print_config()
    
    checkpoint_manager = CheckpointManager(config)
    results_manager = ResultsManager(config)
    
    # <<< แก้ไขส่วนนี้: ตรวจสอบ Unattended Mode ก่อนถามผู้ใช้ >>>
    checkpoint = checkpoint_manager.load_checkpoint()
    if checkpoint and start_from_step is None and not unattended_mode:
        try:
            response = input("📋 Resume from checkpoint? (y/n): ")
            if response.lower() == 'y':
                checkpoint_step = checkpoint['step']
                if '_' in checkpoint_step:
                    start_from_step = int(checkpoint_step.split('_')[0])
                else: start_from_step = 1
        except EOFError:
            print("No input detected, starting fresh.")
            start_from_step = 1
    
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.experimental.enable_op_determinism()
    
    # --- The rest of the training pipeline remains largely the same ---
    # (The full code for steps 1-5 is omitted for brevity but is unchanged)
    if start_from_step is None or start_from_step <= 1:
        print("\n📚 STEP 1: DATA LOADING AND PREPROCESSING")
        processor = DataProcessor(config)
        raw_data = processor.load_currency_data()
        if raw_data is None: return None
        processed_data = processor.preprocess_data(raw_data)
        unified_data, feature_columns = processor.create_unified_dataset(processed_data)
        checkpoint_manager.save_checkpoint("1_data_preprocessing", {'processed_data': processed_data, 'unified_data': unified_data, 'feature_columns': feature_columns})
    else:
        if checkpoint and 'data' in checkpoint and 'processed_data' in checkpoint['data']:
            processed_data, unified_data, feature_columns = checkpoint['data']['processed_data'], checkpoint['data']['unified_data'], checkpoint['data']['feature_columns']
            print("✅ Loaded data from checkpoint")
        else:
            print("❌ No valid data in checkpoint. Please run from step 1."); return None

    if start_from_step is None or start_from_step <= 2:
        print("\n📋 STEP 2: SEQUENCE PREPARATION")
        sequence_prep = SequencePreparator(config)
        X, y, timestamps = sequence_prep.create_sequences(unified_data, target_pair=config.TARGET_PAIR)
        data_splits = sequence_prep.split_temporal_data(X, y, timestamps)
        checkpoint_data = checkpoint_manager.load_checkpoint(verbose=False)['data'] if checkpoint_manager.checkpoint_exists() else {}
        checkpoint_data.update({'X': X, 'y': y, 'timestamps': timestamps, 'data_splits': data_splits})
        checkpoint_manager.save_checkpoint("2_sequence_creation", checkpoint_data)
    else:
        if 'X' not in locals() and checkpoint and 'data' in checkpoint and 'X' in checkpoint['data']:
            X, y, timestamps, data_splits = checkpoint['data']['X'], checkpoint['data']['y'], checkpoint['data']['timestamps'], checkpoint['data']['data_splits']
            print("✅ Loaded sequence data from checkpoint")
        elif 'X' not in locals():
            print("❌ No sequence data found. Please run from step 2."); return None

    if start_from_step is None or start_from_step <= 3:
        print("\n🏗️  STEP 3: MODEL TRAINING")
        model_builder = CNNLSTMModel(config)
        model = model_builder.build_model()
        history = model_builder.train_model(data_splits['train'], data_splits['val'])
        model_builder.save_model()
        checkpoint_data = checkpoint_manager.load_checkpoint(verbose=False)['data'] if checkpoint_manager.checkpoint_exists() else {}
        checkpoint_data.update({'model_info': model_builder.get_model_info(), 'history': history.history if history else {}})
        checkpoint_manager.save_checkpoint("3_model_training", checkpoint_data)
    else:
        model_builder = CNNLSTMModel(config)
        try:
            model_builder.load_model(f"{config.MODELS_PATH}best_model.h5")
        except Exception:
            try:
                model_builder.load_model(f"{config.MODELS_PATH}trained_model.h5")
            except Exception as e:
                print(f"❌ Could not load any model: {str(e)}. Please run from step 3."); return None

    if start_from_step is None or start_from_step <= 4:
        print("\n📊 STEP 4: MODEL EVALUATION")
        eval_set = 'test' if use_test_set else 'val'
        if use_test_set: print("⚠️  Using TEST SET for evaluation!")
        eval_metrics = model_builder.evaluate_model_enhanced(data_splits[eval_set])
        results_manager.save_model_metrics(eval_metrics)
        checkpoint_data = checkpoint_manager.load_checkpoint(verbose=False)['data'] if checkpoint_manager.checkpoint_exists() else {}
        checkpoint_data.update({'eval_metrics': eval_metrics, 'eval_set': eval_set})
        checkpoint_manager.save_checkpoint("4_model_evaluation", checkpoint_data)
    else:
        eval_metrics = results_manager.load_results("cnn_lstm_metrics.pkl", verbose=False)
        if eval_metrics is None: print("❌ No evaluation metrics found. Please run from step 4."); return None

    if start_from_step is None or start_from_step <= 5:
        print("\n💼 STEP 5: TRADING STRATEGY TESTING")
        eval_set = 'test' if use_test_set else 'val'
        X_eval, _, eval_timestamps = data_splits[eval_set]
        predictions = model_builder.predict(X_eval)
        processor = DataProcessor(config)
        target_pair = config.TARGET_PAIR
        eval_price_data = processor.get_price_data(processed_data, eval_timestamps, target_pair)
        eval_prices = eval_price_data['Close_Price']
        technical_indicators = processor.get_technical_indicators(processed_data, eval_timestamps, target_pair)
        
        fixed_strategy = FixedHoldingTradingStrategy(config)
        trading_results = {}
        multi_pairs_results = None
        
        if model_type == 'multi':
            multi_pairs_results = fixed_strategy.apply_multi_model_to_all_pairs(model_builder, processed_data, data_splits, eval_set, verbose=True)
            for currency_pair, pair_strategies in multi_pairs_results.items():
                for strategy_name, strategy_result in pair_strategies.items():
                    key = f"Multi-Model {currency_pair} CNN-LSTM {strategy_name}"
                    trading_results[key] = strategy_result
        else:
            cnn_strategies = fixed_strategy.compare_strategies(predictions, eval_prices, eval_timestamps, model_type, verbose=True)
            for strategy_name, strategy_result in cnn_strategies.items():
                key = f"{model_type} CNN-LSTM {strategy_name}"
                trading_results[key] = strategy_result
        
        tech_strategy = TechnicalIndicatorStrategies(config)
        baseline_strategy = SimpleBaselineStrategies(config)
        trading_results['Buy & Hold'] = baseline_strategy.buy_and_hold(eval_prices, eval_timestamps, target_pair)
        trading_results['RSI-based Trading'] = tech_strategy.rsi_strategy(eval_prices, eval_timestamps, technical_indicators, target_pair)
        trading_results['MACD-based Trading'] = tech_strategy.macd_strategy(eval_prices, eval_timestamps, technical_indicators, target_pair)
        
        results_manager.save_trading_results(trading_results)
        checkpoint_data = checkpoint_manager.load_checkpoint(verbose=False)['data'] if checkpoint_manager.checkpoint_exists() else {}
        checkpoint_data.update({'trading_results': trading_results, 'predictions': predictions, 'multi_pairs_results': multi_pairs_results})
        checkpoint_manager.save_checkpoint("5_strategy_testing", checkpoint_data)
    else:
        trading_results = results_manager.load_results("fixed_holding_results.pkl", verbose=False)
        if trading_results is None: print("❌ No trading results found. Please run from step 5."); return None

    # Step 6 is now handled by the calling function to allow for batch processing
    print(f"\n✅ {model_type.upper()} MODEL TRAINING COMPLETED!")
    print("="*70)
    checkpoint_manager.save_checkpoint("6_final_results", {'eval_metrics': eval_metrics, 'trading_results': trading_results})
    return {'model_type': model_type, 'eval_metrics': eval_metrics, 'trading_results': trading_results}


def compare_all_models(currency_pair='EURUSD', unattended_mode=False):
    """
    Compare multi vs single model, passing the unattended flag to the visualizer.
    """
    print(f"\n🔄 Comparing Multi vs Single Model for {currency_pair}")
    print("="*70)
    multi_config, single_config = Config(model_type='multi'), Config(model_type=currency_pair)
    multi_results_manager, single_results_manager = ResultsManager(multi_config), ResultsManager(single_config)
    multi_trading = multi_results_manager.load_results("fixed_holding_results.pkl", verbose=False)
    single_trading = single_results_manager.load_results("fixed_holding_results.pkl", verbose=False)

    if not multi_trading or not single_trading:
        print(f"❌ Results not found for comparison. Please train both models first."); return None

    comparison_results = {}
    
    # Find best Multi-Model result
    best_multi_return, best_multi_strategy_result = -float('inf'), None
    for name, result in multi_trading.items():
        if f'Multi-Model {currency_pair} CNN-LSTM' in name and result.get('performance', {}).get('total_return_pct', -float('inf')) > best_multi_return:
            best_multi_return = result['performance']['total_return_pct']
            best_multi_strategy_result = result
    if best_multi_strategy_result: comparison_results['Multi-Model CNN-LSTM'] = best_multi_strategy_result
    else: print(f"⚠️ Could not find multi-model results for {currency_pair}")

    # Find best Single-Model result
    best_single_return, best_single_strategy_result = -float('inf'), None
    for name, result in single_trading.items():
        if f'{currency_pair} CNN-LSTM' in name and result.get('performance', {}).get('total_return_pct', -float('inf')) > best_single_return:
            best_single_return = result['performance']['total_return_pct']
            best_single_strategy_result = result
    if best_single_strategy_result: comparison_results[f'{currency_pair} CNN-LSTM'] = best_single_strategy_result
    else: print(f"⚠️ Could not find single-model results for {currency_pair}")

    # Add baselines
    for name, result in single_trading.items():
        if 'Buy & Hold' in name: comparison_results['Buy & Hold'] = result
        elif 'RSI-based' in name: comparison_results['RSI-based Trading'] = result
        elif 'MACD-based' in name: comparison_results['MACD-based Trading'] = result

    # <<< แก้ไขส่วนนี้: ส่ง Flag unattended_mode ไปให้ Visualizer >>>
    visualizer = ForexVisualizer(single_config, unattended_mode=unattended_mode)
    visualizer.plot_comprehensive_comparison(comparison_results, currency_pair)
    visualizer.plot_cumulative_returns_comparison(comparison_results, currency_pair)
    print(f"✅ Comparison completed for {currency_pair}")
    return comparison_results

def run_pipeline(args):
    """
    Main pipeline to orchestrate training and visualization based on args.
    """
    if args.visualize_only:
        print("📊 Running in visualization-only mode...")
        currency_comparisons = {}
        for currency in ['EURUSD', 'GBPUSD', 'USDJPY']:
            comparison = compare_all_models(currency, unattended_mode=args.unattended)
            if comparison: currency_comparisons[currency] = comparison
        if currency_comparisons:
            visualizer = ForexVisualizer(Config(model_type='multi'), unattended_mode=args.unattended)
            visualizer.create_performance_summary_table(currency_comparisons)
        print("\n✅ Visualization and comparison complete.")
        return

    if args.model == 'all':
        all_results = {}
        model_types_to_train = ['multi', 'EURUSD', 'GBPUSD', 'USDJPY']
        for model_type in model_types_to_train:
            # <<< แก้ไขส่วนนี้: ลบ checkpoint อัตโนมัติใน Unattended Mode >>>
            if args.unattended:
                config = Config(model_type=model_type)
                checkpoint_manager = CheckpointManager(config)
                if checkpoint_manager.checkpoint_exists():
                    print(f"🧹 Unattended mode: Deleting existing checkpoint for '{model_type}'.")
                    checkpoint_manager.delete_checkpoint(verbose=False)
            
            results = train_single_model(model_type, use_test_set=args.test, unattended_mode=args.unattended)
            if results: all_results[model_type] = results
        
        print("\n" + "="*70 + "\nTRAINING COMPLETE. NOW RUNNING FINAL ANALYSIS.\n" + "="*70)
        currency_comparisons = {}
        for currency in ['EURUSD', 'GBPUSD', 'USDJPY']:
            comparison = compare_all_models(currency, unattended_mode=args.unattended)
            if comparison: currency_comparisons[currency] = comparison
        if currency_comparisons:
            visualizer = ForexVisualizer(Config(model_type='multi'), unattended_mode=args.unattended)
            visualizer.create_performance_summary_table(currency_comparisons)
        print("\n✅✅✅ FULL PIPELINE COMPLETED SUCCESSFULLY! ✅✅✅")
    
    else: # Run for a single specified model
        if args.new:
            config = Config(model_type=args.model)
            checkpoint_manager = CheckpointManager(config)
            checkpoint_manager.delete_checkpoint()
            
        results = train_single_model(args.model, start_from_step=args.step, use_test_set=args.test, unattended_mode=args.unattended)
        # Display results for single run
        if results:
            print("\n📊 RESULTS SUMMARY AND VISUALIZATION")
            config = Config(model_type=args.model)
            print_detailed_results(results['eval_metrics'], results['trading_results'], config)
            visualizer = ForexVisualizer(config, unattended_mode=args.unattended)
            # You might need to load history from checkpoint to plot curves here
            print("To see training curves, please check the saved file in the results directory.")


def print_detailed_results(eval_metrics, trading_results, config):
    # This function is unchanged but included for completeness.
    """Print detailed results summary"""
    print("\n📈 MODEL PERFORMANCE SUMMARY")
    print("-" * 50)
    print(f"Model Type: {config.MODEL_TYPE}")
    for key, val in eval_metrics.items():
        if isinstance(val, (int, float)): print(f"{key.replace('_', ' ').title():<12}: {val:.4f}")
    
    print("\n💼 TRADING STRATEGY PERFORMANCE")
    header = f"{'Strategy':<45} {'Trades':<8} {'Return%':<10} {'Win Rate':<10} {'Sharpe':<10} {'Max DD%':<10} {'Final $':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    sorted_results = sorted(trading_results.items(), key=lambda item: item[1].get('performance', {}).get('total_return_pct', -9e9), reverse=True)
    for name, result in sorted_results:
        perf = result.get('performance', result)
        print(f"{name:<45} {perf.get('total_trades', 0):<8} "
              f"{perf.get('total_return_pct', 0):<10.2f} {perf.get('win_rate', 0):<10.4f} "
              f"{perf.get('sharpe_ratio', 0):<10.4f} {perf.get('max_drawdown_pct', 0):<10.2f} "
              f"${perf.get('final_capital', config.INITIAL_CAPITAL):<11,.2f}")
    print("-" * len(header))
    
    if sorted_results:
        best_name, best_res = sorted_results[0]
        print(f"\n🏆 Best performing strategy: {best_name} (Return: {best_res.get('performance', best_res).get('total_return_pct', 0):.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Currency CNN-LSTM Forex Prediction', formatter_class=argparse.RawTextHelpFormatter)
    
    # Use the main parser, not the temporary one
    parser.add_argument('--model', type=str, choices=['multi', 'EURUSD', 'GBPUSD', 'USDJPY', 'all'], default='all', help='Model type to train.')
    parser.add_argument('--step', type=int, choices=range(1, 7), help='Start from specific step (1-6) for a single model run.')
    parser.add_argument('--test', action='store_true', help='Use test set for final evaluation instead of validation set.')
    parser.add_argument('--new', action='store_true', help="Start a fresh run for a single model, deleting its checkpoint.")
    parser.add_argument('--visualize-only', action='store_true', help='Skip training. Load saved results and generate comparison visualizations.')
    # <<< เพิ่ม Argument ใหม่ >>>
    parser.add_argument('--unattended', action='store_true', help='Run in non-interactive mode. Skips user prompts and does not display plot windows.')

    args = parser.parse_args()
    
    run_pipeline(args)