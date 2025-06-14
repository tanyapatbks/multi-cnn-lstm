"""
Multi-Currency CNN-LSTM Forex Prediction System
Enhanced version with Single Currency Model Support

Author: Your Name
Thesis: Master's Degree - Forex Trend Prediction
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from config import Config
from data_processor import DataProcessor, SequencePreparator
from cnn_lstm_model import CNNLSTMModel
from trading_strategy import FixedHoldingTradingStrategy, TechnicalIndicatorStrategies, SimpleBaselineStrategies, ForexPortfolioManager
from checkpoint import CheckpointManager, ResultsManager
from visualization import ForexVisualizer

def train_single_model(model_type='multi', start_from_step=None, use_test_set=False):
    """
    Train a single model (either multi-currency or single currency)
    
    Args:
        model_type: 'multi' or specific currency pair ('EURUSD', 'GBPUSD', 'USDJPY')
        start_from_step: Step number to start from
        use_test_set: Whether to use test set for final evaluation
    """
    
    print(f"\n🚀 Training {model_type.upper()} Model")
    print("="*70)
    
    # Initialize components with specific model type
    config = Config(model_type=model_type)
    config.print_config()
    
    checkpoint_manager = CheckpointManager(config)
    results_manager = ResultsManager(config)
    
    # Check for existing checkpoint
    checkpoint = checkpoint_manager.load_checkpoint()
    if checkpoint and start_from_step is None:
        response = input("📋 Resume from checkpoint? (y/n): ")
        if response.lower() == 'y':
            # Extract step number from checkpoint step name
            checkpoint_step = checkpoint['step']
            # Convert step name to number (e.g., '4_model_evaluation' -> 4)
            if '_' in checkpoint_step:
                start_from_step = int(checkpoint_step.split('_')[0])
            else:
                start_from_step = 1
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.experimental.enable_op_determinism()
    
    # Step 1: Data Loading and Preprocessing
    if start_from_step is None or start_from_step <= 1:
        print("\n📚 STEP 1: DATA LOADING AND PREPROCESSING")
        print("-" * 50)
        
        # Load data
        processor = DataProcessor(config)
        raw_data = processor.load_currency_data()
        
        if raw_data is None:
            print("❌ Failed to load data. Please check data files.")
            return None
        
        # Preprocess data
        processed_data = processor.preprocess_data(raw_data)
        unified_data, feature_columns = processor.create_unified_dataset(processed_data)
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint("1_data_preprocessing", {
            'processed_data': processed_data,
            'unified_data': unified_data,
            'feature_columns': feature_columns
        })
        
    else:
        # Load from checkpoint
        checkpoint = checkpoint_manager.load_checkpoint(verbose=False)
        if checkpoint and 'data' in checkpoint:
            processed_data = checkpoint['data']['processed_data']
            unified_data = checkpoint['data']['unified_data']
            feature_columns = checkpoint['data']['feature_columns']
            print("✅ Loaded data from checkpoint")
        else:
            print("❌ No checkpoint data found. Please run from step 1.")
            return None
    
    # Step 2: Sequence Preparation
    if start_from_step is None or start_from_step <= 2:
        print("\n📋 STEP 2: SEQUENCE PREPARATION")
        print("-" * 50)
        
        sequence_prep = SequencePreparator(config)
        X, y, timestamps = sequence_prep.create_sequences(unified_data, target_pair=config.TARGET_PAIR)
        data_splits = sequence_prep.split_temporal_data(X, y, timestamps)
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint("2_sequence_creation", {
            'processed_data': processed_data,
            'unified_data': unified_data,
            'feature_columns': feature_columns,
            'X': X, 'y': y, 'timestamps': timestamps,
            'data_splits': data_splits
        })
        
    else:
        # Load from checkpoint
        if 'X' not in locals():
            checkpoint = checkpoint_manager.load_checkpoint(verbose=False)
            if checkpoint and 'data' in checkpoint:
                X = checkpoint['data']['X']
                y = checkpoint['data']['y']
                timestamps = checkpoint['data']['timestamps']
                data_splits = checkpoint['data']['data_splits']
                print("✅ Loaded sequence data from checkpoint")
            else:
                print("❌ No sequence data found. Please run from step 2.")
                return None
    
    # Step 3: Model Training
    if start_from_step is None or start_from_step <= 3:
        print("\n🏗️  STEP 3: MODEL TRAINING")
        print("-" * 50)
        
        # Build and train model
        model_builder = CNNLSTMModel(config)
        model = model_builder.build_model()
        
        # Train model
        history = model_builder.train_model(data_splits['train'], data_splits['val'])
        
        # Save model
        model_builder.save_model()
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint("3_model_training", {
            'processed_data': processed_data,
            'unified_data': unified_data,
            'feature_columns': feature_columns,
            'X': X, 'y': y, 'timestamps': timestamps,
            'data_splits': data_splits,
            'model_info': model_builder.get_model_info(),
            'history': history.history if history else {}
        })
        
    else:
        # Load trained model
        if 'model_builder' not in locals():
            model_builder = CNNLSTMModel(config)
            try:
                # Try to load best model first
                try:
                    model_builder.load_model(f"{config.MODELS_PATH}best_model.h5")
                    print("✅ Loaded best model")
                except:
                    # Fallback to trained model
                    model_builder.load_model(f"{config.MODELS_PATH}trained_model.h5")
                    print("✅ Loaded trained model")
            except Exception as e:
                print(f"❌ Could not load model: {str(e)}")
                print("Please run from step 3.")
                return None
    
    # Step 4: Model Evaluation
    if start_from_step is None or start_from_step <= 4:
        print("\n📊 STEP 4: MODEL EVALUATION")
        print("-" * 50)
        
        # Choose evaluation set
        eval_set = 'test' if use_test_set else 'val'
        
        if use_test_set:
            print("⚠️  Using TEST SET for evaluation!")
        
        # Check if we have enhanced evaluate method
        if hasattr(model_builder, 'evaluate_model_enhanced'):
            eval_metrics = model_builder.evaluate_model_enhanced(data_splits[eval_set])
        else:
            eval_metrics = model_builder.evaluate_model(data_splits[eval_set])
        
        # Save model metrics
        results_manager.save_model_metrics(eval_metrics)
        
        # Save checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint(verbose=False)['data'] if checkpoint_manager.checkpoint_exists() else {}
        checkpoint_data.update({
            'eval_metrics': eval_metrics,
            'eval_set': eval_set
        })
        
        checkpoint_manager.save_checkpoint("4_model_evaluation", checkpoint_data)
        
    else:
        # Load evaluation metrics
        eval_metrics = results_manager.load_results("cnn_lstm_metrics.pkl", verbose=False)
        if eval_metrics is None:
            print("❌ No evaluation metrics found. Please run from step 4.")
            return None
    
    # Step 5: Trading Strategy Testing
    if start_from_step is None or start_from_step <= 5:
        print("\n💼 STEP 5: TRADING STRATEGY TESTING")
        print("-" * 50)
        
        # Get evaluation data
        eval_set = 'test' if use_test_set else 'val'
        X_eval, y_eval, eval_timestamps = data_splits[eval_set]
        
        # Get predictions
        predictions = model_builder.predict(X_eval)
        
        # Get price data and technical indicators
        processor = DataProcessor(config)
        target_pair = config.TARGET_PAIR
        eval_price_data = processor.get_price_data(processed_data, eval_timestamps, target_pair)
        eval_prices = eval_price_data['Close_Price']
        technical_indicators = processor.get_technical_indicators(processed_data, eval_timestamps, target_pair)
        
        # Initialize strategy managers
        fixed_strategy = FixedHoldingTradingStrategy(config)
        tech_strategy = TechnicalIndicatorStrategies(config)
        baseline_strategy = SimpleBaselineStrategies(config)
        
        # Different approach for multi vs single model
        trading_results = {}
        
        if model_type == 'multi':
            # For multi-model: test on ALL currency pairs with ALL strategies
            print("\n🌍 Testing Multi-Model on Each Currency Pair...")
            
            multi_pairs_results = fixed_strategy.apply_multi_model_to_all_pairs(
                model_builder, processed_data, data_splits, eval_set, verbose=True
            )
            
            # Store results for each pair and strategy combination
            for currency_pair, pair_strategies in multi_pairs_results.items():
                for strategy_name, strategy_result in pair_strategies.items():
                    key = f"Multi-Model {currency_pair} {strategy_name}"
                    trading_results[key] = strategy_result
            
            # Also add baseline strategies for comparison (using EURUSD as default)
            trading_results['Buy & Hold'] = baseline_strategy.buy_and_hold(
                eval_prices, eval_timestamps, target_pair
            )
            trading_results['RSI-based Trading'] = tech_strategy.rsi_strategy(
                eval_prices, eval_timestamps, technical_indicators, target_pair
            )
            trading_results['MACD-based Trading'] = tech_strategy.macd_strategy(
                eval_prices, eval_timestamps, technical_indicators, target_pair
            )
            
        else:
            # For single model: test all strategies on that specific pair
            print(f"\n💱 Testing {model_type} Model with All Strategies...")
            
            # Test all three CNN-LSTM strategies
            cnn_strategies = fixed_strategy.compare_strategies(
                predictions, eval_prices, eval_timestamps, model_type, verbose=True
            )
            
            for strategy_name, strategy_result in cnn_strategies.items():
                key = f"{model_type} {strategy_name}"
                trading_results[key] = strategy_result
            
            # Apply baseline strategies
            trading_results['Buy & Hold'] = baseline_strategy.buy_and_hold(
                eval_prices, eval_timestamps, model_type
            )
            trading_results['RSI-based Trading'] = tech_strategy.rsi_strategy(
                eval_prices, eval_timestamps, technical_indicators, model_type
            )
            trading_results['MACD-based Trading'] = tech_strategy.macd_strategy(
                eval_prices, eval_timestamps, technical_indicators, model_type
            )
        
        # Save trading results
        results_manager.save_trading_results(trading_results)
        
        # Save checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint(verbose=False)['data'] if checkpoint_manager.checkpoint_exists() else {}
        checkpoint_data.update({
            'trading_results': trading_results,
            'predictions': predictions,
            'multi_pairs_results': multi_pairs_results if model_type == 'multi' else None
        })
        
        checkpoint_manager.save_checkpoint("5_strategy_testing", checkpoint_data)
        
    else:
        # Load trading results
        trading_results = results_manager.load_results("fixed_holding_results.pkl", verbose=False)
        
        if trading_results is None:
            print("❌ No trading results found. Please run from step 5.")
            return None
    
    # Step 6: Results Summary and Visualization
    if start_from_step is None or start_from_step <= 6:
        print("\n📊 STEP 6: RESULTS SUMMARY AND VISUALIZATION")
        print("-" * 50)
        
        # Create summary
        summary_report = results_manager.create_summary_report(eval_metrics, trading_results)
        
        # Print results
        print_detailed_results(eval_metrics, trading_results, config)
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        visualizer = ForexVisualizer(config)
        
        # Get training history
        checkpoint_data = checkpoint_manager.load_checkpoint(verbose=False)
        if checkpoint_data and 'data' in checkpoint_data:
            model_history = checkpoint_data['data'].get('history', {})
        else:
            model_history = {}
        
        # Create training curves
        if model_history:
            visualizer.plot_training_curves(model_history)
        
        # Save performance summary
        visualizer.save_performance_summary(eval_metrics, trading_results)
        
        print(f"\n✅ {model_type.upper()} MODEL TRAINING COMPLETED!")
        print("="*70)
        
        # Save final checkpoint
        checkpoint_manager.save_checkpoint("6_final_results", {
            'summary_report': summary_report,
            'eval_metrics': eval_metrics,
            'trading_results': trading_results,
            'visualizations_created': True
        })
        
        return {
            'model_type': model_type,
            'eval_metrics': eval_metrics,
            'trading_results': trading_results,
            'summary_report': summary_report
        }

def compare_all_models(currency_pair='EURUSD'):
    """
    Compare multi-currency model with single currency model for a specific pair
    """
    print(f"\n🔄 Comparing Multi vs Single Model for {currency_pair}")
    print("="*70)
    
    # Load results for multi-model
    multi_config = Config(model_type='multi')
    multi_results_manager = ResultsManager(multi_config)
    multi_trading = multi_results_manager.load_results("fixed_holding_results.pkl", verbose=False)
    
    # Load results for single model
    single_config = Config(model_type=currency_pair)
    single_results_manager = ResultsManager(single_config)
    single_trading = single_results_manager.load_results("fixed_holding_results.pkl", verbose=False)
    
    if not multi_trading or not single_trading:
        print("❌ Results not found for comparison. Please train both models first.")
        return
    
    # Combine results
    comparison_results = {}
    
    # Add multi-model results
    for strategy_name, result in multi_trading.items():
        if 'CNN-LSTM' in strategy_name and 'Multi' in strategy_name:
            comparison_results['Multi-Model CNN-LSTM'] = result
        elif 'Buy & Hold' in strategy_name:
            comparison_results['Buy & Hold'] = result
        elif 'RSI' in strategy_name:
            comparison_results['RSI-based Trading'] = result
        elif 'MACD' in strategy_name:
            comparison_results['MACD-based Trading'] = result
    
    # Add single model result
    for strategy_name, result in single_trading.items():
        if 'CNN-LSTM' in strategy_name and currency_pair in strategy_name:
            comparison_results[f'{currency_pair} CNN-LSTM'] = result
    
    # Create comparison visualizations
    visualizer = ForexVisualizer(single_config)
    visualizer.plot_model_comparison(comparison_results, currency_pair)
    visualizer.plot_cumulative_returns_comparison(comparison_results, currency_pair)
    
    print(f"✅ Comparison completed for {currency_pair}")
    
    return comparison_results

def train_all_models(use_test_set=False):
    """
    Train all models: multi-currency and individual currency models
    """
    all_results = {}
    
    # Train multi-currency model
    print("\n" + "="*70)
    print("TRAINING MULTI-CURRENCY MODEL")
    print("="*70)
    multi_results = train_single_model('multi', use_test_set=use_test_set)
    if multi_results:
        all_results['multi'] = multi_results
    
    # Train single currency models
    for currency in ['EURUSD', 'GBPUSD', 'USDJPY']:
        print("\n" + "="*70)
        print(f"TRAINING SINGLE CURRENCY MODEL: {currency}")
        print("="*70)
        single_results = train_single_model(currency, use_test_set=use_test_set)
        if single_results:
            all_results[currency] = single_results
    
    # Compare results for each currency pair
    print("\n" + "="*70)
    print("COMPARING MULTI VS SINGLE MODELS")
    print("="*70)
    
    currency_comparisons = {}
    for currency in ['EURUSD', 'GBPUSD', 'USDJPY']:
        comparison = compare_all_models(currency)
        if comparison:
            currency_comparisons[currency] = comparison
    
    # Create comprehensive summary table
    if currency_comparisons:
        config = Config()
        visualizer = ForexVisualizer(config)
        visualizer.create_performance_summary_table(currency_comparisons)
    
    print("\n✅ ALL MODELS TRAINED AND COMPARED SUCCESSFULLY!")
    return all_results

def print_detailed_results(eval_metrics, trading_results, config):
    """Print detailed results summary"""
    
    print("\n📈 MODEL PERFORMANCE SUMMARY")
    print("-" * 50)
    print(f"Model Type: {config.MODEL_TYPE}")
    print(f"Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {eval_metrics.get('precision', 0):.4f}")
    print(f"Recall: {eval_metrics.get('recall', 0):.4f}")
    print(f"F1-Score: {eval_metrics.get('f1_score', 0):.4f}")
    
    # Print advanced metrics if available
    if 'auc_roc' in eval_metrics:
        print(f"AUC-ROC: {eval_metrics.get('auc_roc', 0):.4f}")
    if 'log_loss' in eval_metrics:
        print(f"Log Loss: {eval_metrics.get('log_loss', 0):.4f}")
    
    print("\n💼 TRADING STRATEGY PERFORMANCE")
    print("-" * 100)
    print(f"{'Strategy':<25} {'Trades':<8} {'Return%':<10} {'Win Rate':<10} {'Sharpe':<10} {'Max DD%':<10} {'Final $':<12}")
    print("-" * 100)
    
    for strategy_name, result in trading_results.items():
        if isinstance(result, dict):
            if 'performance' in result:
                perf = result['performance']
                portfolio_manager = result.get('portfolio_manager')
                final_capital = portfolio_manager.capital if portfolio_manager else config.INITIAL_CAPITAL
                
                print(f"{strategy_name:<25} {perf.get('total_trades', 0):<8} "
                      f"{perf.get('total_return_pct', 0):<10.2f} {perf.get('win_rate', 0):<10.4f} "
                      f"{perf.get('sharpe_ratio', 0):<10.4f} {perf.get('max_drawdown_pct', 0):<10.2f} "
                      f"${final_capital:<11,.2f}")
            else:
                # Handle baseline strategies
                perf = result
                final_capital = perf.get('final_capital', config.INITIAL_CAPITAL)
                sharpe_ratio = perf.get('sharpe_ratio', 0)
                
                print(f"{result.get('strategy_name', strategy_name):<25} "
                      f"{perf.get('total_trades', 0):<8} "
                      f"{perf.get('total_return_pct', 0):<10.2f} "
                      f"{perf.get('win_rate', 0):<10.4f} "
                      f"{sharpe_ratio:<10.4f} "
                      f"{perf.get('max_drawdown_pct', 0):<10.2f} "
                      f"${final_capital:<11,.2f}")
    
    print("-" * 100)
    
    # Find best performing strategy
    best_strategy = None
    best_return = -float('inf')
    
    for strategy_name, result in trading_results.items():
        if isinstance(result, dict):
            if 'performance' in result:
                current_return = result['performance'].get('total_return_pct', 0)
            else:
                current_return = result.get('total_return_pct', 0)
            
            if current_return > best_return:
                best_return = current_return
                best_strategy = strategy_name
    
    print(f"\n🏆 Best performing strategy: {best_strategy} (Return: {best_return:.2f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Currency CNN-LSTM Forex Prediction')
    parser.add_argument('--model', type=str, choices=['multi', 'EURUSD', 'GBPUSD', 'USDJPY', 'all'], 
                       default='all', help='Model type to train')
    parser.add_argument('--step', type=int, choices=range(1, 7), 
                       help='Start from specific step (1-6)')
    parser.add_argument('--test', action='store_true', 
                       help='Use test set for final evaluation')
    parser.add_argument('--compare', type=str, choices=['EURUSD', 'GBPUSD', 'USDJPY'],
                       help='Compare multi vs single model for specific currency')
    parser.add_argument('--new', action='store_true',
                       help='Start fresh (delete existing checkpoint)')
    
    args = parser.parse_args()
    
    # If comparison is requested
    if args.compare:
        compare_all_models(args.compare)
        exit()
    
    # Delete checkpoint if starting fresh
    if args.new and args.model != 'all':
        config = Config(model_type=args.model)
        checkpoint_manager = CheckpointManager(config)
        checkpoint_manager.delete_checkpoint()
    
    # Run training
    if args.model == 'all':
        train_all_models(use_test_set=args.test)
    else:
        train_single_model(args.model, start_from_step=args.step, use_test_set=args.test)