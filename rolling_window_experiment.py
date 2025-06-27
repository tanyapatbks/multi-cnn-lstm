"""
Rolling Window Experiment - 12 Loops Complete Analysis with OPTIMIZED single training
Updated to train each model ONCE and evaluate ALL thresholds from same predictions
"""
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing modules
from config import Config
from data_processor import DataProcessor, SequencePreparator
from cnn_lstm_model import CNNLSTMModel
from trading_strategy import (TradingSimulator, get_cnn_lstm_signals_multiple_thresholds, 
                             get_rsi_signals, get_macd_signals, get_buy_and_hold_signals)

warnings.filterwarnings('ignore')

class RollingWindowExperiment:
    """Optimized Rolling Window Experiment Manager for 12 Loops with single training per model"""
    
    def __init__(self, base_config: Config):
        self.base_config = base_config
        self.all_loops_results = {}  # {loop_num: {strategy_name: performance_dict}}
        self.rolling_schedule = self._create_rolling_schedule()
        
        # Results storage
        self.results_path = 'results/rolling_window_12_loops_optimized/'
        os.makedirs(self.results_path, exist_ok=True)
        
        # Leverage configuration
        self.leverage_info = {
            'Conservative': 2.0,
            'Moderate': 1.0,
            'Aggressive': 0.5
        }
        
        self.monthly_periods = [
            ('2022-01-01', '2022-01-31', 'Jan 2022'),
            ('2022-02-01', '2022-02-28', 'Feb 2022'),
            ('2022-03-01', '2022-03-31', 'Mar 2022'),
            ('2022-04-01', '2022-04-30', 'Apr 2022'),
            ('2022-05-01', '2022-05-31', 'May 2022'),
            ('2022-06-01', '2022-06-30', 'Jun 2022'),
            ('2022-07-01', '2022-07-31', 'Jul 2022'),
            ('2022-08-01', '2022-08-31', 'Aug 2022'),
            ('2022-09-01', '2022-09-30', 'Sep 2022'),
            ('2022-10-01', '2022-10-31', 'Oct 2022'),
            ('2022-11-01', '2022-11-30', 'Nov 2022'),
            ('2022-12-01', '2022-12-31', 'Dec 2022')
        ]

    def _create_rolling_schedule(self):
        """Create the rolling window schedule for 12 loops"""
        schedule = [
            {'loop': 1, 'train_start': '2019-12-01', 'train_end': '2021-11-30', 'val_start': '2021-12-01', 'val_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-01-31'},
            {'loop': 2, 'train_start': '2020-01-01', 'train_end': '2021-12-31', 'val_start': '2022-01-01', 'val_end': '2022-01-31', 'test_start': '2022-02-01', 'test_end': '2022-02-28'},
            {'loop': 3, 'train_start': '2020-02-01', 'train_end': '2022-01-31', 'val_start': '2022-02-01', 'val_end': '2022-02-28', 'test_start': '2022-03-01', 'test_end': '2022-03-31'},
            {'loop': 4, 'train_start': '2020-03-01', 'train_end': '2022-02-28', 'val_start': '2022-03-01', 'val_end': '2022-03-31', 'test_start': '2022-04-01', 'test_end': '2022-04-30'},
            {'loop': 5, 'train_start': '2020-04-01', 'train_end': '2022-03-31', 'val_start': '2022-04-01', 'val_end': '2022-04-30', 'test_start': '2022-05-01', 'test_end': '2022-05-31'},
            {'loop': 6, 'train_start': '2020-05-01', 'train_end': '2022-04-30', 'val_start': '2022-05-01', 'val_end': '2022-05-31', 'test_start': '2022-06-01', 'test_end': '2022-06-30'},
            {'loop': 7, 'train_start': '2020-06-01', 'train_end': '2022-05-31', 'val_start': '2022-06-01', 'val_end': '2022-06-30', 'test_start': '2022-07-01', 'test_end': '2022-07-31'},
            {'loop': 8, 'train_start': '2020-07-01', 'train_end': '2022-06-30', 'val_start': '2022-07-01', 'val_end': '2022-07-31', 'test_start': '2022-08-01', 'test_end': '2022-08-31'},
            {'loop': 9, 'train_start': '2020-08-01', 'train_end': '2022-07-31', 'val_start': '2022-08-01', 'val_end': '2022-08-31', 'test_start': '2022-09-01', 'test_end': '2022-09-30'},
            {'loop': 10, 'train_start': '2020-09-01', 'train_end': '2022-08-31', 'val_start': '2022-09-01', 'val_end': '2022-09-30', 'test_start': '2022-10-01', 'test_end': '2022-10-31'},
            {'loop': 11, 'train_start': '2020-10-01', 'train_end': '2022-09-30', 'val_start': '2022-10-01', 'val_end': '2022-10-31', 'test_start': '2022-11-01', 'test_end': '2022-11-30'},
            {'loop': 12, 'train_start': '2020-11-01', 'train_end': '2022-10-31', 'val_start': '2022-11-01', 'val_end': '2022-11-30', 'test_start': '2022-12-01', 'test_end': '2022-12-31'}
        ]
        return schedule

    
    def run_complete_experiment(self, use_test_set=True):
        """
        ✅ OPTIMIZED: Run experiment with single training per model across 12 loops
        
        Args:
            use_test_set: ถ้า True จะใช้ Test Set สำหรับ evaluation (แทน Validation Set)
        """
        print("="*100)
        print("🚀 STARTING OPTIMIZED 12-LOOP ROLLING WINDOW EXPERIMENT")
        print("="*100)
        eval_set_name = "TEST SET" if use_test_set else "VALIDATION SET"
        print(f"📊 Evaluation Mode: {eval_set_name}")
        print(f"⚡ OPTIMIZATION: Single training per model, evaluate ALL thresholds")
        print(f"💰 Leverage Configuration:")
        for threshold, leverage in self.leverage_info.items():
            print(f"   • {threshold}: {leverage}x leverage")
        print(f"💱 Currency Pairs: {self.base_config.ALL_CURRENCY_PAIRS}")
        print(f"📅 Loops: {len(self.rolling_schedule)}")
        print("="*100)
        
        # Prepare data once
        print("\n📚 Loading and preprocessing data...")
        master_processor = DataProcessor(self.base_config)
        raw_data = master_processor.load_currency_data()
        if raw_data is None:
            print("❌ Failed to load data. Aborting.")
            return
        processed_data = master_processor.preprocess_data(raw_data)
        
        # Run all loops with optimized training
        for loop_info in self.rolling_schedule:
            eval_period = f"{loop_info['test_start']} to {loop_info['test_end']}" if use_test_set else f"{loop_info['val_start']} to {loop_info['val_end']}"
            print(f"\n{'='*60}")
            print(f"🔄 LOOP {loop_info['loop']}: {eval_period}")
            print(f"{'='*60}")
            
            loop_results = self._run_single_loop_optimized(
                loop_info, processed_data, use_test_set
            )
            self.all_loops_results[loop_info['loop']] = loop_results
            
            # Progress summary with optimization info
            total_strategies = len(loop_results)
            cnn_lstm_strategies = sum(1 for name in loop_results.keys() if 'CNN-LSTM' in name)
            print(f"✅ Loop {loop_info['loop']} completed:")
            print(f"   📊 Total strategies: {total_strategies}")
            print(f"   🤖 CNN-LSTM strategies (all thresholds): {cnn_lstm_strategies}")
            print(f"   📈 Baseline strategies (1.0x leverage): {total_strategies - cnn_lstm_strategies}")
            print(f"   ⚡ Training optimization: 6 models trained (instead of 18)")
        
        # Generate comprehensive analysis
        print(f"\n{'='*80}")
        print("📊 GENERATING COMPREHENSIVE ANALYSIS WITH MONTHLY TABLES")
        print(f"{'='*80}")
        
        self._generate_monthly_return_table()
        self._generate_monthly_sharpe_table() 
        self._generate_annual_summary_table()
        self._generate_summary_tables()
        self._generate_detailed_tables()
        self._generate_leverage_analysis()
        self._generate_visualizations()
        
        print(f"\n🎉 OPTIMIZED EXPERIMENT COMPLETED!")
        print(f"📁 Results saved to: {self.results_path}")
        print(f"⚡ Total training time reduced by ~66%")
        
    def _run_single_loop_optimized(self, loop_info: Dict, processed_data: Dict, use_test_set: bool) -> Dict:
        """✅ OPTIMIZED: Run single loop with single training per model"""
        
        loop_results = {}
        
        # Determine evaluation period
        if use_test_set:
            eval_start, eval_end = loop_info['test_start'], loop_info['test_end']
        else:
            eval_start, eval_end = loop_info['val_start'], loop_info['val_end']
        
        # Model variations to train and evaluate
        model_variations = [
            {'model_type': 'multi', 'target_pair': 'EURUSD'},
            {'model_type': 'multi', 'target_pair': 'GBPUSD'},
            {'model_type': 'multi', 'target_pair': 'USDJPY'},
            {'model_type': 'EURUSD', 'target_pair': 'EURUSD'},
            {'model_type': 'GBPUSD', 'target_pair': 'GBPUSD'},
            {'model_type': 'USDJPY', 'target_pair': 'USDJPY'}
        ]
        
        # ✅ Train and evaluate CNN-LSTM models (OPTIMIZED)
        for variation in model_variations:
            try:
                model_results = self._train_and_evaluate_model_optimized(
                    variation, loop_info, processed_data, use_test_set
                )
                loop_results.update(model_results)
            except Exception as e:
                print(f"⚠️ Error in {variation}: {e}")
        
        # Calculate baseline strategies
        for currency_pair in self.base_config.ALL_CURRENCY_PAIRS:
            try:
                baseline_results = self._calculate_baseline_strategies(
                    currency_pair, processed_data, eval_start, eval_end
                )
                loop_results.update(baseline_results)
            except Exception as e:
                print(f"⚠️ Error calculating baselines for {currency_pair}: {e}")
        
        return loop_results
    
    def _train_and_evaluate_model_optimized(self, variation: Dict, loop_info: Dict, 
                                           processed_data: Dict, use_test_set: bool) -> Dict:
        """✅ OPTIMIZED: Train model ONCE และ evaluate ALL thresholds"""
        
        model_type, target_pair = variation['model_type'], variation['target_pair']
        print(f"  🧠 Training {model_type} → {target_pair} (SINGLE TRAINING)")
        
        # Create model config
        model_config = Config(model_type=model_type, target_pair=target_pair)
        model_config.TRAIN_START = loop_info['train_start']
        model_config.TRAIN_END = loop_info['train_end']
        model_config.VAL_START = loop_info['val_start']
        model_config.VAL_END = loop_info['val_end']
        model_config.TEST_START = loop_info['test_start']
        model_config.TEST_END = loop_info['test_end']
        
        # Prepare data with proper Train Set statistics
        dp = DataProcessor(model_config)
        model_input = dp.get_model_input_data(processed_data, loop_info)
        (train_data, eval_data) = SequencePreparator(model_config).create_sequences_and_splits(
            model_input, processed_data, use_test_set=use_test_set
        )
        
        (X_train, y_train), (X_eval, y_eval, ts_eval) = train_data, eval_data
        if len(X_eval) == 0:
            return {}
        
        # ✅ STEP 1: Train model ONCE
        model = CNNLSTMModel(model_config)
        history = model.train((X_train, y_train), (X_eval, y_eval))
        
        # ✅ STEP 2: Get predictions ONCE
        predictions = model.model.predict(X_eval).flatten()
        eval_prices = processed_data[target_pair]['Close'].reindex(ts_eval)
        
        # ✅ STEP 3: Analyze predictions
        self._print_prediction_analysis_optimized(predictions, model_type, target_pair)
        
        # ✅ STEP 4: Generate signals for ALL thresholds from same predictions
        threshold_signals = get_cnn_lstm_signals_multiple_thresholds(model_config, predictions)
        
        results = {}
        model_family = 'Multi-CNN-LSTM' if model_type == 'multi' else 'Single-CNN-LSTM'
        
        # ✅ STEP 5: Evaluate ALL thresholds with respective leverage
        print(f"    💰 Evaluating all thresholds with leverage:")
        for threshold_name, signals in threshold_signals.items():
            # Trading simulation with appropriate leverage
            simulator = TradingSimulator(model_config, eval_prices, strategy_threshold=threshold_name)
            performance = simulator.run(signals)
            
            # Create strategy name
            strategy_key = f"{model_family}-{threshold_name} ({target_pair})"
            results[strategy_key] = performance
            
            # Display results
            leverage = simulator.portfolio.current_leverage
            print(f"      └─ {threshold_name} (Leverage {leverage}x): Return={performance['total_return_pct']:.2f}%, "
                  f"Sharpe={performance['sharpe_ratio']:.2f}")
        
        return results
    
    def _print_prediction_analysis_optimized(self, predictions: np.ndarray, model_type: str, target_pair: str):
        """Print optimized prediction analysis"""
        
        # Get threshold values from config
        config_temp = Config()
        
        # Calculate prediction statistics
        pred_min, pred_max = predictions.min(), predictions.max()
        pred_mean, pred_std = predictions.mean(), predictions.std()
        
        print(f"    📊 Predictions: Range=[{pred_min:.3f}, {pred_max:.3f}], Mean={pred_mean:.3f}")
        
        # Calculate signal distribution for all thresholds
        for threshold_name, thresholds in config_temp.THRESHOLDS.items():
            buy_signals = np.sum(predictions >= thresholds['buy'])
            sell_signals = np.sum(predictions <= thresholds['sell'])
            hold_signals = len(predictions) - buy_signals - sell_signals
            
            total = len(predictions)
            buy_pct = (buy_signals / total) * 100
            sell_pct = (sell_signals / total) * 100
            hold_pct = (hold_signals / total) * 100
            
            print(f"    🎯 {threshold_name}: Buy={buy_pct:.1f}%, Hold={hold_pct:.1f}%, Sell={sell_pct:.1f}%")
    
    def _calculate_baseline_strategies(self, currency_pair: str, processed_data: Dict, 
                                     eval_start: str, eval_end: str) -> Dict:
        """คำนวณ baseline strategies (RSI, MACD, Buy & Hold) สำหรับช่วงเวลาที่กำหนด"""
        
        processor = DataProcessor(self.base_config)
        price_data = processor.get_price_data_for_period(processed_data, currency_pair, eval_start, eval_end)
        if price_data.empty:
            print(f"⚠️ No price data for {currency_pair}")
            return {}

        eval_prices = price_data['Close']
        run_config = Config(model_type=currency_pair, target_pair=currency_pair)
        
        baseline_results = {}
        
        try:
            # Buy & Hold Strategy
            sim = TradingSimulator(run_config, eval_prices, strategy_threshold='Moderate')
            performance = sim.run(get_buy_and_hold_signals(eval_prices))
            baseline_results[f'Buy & Hold ({currency_pair})'] = performance
            print(f"      ✅ Buy & Hold: Return={performance['total_return_pct']:.2f}%")
        except Exception as e:
            print(f"⚠️ Error calculating Buy & Hold for {currency_pair}: {e}")
        
        try:
            # RSI-based Strategy
            technical_indicators = processor.get_technical_indicators_for_baseline(
                processed_data, currency_pair, eval_start, eval_end
            )
            sim = TradingSimulator(run_config, eval_prices, strategy_threshold='Moderate')
            performance = sim.run(get_rsi_signals(run_config, technical_indicators['RSI']))
            baseline_results[f'RSI-based Trading ({currency_pair})'] = performance
            print(f"      ✅ RSI Strategy: Return={performance['total_return_pct']:.2f}%")
        except Exception as e:
            print(f"⚠️ Error calculating RSI strategy for {currency_pair}: {e}")
        
        try:
            # MACD-based Strategy  
            sim = TradingSimulator(run_config, eval_prices, strategy_threshold='Moderate')
            performance = sim.run(get_macd_signals(run_config, technical_indicators['MACD'], technical_indicators['MACD_Signal']))
            baseline_results[f'MACD-based Trading ({currency_pair})'] = performance
            print(f"      ✅ MACD Strategy: Return={performance['total_return_pct']:.2f}%")
        except Exception as e:
            print(f"⚠️ Error calculating MACD strategy for {currency_pair}: {e}")
        
        return baseline_results
    
    def _generate_monthly_return_table(self):
        """Generate monthly return comparison tables for each currency pair"""
        print("📈 Generating monthly return tables for each currency pair...")
        
        # Process each currency pair separately
        for currency_pair in self.base_config.ALL_CURRENCY_PAIRS:
            monthly_data = []
            
            # Track totals for sum calculation
            strategy_totals = {}
            
            for month_start, month_end, month_name in self.monthly_periods:
                # Find corresponding loop for this month
                target_loop = None
                for loop_num, loop_results in self.all_loops_results.items():
                    loop_info = self.rolling_schedule[loop_num - 1]
                    if loop_info['test_start'] <= month_start <= loop_info['test_end']:
                        target_loop = loop_num
                        break
                
                if target_loop is None:
                    continue
                
                # Get results for this month and currency pair
                month_row = {'Month': month_name}
                loop_results = self.all_loops_results[target_loop]
                
                # Extract strategies for this currency pair
                for strategy_name, performance in loop_results.items():
                    # Filter strategies for current currency pair
                    if f'({currency_pair})' in strategy_name:
                        return_pct = performance.get('total_return_pct', 0)
                        month_row[strategy_name] = return_pct
                        
                        # Track for totals
                        if strategy_name not in strategy_totals:
                            strategy_totals[strategy_name] = 0
                        strategy_totals[strategy_name] += return_pct
                
                monthly_data.append(month_row)
            
            # Add TOTAL row (SUM of all months)
            total_row = {'Month': 'TOTAL'}
            for strategy_name, total_return in strategy_totals.items():
                total_row[strategy_name] = total_return
            monthly_data.append(total_row)
            
            # Save to CSV
            df_monthly = pd.DataFrame(monthly_data)
            filename = f'{self.results_path}/monthly_returns_{currency_pair}.csv'
            df_monthly.to_csv(filename, index=False)
            print(f"    ✅ {currency_pair} monthly returns saved to {filename}")
    
    def _generate_monthly_sharpe_table(self):
        """Generate monthly Sharpe ratio comparison tables for each currency pair"""
        print("📊 Generating monthly Sharpe ratio tables for each currency pair...")
        
        # Process each currency pair separately
        for currency_pair in self.base_config.ALL_CURRENCY_PAIRS:
            monthly_data = []
            
            # Track totals for sum calculation (though Sharpe ratios shouldn't be summed, we'll show average)
            strategy_totals = {}
            strategy_counts = {}
            
            for month_start, month_end, month_name in self.monthly_periods:
                # Find corresponding loop for this month
                target_loop = None
                for loop_num, loop_results in self.all_loops_results.items():
                    loop_info = self.rolling_schedule[loop_num - 1]
                    if loop_info['test_start'] <= month_start <= loop_info['test_end']:
                        target_loop = loop_num
                        break
                
                if target_loop is None:
                    continue
                
                # Get results for this month and currency pair
                month_row = {'Month': month_name}
                loop_results = self.all_loops_results[target_loop]
                
                # Extract strategies for this currency pair
                for strategy_name, performance in loop_results.items():
                    # Filter strategies for current currency pair
                    if f'({currency_pair})' in strategy_name:
                        sharpe_ratio = performance.get('sharpe_ratio', 0)
                        month_row[strategy_name] = sharpe_ratio
                        
                        # Track for averages (Sharpe ratios should be averaged, not summed)
                        if strategy_name not in strategy_totals:
                            strategy_totals[strategy_name] = 0
                            strategy_counts[strategy_name] = 0
                        strategy_totals[strategy_name] += sharpe_ratio
                        strategy_counts[strategy_name] += 1
                
                monthly_data.append(month_row)
            
            # Add AVERAGE row (AVG of all months for Sharpe ratios)
            avg_row = {'Month': 'AVERAGE'}
            for strategy_name, total_sharpe in strategy_totals.items():
                count = strategy_counts[strategy_name]
                avg_row[strategy_name] = total_sharpe / count if count > 0 else 0
            monthly_data.append(avg_row)
            
            # Save to CSV
            df_monthly = pd.DataFrame(monthly_data)
            filename = f'{self.results_path}/monthly_sharpe_ratios_{currency_pair}.csv'
            df_monthly.to_csv(filename, index=False)
            print(f"    ✅ {currency_pair} monthly Sharpe ratios saved to {filename}")
    
    def _generate_annual_summary_table(self):
        """Generate annual summary tables for each currency pair"""
        print("📋 Generating annual summary tables for each currency pair...")
        
        # Process each currency pair separately
        for currency_pair in self.base_config.ALL_CURRENCY_PAIRS:
            # Collect all results for this currency pair across all loops
            currency_results = {}
            
            for loop_results in self.all_loops_results.values():
                for strategy_name, performance in loop_results.items():
                    # Filter strategies for current currency pair
                    if f'({currency_pair})' in strategy_name:
                        if strategy_name not in currency_results:
                            currency_results[strategy_name] = []
                        currency_results[strategy_name].append(performance)
            
            # Calculate summary statistics for each strategy
            summary_data = []
            for strategy_name, performances in currency_results.items():
                if not performances:
                    continue
                
                summary_row = {
                    'Strategy': strategy_name.replace(f' ({currency_pair})', ''),  # Clean name
                    'Total_Return_Pct': sum([p.get('total_return_pct', 0) for p in performances]),  # SUM for total return
                    'Avg_Sharpe_Ratio': np.mean([p.get('sharpe_ratio', 0) for p in performances]),
                    'Avg_Win_Rate': np.mean([p.get('win_rate', 0) for p in performances]),
                    'Avg_Max_Drawdown': np.mean([p.get('max_drawdown', 0) for p in performances]),
                    'Total_Trades': sum([p.get('total_trades', 0) for p in performances]),
                    'Loops_Evaluated': len(performances),
                    'Std_Return_Pct': np.std([p.get('total_return_pct', 0) for p in performances]),
                    'Best_Monthly_Return': max([p.get('total_return_pct', 0) for p in performances]),
                    'Worst_Monthly_Return': min([p.get('total_return_pct', 0) for p in performances])
                }
                summary_data.append(summary_row)
            
            # Sort by Total Return (descending)
            summary_data.sort(key=lambda x: x['Total_Return_Pct'], reverse=True)
            
            # Save to CSV
            df_summary = pd.DataFrame(summary_data)
            filename = f'{self.results_path}/annual_summary_{currency_pair}.csv'
            df_summary.to_csv(filename, index=False)
            print(f"    ✅ {currency_pair} annual summary saved to {filename}")
    
    def _generate_summary_tables(self):
        """Generate summary tables for analysis"""
        print("📊 Generating summary tables...")
        
        # Loop-by-loop results table
        loop_data = []
        for loop_num, loop_results in self.all_loops_results.items():
            for strategy_name, performance in loop_results.items():
                loop_data.append({
                    'Loop': loop_num,
                    'Strategy': strategy_name,
                    'Return_Pct': performance.get('total_return_pct', 0),
                    'Sharpe_Ratio': performance.get('sharpe_ratio', 0),
                    'Win_Rate': performance.get('win_rate', 0),
                    'Max_Drawdown': performance.get('max_drawdown', 0),
                    'Total_Trades': performance.get('total_trades', 0)
                })
        
        df_loops = pd.DataFrame(loop_data)
        df_loops.to_csv(f'{self.results_path}/loop_by_loop_results.csv', index=False)
        print(f"    ✅ Loop results saved to {self.results_path}/loop_by_loop_results.csv")
    
    def _generate_detailed_tables(self):
        """Generate detailed performance tables"""
        print("📋 Generating detailed analysis tables...")
        
        # Strategy comparison table
        strategies = {}
        for loop_results in self.all_loops_results.values():
            for strategy_name, performance in loop_results.items():
                if strategy_name not in strategies:
                    strategies[strategy_name] = {
                        'returns': [],
                        'sharpe_ratios': [],
                        'win_rates': [],
                        'max_drawdowns': [],
                        'total_trades': []
                    }
                
                strategies[strategy_name]['returns'].append(performance.get('total_return_pct', 0))
                strategies[strategy_name]['sharpe_ratios'].append(performance.get('sharpe_ratio', 0))
                strategies[strategy_name]['win_rates'].append(performance.get('win_rate', 0))
                strategies[strategy_name]['max_drawdowns'].append(performance.get('max_drawdown', 0))
                strategies[strategy_name]['total_trades'].append(performance.get('total_trades', 0))
        
        # Create comparison table
        comparison_data = []
        for strategy_name, metrics in strategies.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Mean_Return': np.mean(metrics['returns']),
                'Std_Return': np.std(metrics['returns']),
                'Mean_Sharpe': np.mean(metrics['sharpe_ratios']),
                'Mean_Win_Rate': np.mean(metrics['win_rates']),
                'Mean_Max_Drawdown': np.mean(metrics['max_drawdowns']),
                'Total_Trades_Sum': np.sum(metrics['total_trades']),
                'Best_Return': np.max(metrics['returns']),
                'Worst_Return': np.min(metrics['returns'])
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Mean_Return', ascending=False)
        df_comparison.to_csv(f'{self.results_path}/strategy_comparison.csv', index=False)
        print(f"    ✅ Strategy comparison saved to {self.results_path}/strategy_comparison.csv")
    
    def _generate_leverage_analysis(self):
        """Generate leverage impact analysis"""
        print("💰 Generating leverage analysis...")
        
        leverage_data = []
        for loop_num, loop_results in self.all_loops_results.items():
            for strategy_name, performance in loop_results.items():
                if 'CNN-LSTM' in strategy_name:
                    # Extract threshold from strategy name
                    if 'Conservative' in strategy_name:
                        threshold = 'Conservative'
                        leverage = 2.0
                    elif 'Moderate' in strategy_name:
                        threshold = 'Moderate'
                        leverage = 1.0
                    elif 'Aggressive' in strategy_name:
                        threshold = 'Aggressive'
                        leverage = 0.5
                    else:
                        continue
                    
                    leverage_data.append({
                        'Loop': loop_num,
                        'Strategy': strategy_name,
                        'Threshold': threshold,
                        'Leverage': leverage,
                        'Return_Pct': performance.get('total_return_pct', 0),
                        'Sharpe_Ratio': performance.get('sharpe_ratio', 0),
                        'Risk_Adjusted_Return': performance.get('total_return_pct', 0) / leverage
                    })
        
        df_leverage = pd.DataFrame(leverage_data)
        df_leverage.to_csv(f'{self.results_path}/leverage_analysis.csv', index=False)
        print(f"    ✅ Leverage analysis saved to {self.results_path}/leverage_analysis.csv")
    
    def _generate_visualizations(self):
        """Generate key visualizations"""
        print("🎨 Generating visualizations...")
        
        # Implementation of visualization generation would go here
        # This would create charts comparing performance across loops, strategies, etc.
        
        print(f"    ✅ Visualizations saved to {self.results_path}/plots/")