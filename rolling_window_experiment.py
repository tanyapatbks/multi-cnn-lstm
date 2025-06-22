"""
Rolling Window Experiment - 12 Loops Complete Analysis with Test Set Support
Updated to include Test Set in rolling window and enhanced monthly reporting
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
    """Complete Rolling Window Experiment Manager for 12 Loops with Test Set Support"""
    
    def __init__(self, base_config: Config):
        self.base_config = base_config
        self.all_loops_results = {}  # {loop_num: {strategy_name: performance_dict}}
        self.rolling_schedule = self._create_rolling_schedule()
        
        # Results storage
        self.results_path = 'results/rolling_window_12_loops_with_test_set/'
        os.makedirs(self.results_path, exist_ok=True)
        
        # Leverage configuration
        self.leverage_info = {
            'Conservative': 2.0,
            'Moderate': 1.0,
            'Aggressive': 0.5
        }
        
        # Monthly periods for detailed analysis
        self.monthly_periods = [
            ('2021-01-01', '2021-01-31', 'Jan 2021'),
            ('2021-02-01', '2021-02-28', 'Feb 2021'),
            ('2021-03-01', '2021-03-31', 'Mar 2021'),
            ('2021-04-01', '2021-04-30', 'Apr 2021'),
            ('2021-05-01', '2021-05-31', 'May 2021'),
            ('2021-06-01', '2021-06-30', 'Jun 2021'),
            ('2021-07-01', '2021-07-31', 'Jul 2021'),
            ('2021-08-01', '2021-08-31', 'Aug 2021'),
            ('2021-09-01', '2021-09-30', 'Sep 2021'),
            ('2021-10-01', '2021-10-31', 'Oct 2021'),
            ('2021-11-01', '2021-11-30', 'Nov 2021'),
            ('2021-12-01', '2021-12-31', 'Dec 2021')
        ]
        
    def _create_rolling_schedule(self) -> List[Dict]:
        """สร้างตารางเวลาสำหรับ 12 loops พร้อม Test Set"""
        schedule = [
            {'loop': 1,  'train_start': '2018-12-01', 'train_end': '2020-11-30', 'val_start': '2020-12-01', 'val_end': '2020-12-31', 'test_start': '2021-01-01', 'test_end': '2021-01-31'},
            {'loop': 2,  'train_start': '2019-01-01', 'train_end': '2020-12-31', 'val_start': '2021-01-01', 'val_end': '2021-01-31', 'test_start': '2021-02-01', 'test_end': '2021-02-28'},
            {'loop': 3,  'train_start': '2019-02-01', 'train_end': '2021-01-31', 'val_start': '2021-02-01', 'val_end': '2021-02-28', 'test_start': '2021-03-01', 'test_end': '2021-03-31'},
            {'loop': 4,  'train_start': '2019-03-01', 'train_end': '2021-02-28', 'val_start': '2021-03-01', 'val_end': '2021-03-31', 'test_start': '2021-04-01', 'test_end': '2021-04-30'},
            {'loop': 5,  'train_start': '2019-04-01', 'train_end': '2021-03-31', 'val_start': '2021-04-01', 'val_end': '2021-04-30', 'test_start': '2021-05-01', 'test_end': '2021-05-31'},
            {'loop': 6,  'train_start': '2019-05-01', 'train_end': '2021-04-30', 'val_start': '2021-05-01', 'val_end': '2021-05-31', 'test_start': '2021-06-01', 'test_end': '2021-06-30'},
            {'loop': 7,  'train_start': '2019-06-01', 'train_end': '2021-05-31', 'val_start': '2021-06-01', 'val_end': '2021-06-30', 'test_start': '2021-07-01', 'test_end': '2021-07-31'},
            {'loop': 8,  'train_start': '2019-07-01', 'train_end': '2021-06-30', 'val_start': '2021-07-01', 'val_end': '2021-07-31', 'test_start': '2021-08-01', 'test_end': '2021-08-31'},
            {'loop': 9,  'train_start': '2019-08-01', 'train_end': '2021-07-31', 'val_start': '2021-08-01', 'val_end': '2021-08-31', 'test_start': '2021-09-01', 'test_end': '2021-09-30'},
            {'loop': 10, 'train_start': '2019-09-01', 'train_end': '2021-08-31', 'val_start': '2021-09-01', 'val_end': '2021-09-30', 'test_start': '2021-10-01', 'test_end': '2021-10-31'},
            {'loop': 11, 'train_start': '2019-10-01', 'train_end': '2021-09-30', 'val_start': '2021-10-01', 'val_end': '2021-10-31', 'test_start': '2021-11-01', 'test_end': '2021-11-30'},
            {'loop': 12, 'train_start': '2019-11-01', 'train_end': '2021-10-31', 'val_start': '2021-11-01', 'val_end': '2021-11-30', 'test_start': '2021-12-01', 'test_end': '2021-12-31'}
        ]
        return schedule
    
    def run_complete_experiment(self, use_test_set=True, threshold_choice='Moderate'):
        """
        รัน experiment ครบ 12 loops พร้อม Test Set support
        
        Args:
            use_test_set: ถ้า True จะใช้ Test Set สำหรับ evaluation (แทน Validation Set)
            threshold_choice: เลือก threshold strategy ('Conservative', 'Moderate', 'Aggressive')
        """
        print("="*100)
        print("🚀 STARTING COMPLETE 12-LOOP ROLLING WINDOW EXPERIMENT WITH TEST SET")
        print("="*100)
        eval_set_name = "TEST SET" if use_test_set else "VALIDATION SET"
        print(f"📊 Evaluation Mode: {eval_set_name}")
        print(f"🎯 Threshold Strategy: {threshold_choice}")
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
        
        # Run all loops
        for loop_info in self.rolling_schedule:
            eval_period = f"{loop_info['test_start']} to {loop_info['test_end']}" if use_test_set else f"{loop_info['val_start']} to {loop_info['val_end']}"
            print(f"\n{'='*60}")
            print(f"🔄 LOOP {loop_info['loop']}: {eval_period}")
            print(f"{'='*60}")
            
            loop_results = self._run_single_loop(
                loop_info, processed_data, use_test_set, threshold_choice
            )
            self.all_loops_results[loop_info['loop']] = loop_results
            
            # Progress summary with leverage info
            total_strategies = len(loop_results)
            cnn_lstm_strategies = sum(1 for name in loop_results.keys() if 'CNN-LSTM' in name)
            print(f"✅ Loop {loop_info['loop']} completed:")
            print(f"   📊 Total strategies: {total_strategies}")
            print(f"   🤖 CNN-LSTM strategies (with leverage): {cnn_lstm_strategies}")
            print(f"   📈 Baseline strategies (1.0x leverage): {total_strategies - cnn_lstm_strategies}")
        
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
        
        print(f"\n🎉 EXPERIMENT COMPLETED!")
        print(f"📁 Results saved to: {self.results_path}")
        
    def _run_single_loop(self, loop_info: Dict, processed_data: Dict, use_test_set: bool, threshold_choice: str) -> Dict:
        """รัน single loop พร้อม Test Set support และ enhanced reporting"""
        
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
        
        # Train and evaluate CNN-LSTM models
        for variation in model_variations:
            try:
                model_results = self._train_and_evaluate_model(
                    variation, loop_info, processed_data, use_test_set, threshold_choice
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
    
    def _train_and_evaluate_model(self, variation: Dict, loop_info: Dict, 
                                 processed_data: Dict, use_test_set: bool, 
                                 threshold_choice: str) -> Dict:
        """Train และ evaluate โมเดล CNN-LSTM พร้อม enhanced prediction reporting"""
        
        model_type, target_pair = variation['model_type'], variation['target_pair']
        
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
        
        # Train model
        model = CNNLSTMModel(model_config)
        history = model.train((X_train, y_train), (X_eval, y_eval))
        
        # Get predictions with detailed analysis
        predictions = model.model.predict(X_eval).flatten()
        eval_prices = processed_data[target_pair]['Close'].reindex(ts_eval)
        
        # Enhanced prediction reporting
        self._print_prediction_analysis(predictions, model_type, target_pair, threshold_choice)
        
        results = {}
        
        # Generate signals and run trading simulation
        threshold_signals = get_cnn_lstm_signals_multiple_thresholds(model_config, predictions)
        signals = threshold_signals[threshold_choice]
        
        # Trading simulation with leverage
        simulator = TradingSimulator(model_config, eval_prices, strategy_threshold=threshold_choice)
        performance = simulator.run(signals)
        
        # Create strategy name
        model_family = 'Multi-CNN-LSTM' if model_type == 'multi' else 'Single-CNN-LSTM'
        strategy_key = f"{model_family}-{threshold_choice} ({target_pair})"
        results[strategy_key] = performance
        
        return results
    
    def _print_prediction_analysis(self, predictions: np.ndarray, model_type: str, 
                                 target_pair: str, threshold_choice: str):
        """Print detailed prediction analysis"""
        
        # Get threshold values from config
        config_temp = Config()
        thresholds = config_temp.THRESHOLDS[threshold_choice]
        buy_threshold = thresholds['buy']
        sell_threshold = thresholds['sell']
        
        # Calculate prediction statistics
        pred_min, pred_max = predictions.min(), predictions.max()
        
        # Count signals
        buy_signals = np.sum(predictions >= buy_threshold)
        sell_signals = np.sum(predictions <= sell_threshold)
        hold_signals = len(predictions) - buy_signals - sell_signals
        
        total_predictions = len(predictions)
        buy_pct = (buy_signals / total_predictions) * 100
        sell_pct = (sell_signals / total_predictions) * 100
        hold_pct = (hold_signals / total_predictions) * 100
        
        model_name = f"{'Multi' if model_type == 'multi' else 'Single'}-CNN-LSTM"
        
        print(f"   🎯 {model_name} ({target_pair}) - {threshold_choice} Strategy:")
        print(f"      • Prediction range: [{pred_min:.3f}, {pred_max:.3f}]")
        print(f"      • Trading signals generated:")
        print(f"        - HOLD: {hold_signals} ({hold_pct:.1f}%)")
        print(f"        - SELL: {sell_signals} ({sell_pct:.1f}%)")
        print(f"        - BUY: {buy_signals} ({buy_pct:.1f}%)")
    
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
            technical_indicators = processor.get_technical_indicators_for_baseline(processed_data, currency_pair, eval_start, eval_end)
            sim = TradingSimulator(run_config, eval_prices, strategy_threshold='Moderate')
            performance = sim.run(get_rsi_signals(run_config, technical_indicators['RSI']))
            baseline_results[f'RSI Based ({currency_pair})'] = performance
            print(f"      ✅ RSI Strategy: Return={performance['total_return_pct']:.2f}%")
        except Exception as e:
            print(f"⚠️ Error calculating RSI strategy for {currency_pair}: {e}")
        
        try:
            # MACD-based Strategy
            sim = TradingSimulator(run_config, eval_prices, strategy_threshold='Moderate')
            performance = sim.run(get_macd_signals(run_config, technical_indicators['MACD'], technical_indicators['MACD_Signal']))
            baseline_results[f'MACD Based ({currency_pair})'] = performance
            print(f"      ✅ MACD Strategy: Return={performance['total_return_pct']:.2f}%")
        except Exception as e:
            print(f"⚠️ Error calculating MACD strategy for {currency_pair}: {e}")
        
        return baseline_results
    
    def _generate_monthly_return_table(self):
        """สร้างตาราง Monthly Return% ของทั้ง 3 คู่เงิน"""
        print("\n📊 Generating Monthly Return% Tables...")
        
        currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        for currency in currency_pairs:
            monthly_data = []
            
            for month_start, month_end, month_name in self.monthly_periods:
                # Find the corresponding loop for this month
                corresponding_loop = None
                for loop_num, loop_results in self.all_loops_results.items():
                    loop_info = self.rolling_schedule[loop_num - 1]
                    if loop_info['test_start'] == month_start:
                        corresponding_loop = loop_num
                        break
                
                if corresponding_loop is None:
                    continue
                
                loop_results = self.all_loops_results[corresponding_loop]
                row_data = {'Month': month_name}
                
                # Extract results for this currency
                strategies = [
                    f'Multi-CNN-LSTM-Conservative ({currency})',
                    f'Multi-CNN-LSTM-Moderate ({currency})',
                    f'Multi-CNN-LSTM-Aggressive ({currency})',
                    f'Single-CNN-LSTM ({currency})',
                    f'Buy & Hold ({currency})',
                    f'RSI Based ({currency})',
                    f'MACD Based ({currency})'
                ]
                
                for strategy in strategies:
                    # Find matching result
                    matching_result = None
                    for result_name, performance in loop_results.items():
                        if strategy in result_name or (strategy.split('(')[0].strip() in result_name and currency in result_name):
                            matching_result = performance
                            break
                    
                    strategy_short = strategy.split('(')[0].strip()
                    if matching_result:
                        row_data[strategy_short] = f"{matching_result.get('total_return_pct', 0):.2f}%"
                    else:
                        row_data[strategy_short] = "N/A"
                
                monthly_data.append(row_data)
            
            # Create DataFrame and save
            if monthly_data:
                df = pd.DataFrame(monthly_data)
                file_path = os.path.join(self.results_path, f'Monthly_Return_Table_{currency}.csv')
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"✅ Monthly Return table saved for {currency}: {file_path}")
    
    def _generate_monthly_sharpe_table(self):
        """สร้างตาราง Monthly Sharpe Ratio ของทั้ง 3 คู่เงิน"""
        print("\n📊 Generating Monthly Sharpe Ratio Tables...")
        
        currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        for currency in currency_pairs:
            monthly_data = []
            
            for month_start, month_end, month_name in self.monthly_periods:
                # Find the corresponding loop for this month
                corresponding_loop = None
                for loop_num, loop_results in self.all_loops_results.items():
                    loop_info = self.rolling_schedule[loop_num - 1]
                    if loop_info['test_start'] == month_start:
                        corresponding_loop = loop_num
                        break
                
                if corresponding_loop is None:
                    continue
                
                loop_results = self.all_loops_results[corresponding_loop]
                row_data = {'Month': month_name}
                
                # Extract results for this currency
                strategies = [
                    f'Multi-CNN-LSTM-Conservative ({currency})',
                    f'Multi-CNN-LSTM-Moderate ({currency})',
                    f'Multi-CNN-LSTM-Aggressive ({currency})',
                    f'Single-CNN-LSTM ({currency})',
                    f'Buy & Hold ({currency})',
                    f'RSI Based ({currency})',
                    f'MACD Based ({currency})'
                ]
                
                for strategy in strategies:
                    # Find matching result
                    matching_result = None
                    for result_name, performance in loop_results.items():
                        if strategy in result_name or (strategy.split('(')[0].strip() in result_name and currency in result_name):
                            matching_result = performance
                            break
                    
                    strategy_short = strategy.split('(')[0].strip()
                    if matching_result:
                        row_data[strategy_short] = f"{matching_result.get('sharpe_ratio', 0):.3f}"
                    else:
                        row_data[strategy_short] = "N/A"
                
                monthly_data.append(row_data)
            
            # Create DataFrame and save
            if monthly_data:
                df = pd.DataFrame(monthly_data)
                file_path = os.path.join(self.results_path, f'Monthly_Sharpe_Table_{currency}.csv')
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"✅ Monthly Sharpe table saved for {currency}: {file_path}")
    
    def _generate_annual_summary_table(self):
        """สร้างตาราง Annual Summary แยกตามคู่เงิน (3 ตาราง)"""
        print("\n📊 Generating Annual Summary Tables by Currency Pair...")
        
        currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        for currency in currency_pairs:
            print(f"\n   📈 Processing Annual Summary for {currency}...")
            
            # Collect data for this currency
            currency_data = {
                'Multi-CNN-LSTM Conservative': [],
                'Multi-CNN-LSTM Moderate': [],
                'Multi-CNN-LSTM Aggressive': [],
                'Single-CNN-LSTM': [],
                'Buy & Hold': [],
                'RSI Based': [],
                'MACD Based': []
            }
            
            # Extract results for this currency across all loops
            for loop_num, loop_results in self.all_loops_results.items():
                for strategy_name, performance in loop_results.items():
                    # Check if this strategy belongs to the current currency
                    if f'({currency})' in strategy_name:
                        # Determine strategy type
                        if 'Multi-CNN-LSTM-Conservative' in strategy_name:
                            currency_data['Multi-CNN-LSTM Conservative'].append(performance)
                        elif 'Multi-CNN-LSTM-Moderate' in strategy_name:
                            currency_data['Multi-CNN-LSTM Moderate'].append(performance)
                        elif 'Multi-CNN-LSTM-Aggressive' in strategy_name:
                            currency_data['Multi-CNN-LSTM Aggressive'].append(performance)
                        elif 'Single-CNN-LSTM' in strategy_name:
                            currency_data['Single-CNN-LSTM'].append(performance)
                        elif 'Buy & Hold' in strategy_name:
                            currency_data['Buy & Hold'].append(performance)
                        elif 'RSI Based' in strategy_name:
                            currency_data['RSI Based'].append(performance)
                        elif 'MACD Based' in strategy_name:
                            currency_data['MACD Based'].append(performance)
            
            # Calculate metrics for each strategy
            summary_row = {}
            
            for strategy_type, performances in currency_data.items():
                if performances:
                    # Calculate aggregated metrics
                    total_returns = [p.get('total_return_pct', 0) for p in performances]
                    sharpe_ratios = [p.get('sharpe_ratio', 0) for p in performances]
                    win_rates = [p.get('win_rate', 0) for p in performances]
                    max_drawdowns = [p.get('max_drawdown_pct', 0) for p in performances]
                    total_trades = [p.get('total_trades', 0) for p in performances]
                    
                    summary_row[strategy_type] = {
                        'Total Return (%)': np.sum(total_returns),  # Sum across 12 months
                        'Avg. Sharpe Ratio': np.mean(sharpe_ratios),
                        'Avg. Win Rate (%)': np.mean(win_rates) * 100,
                        'Avg. Max Drawdown (%)': np.mean(max_drawdowns),
                        'Avg. Total Trades': np.mean(total_trades)
                    }
                else:
                    # No data available
                    summary_row[strategy_type] = {
                        'Total Return (%)': 0,
                        'Avg. Sharpe Ratio': 0,
                        'Avg. Win Rate (%)': 0,
                        'Avg. Max Drawdown (%)': 0,
                        'Avg. Total Trades': 0
                    }
            
            # Create DataFrame in the desired format
            metrics = ['Total Return (%)', 'Avg. Sharpe Ratio', 'Avg. Win Rate (%)', 
                      'Avg. Max Drawdown (%)', 'Avg. Total Trades']
            
            table_data = []
            for metric in metrics:
                row = {'Metric': metric}
                for strategy_type in ['Multi-CNN-LSTM Conservative', 'Multi-CNN-LSTM Moderate', 
                                    'Multi-CNN-LSTM Aggressive', 'Single-CNN-LSTM', 
                                    'Buy & Hold', 'RSI Based', 'MACD Based']:
                    row[strategy_type] = summary_row[strategy_type][metric]
                table_data.append(row)
            
            # Create DataFrame and format
            df = pd.DataFrame(table_data)
            df = df.round(3)
            
            # Save to CSV
            file_path = os.path.join(self.results_path, f'Annual_Summary_{currency}.csv')
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            print(f"      ✅ Annual Summary saved for {currency}: {file_path}")
            
            # Display table
            print(f"\n      📋 ANNUAL SUMMARY TABLE - {currency}:")
            print(f"      {df.to_string(index=False)}")
        
        print(f"\n✅ All Annual Summary tables generated successfully!")
    
    # Keep existing methods for backward compatibility
    def _generate_summary_tables(self):
        """Generate summary tables (kept for compatibility)"""
        pass
    
    def _generate_detailed_tables(self):
        """Generate detailed tables (kept for compatibility)"""
        pass
    
    def _generate_leverage_analysis(self):
        """Generate leverage analysis (kept for compatibility)"""
        pass
    
    def _generate_visualizations(self):
        """Generate visualizations (kept for compatibility)"""
        pass
    
    def _get_month_name(self, loop_num: int) -> str:
        """Get month name for loop number"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        return months[(loop_num - 1) % 12]