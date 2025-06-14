"""
Experiment Runner for Master's Thesis
Runs 12 monthly loops as per Advisor's requirements

Loop 1: Train Dec 2018 - Nov 2020 / Validate Dec 2020 / Test Jan 2021
Loop 2: Train Jan 2019 - Dec 2020 / Validate Jan 2021 / Test Feb 2021
... and so on for 12 loops
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import json

def generate_date_ranges():
    """Generate 12 sets of date ranges for training/validation/testing"""
    date_ranges = []
    
    # Starting points
    train_start = datetime(2018, 12, 1)
    
    for loop in range(12):
        # Calculate dates for this loop
        loop_train_start = train_start + timedelta(days=30*loop)  # Approximate monthly shift
        loop_train_end = loop_train_start + timedelta(days=730)  # ~2 years
        
        # Validation is the next month after training
        loop_val_start = loop_train_end + timedelta(days=1)
        loop_val_end = loop_val_start + timedelta(days=30)  # 1 month validation
        
        # Test is the next month after validation
        loop_test_start = loop_val_end + timedelta(days=1)
        loop_test_end = loop_test_start + timedelta(days=30)  # 1 month test
        
        # Adjust to exact month boundaries
        loop_config = {
            'loop': loop + 1,
            'train_start': loop_train_start.strftime('%Y-%m-01'),
            'train_end': loop_train_end.strftime('%Y-%m-%d'),
            'val_start': loop_val_start.strftime('%Y-%m-01'),
            'val_end': loop_val_end.strftime('%Y-%m-%d'),
            'test_start': loop_test_start.strftime('%Y-%m-01'),
            'test_end': loop_test_end.strftime('%Y-%m-%d')
        }
        
        # Manual adjustment for exact dates as per requirement
        if loop == 0:  # Loop 1
            loop_config = {
                'loop': 1,
                'train_start': '2018-12-01',
                'train_end': '2020-11-30',
                'val_start': '2020-12-01',
                'val_end': '2020-12-31',
                'test_start': '2021-01-01',
                'test_end': '2021-01-31'
            }
        elif loop == 1:  # Loop 2
            loop_config = {
                'loop': 2,
                'train_start': '2019-01-01',
                'train_end': '2020-12-31',
                'val_start': '2021-01-01',
                'val_end': '2021-01-31',
                'test_start': '2021-02-01',
                'test_end': '2021-02-28'
            }
        elif loop == 2:  # Loop 3
            loop_config = {
                'loop': 3,
                'train_start': '2019-02-01',
                'train_end': '2021-01-31',
                'val_start': '2021-02-01',
                'val_end': '2021-02-28',
                'test_start': '2021-03-01',
                'test_end': '2021-03-31'
            }
        elif loop == 3:  # Loop 4
            loop_config = {
                'loop': 4,
                'train_start': '2019-03-01',
                'train_end': '2021-02-28',
                'val_start': '2021-03-01',
                'val_end': '2021-03-31',
                'test_start': '2021-04-01',
                'test_end': '2021-04-30'
            }
        elif loop == 4:  # Loop 5
            loop_config = {
                'loop': 5,
                'train_start': '2019-04-01',
                'train_end': '2021-03-31',
                'val_start': '2021-04-01',
                'val_end': '2021-04-30',
                'test_start': '2021-05-01',
                'test_end': '2021-05-31'
            }
        elif loop == 5:  # Loop 6
            loop_config = {
                'loop': 6,
                'train_start': '2019-05-01',
                'train_end': '2021-04-30',
                'val_start': '2021-05-01',
                'val_end': '2021-05-31',
                'test_start': '2021-06-01',
                'test_end': '2021-06-30'
            }
        elif loop == 6:  # Loop 7
            loop_config = {
                'loop': 7,
                'train_start': '2019-06-01',
                'train_end': '2021-05-31',
                'val_start': '2021-06-01',
                'val_end': '2021-06-30',
                'test_start': '2021-07-01',
                'test_end': '2021-07-31'
            }
        elif loop == 7:  # Loop 8
            loop_config = {
                'loop': 8,
                'train_start': '2019-07-01',
                'train_end': '2021-06-30',
                'val_start': '2021-07-01',
                'val_end': '2021-07-31',
                'test_start': '2021-08-01',
                'test_end': '2021-08-31'
            }
        elif loop == 8:  # Loop 9
            loop_config = {
                'loop': 9,
                'train_start': '2019-08-01',
                'train_end': '2021-07-31',
                'val_start': '2021-08-01',
                'val_end': '2021-08-31',
                'test_start': '2021-09-01',
                'test_end': '2021-09-30'
            }
        elif loop == 9:  # Loop 10
            loop_config = {
                'loop': 10,
                'train_start': '2019-09-01',
                'train_end': '2021-08-31',
                'val_start': '2021-09-01',
                'val_end': '2021-09-30',
                'test_start': '2021-10-01',
                'test_end': '2021-10-31'
            }
        elif loop == 10:  # Loop 11
            loop_config = {
                'loop': 11,
                'train_start': '2019-10-01',
                'train_end': '2021-09-30',
                'val_start': '2021-10-01',
                'val_end': '2021-10-31',
                'test_start': '2021-11-01',
                'test_end': '2021-11-30'
            }
        elif loop == 11:  # Loop 12
            loop_config = {
                'loop': 12,
                'train_start': '2019-11-01',
                'train_end': '2021-10-31',
                'val_start': '2021-11-01',
                'val_end': '2021-11-30',
                'test_start': '2021-12-01',
                'test_end': '2021-12-31'
            }
        
        date_ranges.append(loop_config)
    
    return date_ranges

def create_config_file(loop_config):
    """Create config.py with specific date ranges"""
    config_template = f'''"""
Configuration for Multi-Currency CNN-LSTM Forex Prediction
Loop {loop_config['loop']} Configuration
"""

import os
import numpy as np

class Config:
    """Configuration for Loop {loop_config['loop']}"""
    
    def __init__(self, model_type='multi'):
        self.MODEL_TYPE = model_type
        
        # Currency pairs configuration
        if model_type == 'multi':
            self.CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
            self.FEATURES_PER_PAIR = 5
            self.TOTAL_FEATURES = len(self.CURRENCY_PAIRS) * self.FEATURES_PER_PAIR
            self.TARGET_PAIR = 'EURUSD'
        else:
            self.CURRENCY_PAIRS = [model_type]
            self.FEATURES_PER_PAIR = 5
            self.TOTAL_FEATURES = self.FEATURES_PER_PAIR
            self.TARGET_PAIR = model_type
        
        # Model architecture parameters
        self.WINDOW_SIZE = 60
        
        if model_type == 'multi':
            self.CNN_FILTERS_1 = 32      
            self.CNN_FILTERS_2 = 64     
            self.LSTM_UNITS_1 = 64      
            self.LSTM_UNITS_2 = 32
        else:
            self.CNN_FILTERS_1 = 16      
            self.CNN_FILTERS_2 = 32     
            self.LSTM_UNITS_1 = 32      
            self.LSTM_UNITS_2 = 16
        
        self.CNN_KERNEL_SIZE = 3
        self.DENSE_UNITS = 32
        self.DROPOUT_RATE = 0.25
        
        # Training parameters
        self.LEARNING_RATE = 0.00005
        self.BATCH_SIZE = 32         
        self.EPOCHS = 100            
        self.VALIDATION_SPLIT = 0.2
        
        # Callback parameters
        self.EARLY_STOPPING_PATIENCE = 25
        self.REDUCE_LR_PATIENCE = 12
        self.REDUCE_LR_FACTOR = 0.5          
        self.MIN_LR = 1e-8
        
        # Data splits - Loop {loop_config['loop']}
        self.TRAIN_START = '{loop_config['train_start']}'
        self.TRAIN_END = '{loop_config['train_end']}'
        self.VAL_START = '{loop_config['val_start']}'
        self.VAL_END = '{loop_config['val_end']}'
        self.TEST_START = '{loop_config['test_start']}'
        self.TEST_END = '{loop_config['test_end']}'
        
        # Trading strategy thresholds
        self.THRESHOLDS = {{
            'conservative': {{'buy': 0.7, 'sell': 0.3}},
            'moderate': {{'buy': 0.6, 'sell': 0.4}},
            'aggressive': {{'buy': 0.55, 'sell': 0.45}}
        }}
        
        # Risk management
        self.MIN_HOLDING_HOURS = 1
        self.MAX_HOLDING_HOURS = 3
        self.STOP_LOSS_PCT = 2
        
        # Portfolio settings
        self.INITIAL_CAPITAL = 10000
        self.POSITION_SIZE = 0.1
        self.MAX_POSITIONS = 1
        
        # Technical indicators parameters
        self.RSI_PERIOD = 14
        self.RSI_OVERBOUGHT = 70
        self.RSI_OVERSOLD = 30
        
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # Directory paths
        self.DATA_PATH = 'data/'
        self.RESULTS_PATH = f'results/loop_{loop_config['loop']}/{{model_type}}/'
        self.MODELS_PATH = f'models/loop_{loop_config['loop']}/{{model_type}}/'
        self.CHECKPOINTS_PATH = f'checkpoints/loop_{loop_config['loop']}/{{model_type}}/'
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for path in [self.RESULTS_PATH, self.MODELS_PATH, self.CHECKPOINTS_PATH]:
            os.makedirs(path, exist_ok=True)
    
    def get_model_config(self):
        """Return model configuration as dictionary"""
        return {{
            'model_type': self.MODEL_TYPE,
            'window_size': self.WINDOW_SIZE,
            'total_features': self.TOTAL_FEATURES,
            'cnn_filters_1': self.CNN_FILTERS_1,
            'cnn_filters_2': self.CNN_FILTERS_2,
            'cnn_kernel_size': self.CNN_KERNEL_SIZE,
            'lstm_units_1': self.LSTM_UNITS_1,
            'lstm_units_2': self.LSTM_UNITS_2,
            'dense_units': self.DENSE_UNITS,
            'dropout_rate': self.DROPOUT_RATE,
            'learning_rate': self.LEARNING_RATE
        }}
    
    def get_training_config(self):
        """Return training configuration as dictionary"""
        return {{
            'batch_size': self.BATCH_SIZE,
            'epochs': self.EPOCHS,
            'validation_split': self.VALIDATION_SPLIT
        }}
    
    def print_config(self):
        """Print current configuration"""
        print("="*60)
        print(f"FOREX PREDICTION SYSTEM - LOOP {loop_config['loop']} - {{self.MODEL_TYPE.upper()}}")
        print("="*60)
        print(f"Model Type: {{self.MODEL_TYPE}}")
        print(f"Currency Pairs: {{self.CURRENCY_PAIRS}}")
        print(f"Target Pair: {{self.TARGET_PAIR}}")
        print(f"Total Features: {{self.TOTAL_FEATURES}}")
        print(f"Window Size: {{self.WINDOW_SIZE}}")
        print(f"Model Architecture: CNN({{self.CNN_FILTERS_1}}, {{self.CNN_FILTERS_2}}) + LSTM({{self.LSTM_UNITS_1}}, {{self.LSTM_UNITS_2}})")
        print(f"Training Period: {{self.TRAIN_START}} to {{self.TRAIN_END}}")
        print(f"Validation Period: {{self.VAL_START}} to {{self.VAL_END}}")
        print(f"Test Period: {{self.TEST_START}} to {{self.TEST_END}}")
        print(f"Initial Capital: ${{self.INITIAL_CAPITAL:,}}")
        print(f"Batch Size: {{self.BATCH_SIZE}}")
        print(f"Max Epochs: {{self.EPOCHS}}")
        print(f"Learning Rate: {{self.LEARNING_RATE}}")
        print(f"Dropout Rate: {{self.DROPOUT_RATE}}")
        print("Trading Thresholds:")
        for strategy, thresholds in self.THRESHOLDS.items():
            print(f"  {{strategy.title()}}: Buy ≥ {{thresholds['buy']}}, Sell ≤ {{thresholds['sell']}}")
        print("="*60)
'''
    
    return config_template

def save_loop_results(loop_num, results):
    """Save results for specific loop"""
    results_dir = f'experiment_results/loop_{loop_num}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as JSON for easy analysis
    with open(f'{results_dir}/summary.json', 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"✅ Loop {loop_num} results saved to {results_dir}")

def print_experiment_summary():
    """Print summary of all 12 loops"""
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION SUMMARY - 12 MONTHLY LOOPS")
    print("="*80)
    
    date_ranges = generate_date_ranges()
    
    print(f"{'Loop':<6} {'Train Start':<12} {'Train End':<12} {'Val Start':<12} {'Val End':<12} {'Test Start':<12} {'Test End':<12}")
    print("-"*80)
    
    for config in date_ranges:
        print(f"{config['loop']:<6} {config['train_start']:<12} {config['train_end']:<12} "
              f"{config['val_start']:<12} {config['val_end']:<12} "
              f"{config['test_start']:<12} {config['test_end']:<12}")
    
    print("-"*80)
    print("\nTo run a specific loop manually:")
    print("1. Copy the configuration for that loop to config.py")
    print("2. Run: python main_fx.py --model all --test")
    print("\nTo run all loops automatically:")
    print("python run_experiments.py --run-all")

def main():
    """Main function to display experiment configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment Runner for 12 Monthly Loops')
    parser.add_argument('--loop', type=int, choices=range(1, 13),
                       help='Generate config for specific loop')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all loop configurations')
    parser.add_argument('--generate-config', type=int, choices=range(1, 13),
                       help='Generate config.py for specific loop')
    
    args = parser.parse_args()
    
    if args.show_all:
        print_experiment_summary()
    
    elif args.generate_config:
        date_ranges = generate_date_ranges()
        loop_config = date_ranges[args.generate_config - 1]
        config_content = create_config_file(loop_config)
        
        # Save to a temporary file
        temp_filename = f'config_loop_{args.generate_config}.py'
        with open(temp_filename, 'w') as f:
            f.write(config_content)
        
        print(f"\n✅ Configuration for Loop {args.generate_config} saved to {temp_filename}")
        print(f"\nTo use this configuration:")
        print(f"1. Backup your current config.py: cp config.py config_backup.py")
        print(f"2. Replace with new config: cp {temp_filename} config.py")
        print(f"3. Run training: python main_fx.py --model all --test")
        print(f"4. Results will be saved in: results/loop_{args.generate_config}/")
    
    else:
        print_experiment_summary()
        print("\nFor specific loop config: python run_experiments.py --generate-config <loop_number>")

if __name__ == "__main__":
    main()