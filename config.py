import os
import pandas as pd

class Config:
    def __init__(self, model_type='multi', target_pair='EURUSD'):
        self.MODEL_TYPE = model_type
        self.TARGET_PAIR = target_pair
        self.ALL_CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
        self.FEATURES_PER_PAIR = 5  # OHLCV only
        
        if self.MODEL_TYPE == 'multi':
            self.INPUT_CURRENCY_PAIRS = self.ALL_CURRENCY_PAIRS
            self.TOTAL_FEATURES = len(self.INPUT_CURRENCY_PAIRS) * self.FEATURES_PER_PAIR
        else:
            self.INPUT_CURRENCY_PAIRS = [self.MODEL_TYPE]
            self.TOTAL_FEATURES = self.FEATURES_PER_PAIR

        # Model Architecture Parameters
        self.WINDOW_SIZE = 60
        if self.MODEL_TYPE == 'multi':
            self.CNN_FILTERS_1, self.CNN_FILTERS_2 = 64, 128
            self.LSTM_UNITS_1, self.LSTM_UNITS_2 = 128, 64
        else:
            self.CNN_FILTERS_1, self.CNN_FILTERS_2 = 64, 128
            self.LSTM_UNITS_1, self.LSTM_UNITS_2 = 128, 64
        
        self.CNN_KERNEL_SIZE = 3
        self.DENSE_UNITS = 64
        self.DROPOUT_RATE = 0.4
        
        # Training Parameters
        self.LEARNING_RATE = 0.00001
        self.BATCH_SIZE = 32
        self.EPOCHS = 5
        self.EARLY_STOPPING_PATIENCE = 20
        self.REDUCE_LR_PATIENCE = 5
        
        # Data Split Periods (Default - will be overridden by Rolling Window)
        self.TRAIN_START = '2019-01-01'
        self.TRAIN_END = '2020-12-31'
        self.VAL_START = '2021-01-01'
        self.VAL_END = '2021-01-31'
        self.TEST_START = '2021-02-01'
        self.TEST_END = '2021-02-28'
        
        # Volume Processing Parameters (Updated to 7SD as per specification)
        self.VOLUME_SD_MULTIPLIER = 7  # Use 7 Standard Deviations instead of 3
        self.VOLUME_OUTLIER_METHOD = 'cap'  # 'cap' or 'remove'
        
        # Technical Indicators Parameters (for baseline strategies only)
        self.RSI_PERIOD = 14
        self.RSI_OVERSOLD = 30
        self.RSI_OVERBOUGHT = 70
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # Trading Strategy Thresholds
        self.THRESHOLDS = {
            'Conservative': {'buy': 0.7, 'sell': 0.3},
            'Moderate': {'buy': 0.6, 'sell': 0.4},
            'Aggressive': {'buy': 0.55, 'sell': 0.45}
        }
        
        # Trading Constraints
        self.MIN_HOLD_HOURS = 1
        self.MAX_HOLD_HOURS = 3
        self.STOP_LOSS_PCT = 2.0
        self.TAKE_PROFIT_AFTER_HOURS = 1
        
        # Leverage Configuration
        self.LEVERAGE_SETTINGS = {
            'Conservative': 2.0,  # High confidence → High leverage
            'Moderate': 1.0,      # Standard leverage
            'Aggressive': 0.5     # Low confidence → Low leverage
        }
        
        # Portfolio and Trading Parameters
        self.INITIAL_CAPITAL = 10000
        self.BASE_LOT_SIZE = 0.1
        self.PIP_VALUES = {
            'EURUSD': 10.0,
            'GBPUSD': 10.0,
            'USDJPY': 8.33  # Approximate for JPY pairs
        }
        
        # Monthly Analysis Periods (for detailed reporting)
        self.MONTHLY_PERIODS = [
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
        
        # Results and Logging
        self.RESULTS_PATH = 'results/'
        self.SAVE_MODELS = True
        self.VERBOSE_TRAINING = 1
        
        # Data Quality Parameters
        self.MIN_DATA_POINTS = 1000  # Minimum data points required for training
        self.MAX_MISSING_DATA_PCT = 5  # Maximum percentage of missing data allowed
        
        # File Paths
        self.DATA_PATH = 'data/'  # Updated to read from data folder
        self.MODEL_SAVE_PATH = 'models/'
        self.PLOTS_SAVE_PATH = 'plots/'
        
        # Data Format Settings
        self.DATETIME_FORMAT = '%d.%m.%Y %H:%M:%S.%f GMT%z'  # Your CSV format
        self.DATETIME_FORMATS = [
            '%d.%m.%Y %H:%M:%S.%f GMT%z',
            '%d.%m.%Y %H:%M:%S.%f %Z%z',
            '%d.%m.%Y %H:%M:%S GMT%z'
        ]
        
        # Create directories if they don't exist
        for path in [self.RESULTS_PATH, self.MODEL_SAVE_PATH, self.PLOTS_SAVE_PATH]:
            os.makedirs(path, exist_ok=True)
    
    def _create_directories(self):
        """Create necessary directories for results, models, and plots"""
        directories = [
            self.RESULTS_PATH,
            self.MODEL_SAVE_PATH, 
            self.PLOTS_SAVE_PATH,
            os.path.join(self.RESULTS_PATH, 'models'),
            os.path.join(self.RESULTS_PATH, 'plots'),
            os.path.join(self.RESULTS_PATH, 'logs')
        ]
        
        # Support for legacy MODELS_PATH attribute
        if hasattr(self, 'MODELS_PATH'):
            directories.append(self.MODELS_PATH)
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        return True
    
    def setup_experiment_paths(self, experiment_name=None):
        """Setup paths for a specific experiment"""
        if experiment_name:
            self.RESULTS_PATH = os.path.join('results', experiment_name)
            self.MODEL_SAVE_PATH = os.path.join(self.RESULTS_PATH, 'models')
            self.PLOTS_SAVE_PATH = os.path.join(self.RESULTS_PATH, 'plots')
        
        self._create_directories()
        return self.RESULTS_PATH
    
    def get_model_save_path(self, model_name):
        """Get full path for saving a model"""
        return os.path.join(self.MODEL_SAVE_PATH, f"{model_name}.h5")
    
    def get_results_save_path(self, filename):
        """Get full path for saving results"""
        return os.path.join(self.RESULTS_PATH, filename)
    
    @property
    def MODELS_PATH(self):
        """Legacy property for backward compatibility"""
        return self.MODEL_SAVE_PATH
    
    @MODELS_PATH.setter
    def MODELS_PATH(self, value):
        """Legacy setter for backward compatibility"""
        self.MODEL_SAVE_PATH = value
    
    def print_config_summary(self):
        """Print a summary of current configuration"""
        print("="*80)
        print("⚙️ CONFIGURATION SUMMARY")
        print("="*80)
        print(f"📊 Model Type: {self.MODEL_TYPE}")
        print(f"🎯 Target Pair: {self.TARGET_PAIR}")
        print(f"💱 Currency Pairs: {self.ALL_CURRENCY_PAIRS}")
        print(f"🔢 Total Features: {self.TOTAL_FEATURES}")
        print(f"📏 Window Size: {self.WINDOW_SIZE}")
        print(f"🧠 Architecture: CNN({self.CNN_FILTERS_1},{self.CNN_FILTERS_2}) + LSTM({self.LSTM_UNITS_1},{self.LSTM_UNITS_2})")
        print(f"📈 Volume Processing: {self.VOLUME_SD_MULTIPLIER}SD capping + Min-Max scaling")
        print(f"🎛️ Thresholds: {self.THRESHOLDS}")
        print(f"💰 Leverage: {self.LEVERAGE_SETTINGS}")
        print(f"⏱️ Training Periods: {self.TRAIN_START} to {self.TRAIN_END}")
        print(f"✅ Validation: {self.VAL_START} to {self.VAL_END}")
        print(f"🧪 Test: {self.TEST_START} to {self.TEST_END}")
        print("="*80)
    
    def get_threshold_by_name(self, threshold_name):
        """Get threshold values by name"""
        return self.THRESHOLDS.get(threshold_name, self.THRESHOLDS['Moderate'])
    
    def get_leverage_by_threshold(self, threshold_name):
        """Get leverage setting by threshold name"""
        return self.LEVERAGE_SETTINGS.get(threshold_name, 1.0)
    
    def update_periods(self, train_start, train_end, val_start, val_end, test_start=None, test_end=None):
        """Update training/validation/test periods (used by Rolling Window)"""
        self.TRAIN_START = train_start
        self.TRAIN_END = train_end
        self.VAL_START = val_start
        self.VAL_END = val_end
        if test_start and test_end:
            self.TEST_START = test_start
            self.TEST_END = test_end
    
    def validate_config(self):
        """Validate configuration parameters"""
        issues = []
        
        # Check if all required currency pairs are valid
        valid_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        for pair in self.ALL_CURRENCY_PAIRS:
            if pair not in valid_pairs:
                issues.append(f"Invalid currency pair: {pair}")
        
        # Check threshold values
        for name, thresholds in self.THRESHOLDS.items():
            if not (0 < thresholds['sell'] < thresholds['buy'] < 1):
                issues.append(f"Invalid thresholds for {name}: sell={thresholds['sell']}, buy={thresholds['buy']}")
        
        # Check leverage settings
        for name, leverage in self.LEVERAGE_SETTINGS.items():
            if leverage <= 0:
                issues.append(f"Invalid leverage for {name}: {leverage}")
        
        # Check time periods
        if pd.to_datetime(self.TRAIN_START) >= pd.to_datetime(self.TRAIN_END):
            issues.append("Train start date must be before train end date")
        
        if issues:
            print("⚠️ Configuration Issues Found:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ Configuration validation passed")
            return True