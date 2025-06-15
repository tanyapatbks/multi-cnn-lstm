"""
Configuration for Multi-Currency CNN-LSTM Forex Prediction
Enhanced version with Single Currency support
"""

import os
import numpy as np

class Config:
    """Enhanced configuration for multi and single currency forex prediction"""

    def __init__(self, model_type='multi'):
        """
        Initialize configuration

        Args:
            model_type: 'multi' for multi-currency, or 'EURUSD'/'GBPUSD'/'USDJPY' for single
        """
        self.MODEL_TYPE = model_type

        # Currency pairs configuration
        if model_type == 'multi':
            self.CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
            self.FEATURES_PER_PAIR = 5  # OHLCV
            self.TOTAL_FEATURES = len(self.CURRENCY_PAIRS) * self.FEATURES_PER_PAIR  # 15
            self.TARGET_PAIR = 'EURUSD'  # Default target for multi-currency
        else:
            # Single currency configuration
            self.CURRENCY_PAIRS = [model_type]  # e.g., ['EURUSD']
            self.FEATURES_PER_PAIR = 5  # OHLCV
            self.TOTAL_FEATURES = self.FEATURES_PER_PAIR  # 5 for single currency
            self.TARGET_PAIR = model_type

        # Model architecture parameters (adjust for single vs multi)
        self.WINDOW_SIZE = 60  # Keep same for both

        if model_type == 'multi':
            # ==================================================================
            # <<< แก้ไขส่วนนี้ให้ตรงกับ Thesis Proposal (Page 11-12) >>>
            # ==================================================================
            self.CNN_FILTERS_1 = 64      # เดิม: 32
            self.CNN_FILTERS_2 = 128     # เดิม: 64
            self.LSTM_UNITS_1 = 128      # เดิม: 64
            self.LSTM_UNITS_2 = 64       # เดิม: 32
        else:
            # Smaller architecture for single currency (less complex data)
            self.CNN_FILTERS_1 = 32
            self.CNN_FILTERS_2 = 64
            self.LSTM_UNITS_1 = 64
            self.LSTM_UNITS_2 = 32

        self.CNN_KERNEL_SIZE = 3
        self.DENSE_UNITS = 64
        self.DROPOUT_RATE = 0.2

        # Training parameters
        self.LEARNING_RATE = 0.00005
        self.BATCH_SIZE = 32
        self.EPOCHS = 100
        self.VALIDATION_SPLIT = 0.2

        # Callback parameters
        self.EARLY_STOPPING_PATIENCE = 30
        self.REDUCE_LR_PATIENCE = 15
        self.REDUCE_LR_FACTOR = 0.5
        self.MIN_LR = 1e-8

        # Data splits - Manual configuration
        # หมายเหตุ: ค่าเหล่านี้เป็นแบบ static และจะถูกใช้เมื่อรัน main_fx.py โดยตรง
        # เพื่อให้สอดคล้องกับงานวิจัย ควรใช้ run_experiments.py เพื่อสร้าง date ranges แบบ rolling window
        self.TRAIN_START = '2018-12-01'
        self.TRAIN_END = '2020-11-30'
        self.VAL_START = '2020-12-01'
        self.VAL_END = '2020-12-31'
        self.TEST_START = '2021-01-01'
        self.TEST_END = '2021-01-31'

        # Trading strategy thresholds
        self.THRESHOLDS = {
            'conservative': {'buy': 0.7, 'sell': 0.3},
            'moderate': {'buy': 0.6, 'sell': 0.4},
            'aggressive': {'buy': 0.55, 'sell': 0.45}
        }

        # Position sizing based on confidence (in lots)
        self.LOT_SIZES = {
            'conservative': 1.0,   # High confidence = 1.0 lot
            'moderate': 0.5,       # Medium confidence = 0.5 lot
            'aggressive': 0.1      # Low confidence = 0.1 lot (smallest size)
        }

        # Risk management
        self.MIN_HOLDING_HOURS = 1
        self.MAX_HOLDING_HOURS = 3
        self.STOP_LOSS_PIPS = 20      # Stop loss in pips
        self.TAKE_PROFIT_PIPS = 40    # Take profit in pips (2:1 RR)

        # Portfolio settings
        self.INITIAL_CAPITAL = 10000   # $10,000 initial capital
        self.LEVERAGE = 100            # 1:100 leverage
        self.RISK_PER_TRADE_PCT = 0.02 # Risk 2% per trade
        self.MAX_POSITIONS = 1         # Maximum concurrent positions

        # Pip values for different pairs (USD per pip per standard lot)
        self.PIP_VALUES = {
            'EURUSD': 10.0,
            'GBPUSD': 10.0,
            'USDJPY': 9.09  # Approximate for JPY pairs
        }

        # Technical indicators parameters
        self.RSI_PERIOD = 14
        self.RSI_OVERBOUGHT = 70
        self.RSI_OVERSOLD = 30

        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9

        # Directory paths
        self.DATA_PATH = 'data/'
        self.RESULTS_PATH = f'results/{model_type}/'
        self.MODELS_PATH = f'models/{model_type}/'
        self.CHECKPOINTS_PATH = f'checkpoints/{model_type}/'

        # Create directories if they don't exist
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories"""
        for path in [self.RESULTS_PATH, self.MODELS_PATH, self.CHECKPOINTS_PATH]:
            os.makedirs(path, exist_ok=True)

    def get_model_config(self):
        """Return model configuration as dictionary"""
        return {
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
        }

    def get_training_config(self):
        """Return training configuration as dictionary"""
        return {
            'batch_size': self.BATCH_SIZE,
            'epochs': self.EPOCHS,
            'validation_split': self.VALIDATION_SPLIT
        }

    def print_config(self):
        """Print current configuration"""
        print("="*60)
        print(f"FOREX PREDICTION SYSTEM CONFIGURATION - {self.MODEL_TYPE.upper()}")
        print("="*60)
        print(f"Model Type: {self.MODEL_TYPE}")
        print(f"Currency Pairs: {self.CURRENCY_PAIRS}")
        print(f"Target Pair: {self.TARGET_PAIR}")
        print(f"Total Features: {self.TOTAL_FEATURES}")
        print(f"Window Size: {self.WINDOW_SIZE}")
        print(f"Model Architecture: CNN({self.CNN_FILTERS_1}, {self.CNN_FILTERS_2}) + LSTM({self.LSTM_UNITS_1}, {self.LSTM_UNITS_2})")
        print(f"Training Period: {self.TRAIN_START} to {self.TRAIN_END}")
        print(f"Validation Period: {self.VAL_START} to {self.VAL_END}")
        print(f"Test Period: {self.TEST_START} to {self.TEST_END}")
        print(f"Initial Capital: ${self.INITIAL_CAPITAL:,}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Max Epochs: {self.EPOCHS}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Dropout Rate: {self.DROPOUT_RATE}")
        print("Trading Thresholds:")
        for strategy, thresholds in self.THRESHOLDS.items():
            print(f"  {strategy.title()}: Buy ≥ {thresholds['buy']}, Sell ≤ {thresholds['sell']}")
        print("="*60)