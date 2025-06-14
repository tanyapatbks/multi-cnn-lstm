"""
Data Processor for Multi-Currency CNN-LSTM Forex Prediction
Enhanced version with Single Currency support
Technical indicators are calculated but NOT used as model input
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Enhanced data processor for multi and single currency forex data"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        
    def load_currency_data(self, verbose=True):
        """Load OHLCV data for configured currency pairs"""
        if verbose:
            print(f"📊 Loading currency data for {self.config.MODEL_TYPE} model...")
        
        data = {}
        for pair in self.config.CURRENCY_PAIRS:
            try:
                file_path = f"{self.config.DATA_PATH}{pair}_1H.csv"
                df = pd.read_csv(file_path)
                
                if verbose:
                    print(f"   📈 {pair}: {df.shape[0]} records loaded")
                
                # Handle datetime parsing with multiple strategies
                df['Local time'] = self._parse_datetime(df['Local time'])
                
                # Set index and sort
                df.set_index('Local time', inplace=True)
                df.sort_index(inplace=True)
                
                data[pair] = df
                
                if verbose:
                    print(f"   📅 {pair}: {df.index.min()} to {df.index.max()}")
                
            except Exception as e:
                print(f"   ❌ Error loading {pair}: {str(e)}")
                return None
        
        return data
    
    def _parse_datetime(self, datetime_series):
        """Parse datetime with multiple fallback strategies"""
        try:
            # Strategy 1: Direct conversion
            return pd.to_datetime(datetime_series, infer_datetime_format=True)
        except:
            try:
                # Strategy 2: Remove timezone and milliseconds
                cleaned = datetime_series.astype(str)
                cleaned = cleaned.str.replace(r' GMT[+-]\d{4}', '', regex=True)
                cleaned = cleaned.str.replace(r'\.000', '', regex=True)
                return pd.to_datetime(cleaned, format='%d.%m.%Y %H:%M:%S')
            except:
                try:
                    # Strategy 3: Day first parsing
                    cleaned = datetime_series.astype(str)
                    cleaned = cleaned.str.replace(r' GMT[+-]\d{4}', '', regex=True)
                    cleaned = cleaned.str.replace(r'\.000', '', regex=True)
                    return pd.to_datetime(cleaned, dayfirst=True)
                except Exception as e:
                    raise ValueError(f"All datetime parsing strategies failed: {str(e)}")
    
    def preprocess_data(self, data, verbose=True):
        """
        Preprocess data: handle missing values, calculate returns
        IMPORTANT: Technical indicators are calculated but NOT included in model features
        """
        if verbose:
            print("🔧 Preprocessing data...")
        
        processed_data = {}
        
        for pair, df in data.items():
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # IMPORTANT: Only OHLCV will be used for model training
            # Calculate percentage returns for OHLC (making data stationary)
            for col in ['Open', 'High', 'Low', 'Close']:
                df[f'{col}_Return'] = df[col].pct_change().fillna(0)
                df[f'{col}_Price'] = df[col]  # Keep original prices for trading
            
            # Keep original volume for normalization
            df['Volume_Original'] = df['Volume']
            
            # Calculate technical indicators ONLY for trading simulation
            # These will NOT be included in model features
            df = self._calculate_technical_indicators_for_trading(df)
            
            processed_data[pair] = df
            
            if verbose:
                print(f"   ✅ {pair}: Preprocessing completed (OHLCV only for model)")
        
        return processed_data
    
    def _calculate_technical_indicators_for_trading(self, df):
        """
        Calculate RSI and MACD indicators for trading simulation ONLY
        These are NOT used as model input features
        """
        # RSI Calculation
        close_delta = df['Close'].diff()
        gain = close_delta.where(close_delta > 0, 0)
        loss = -close_delta.where(close_delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.config.RSI_PERIOD).mean()
        avg_loss = loss.rolling(window=self.config.RSI_PERIOD).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'].fillna(50, inplace=True)  # Neutral RSI for initial values
        
        # MACD Calculation
        exp1 = df['Close'].ewm(span=self.config.MACD_FAST, adjust=False).mean()
        exp2 = df['Close'].ewm(span=self.config.MACD_SLOW, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=self.config.MACD_SIGNAL, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Fill any NaN values
        df['MACD'].fillna(0, inplace=True)
        df['MACD_Signal'].fillna(0, inplace=True)
        df['MACD_Histogram'].fillna(0, inplace=True)
        
        return df
    
    def create_unified_dataset(self, processed_data, verbose=True):
        """
        Create unified dataset - ONLY OHLCV features for model input
        Multi-currency: 15 features (OHLCV * 3 pairs)
        Single currency: 5 features (OHLCV * 1 pair)
        """
        if verbose:
            if self.config.MODEL_TYPE == 'multi':
                print("🔗 Creating unified multi-currency dataset (OHLCV only)...")
            else:
                print(f"🔗 Creating single-currency dataset for {self.config.MODEL_TYPE} (OHLCV only)...")
        
        # Find common timestamps across all pairs
        common_index = None
        for pair, df in processed_data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        if verbose:
            print(f"   📅 Common timestamps: {len(common_index)}")
        
        # Create unified feature matrix with FIXED ORDER (prevents data leakage)
        unified_features = []
        feature_columns = []
        
        for pair in self.config.CURRENCY_PAIRS:  # Use config order
            df = processed_data[pair].loc[common_index]
            
            # IMPORTANT: Only select OHLCV features for model input
            # Technical indicators are NOT included
            pair_features = ['Open_Return', 'High_Return', 'Low_Return', 'Close_Return', 'Volume_Original']
            pair_data = df[pair_features]
            
            # Rename columns with pair prefix (only for multi-currency)
            if self.config.MODEL_TYPE == 'multi':
                pair_data.columns = [f'{pair}_{col}' for col in pair_data.columns]
            else:
                # For single currency, use simple column names
                pair_data.columns = pair_features
            
            unified_features.append(pair_data)
            feature_columns.extend(pair_data.columns)
        
        # Concatenate all features
        unified_df = pd.concat(unified_features, axis=1)
        
        # Normalize features
        unified_df = self._normalize_features(unified_df)
        
        if verbose:
            print(f"   ✅ Unified dataset: {unified_df.shape[0]} samples × {unified_df.shape[1]} features")
            print(f"   📊 Expected shape: (samples, {self.config.TOTAL_FEATURES})")
            print(f"   ⚠️  Technical indicators calculated but NOT included in model features")
        
        return unified_df, feature_columns
    
    def _normalize_features(self, df):
        """
        Normalize features using appropriate scalers
        OHLC Returns: StandardScaler (zero mean, unit variance)
        Volume: MinMaxScaler (0-1 range)
        """
        normalized_df = df.copy()
        
        for col in df.columns:
            if 'Return' in col:
                # Use StandardScaler for returns (zero mean, unit variance)
                scaler = StandardScaler()
                normalized_df[col] = scaler.fit_transform(df[[col]]).flatten()
                self.scalers[col] = scaler
            elif 'Volume' in col:
                # Use MinMaxScaler for volume (0-1 range)
                scaler = MinMaxScaler()
                normalized_df[col] = scaler.fit_transform(df[[col]]).flatten()
                self.scalers[col] = scaler
        
        return normalized_df
    
    def get_price_data(self, processed_data, timestamps, target_pair=None):
        """Extract price data for trading strategy evaluation"""
        if target_pair is None:
            target_pair = self.config.TARGET_PAIR
        
        target_data = processed_data[target_pair].loc[timestamps]
        return target_data
    
    def get_technical_indicators(self, processed_data, timestamps, target_pair=None):
        """
        Extract technical indicators for baseline trading strategies
        These are ONLY used for trading simulation, NOT for model training
        """
        if target_pair is None:
            target_pair = self.config.TARGET_PAIR
        
        target_data = processed_data[target_pair].loc[timestamps]
        return {
            'RSI': target_data['RSI'],
            'MACD': target_data['MACD'],
            'MACD_Signal': target_data['MACD_Signal'],
            'MACD_Histogram': target_data['MACD_Histogram']
        }
    
    def save_scalers(self, filepath):
        """Save scalers for future use"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"✅ Scalers saved to {filepath}")
    
    def load_scalers(self, filepath):
        """Load saved scalers"""
        import pickle
        with open(filepath, 'rb') as f:
            self.scalers = pickle.load(f)
        print(f"✅ Scalers loaded from {filepath}")

class SequencePreparator:
    """Create sequences for CNN-LSTM training - works for both multi and single currency"""
    
    def __init__(self, config):
        self.config = config
    
    def create_sequences(self, unified_data, target_pair=None, verbose=True):
        """
        Create sliding window sequences and labels
        Input shape:
        - Multi-currency: (batch, 60, 15) - OHLCV * 3 pairs
        - Single currency: (batch, 60, 5) - OHLCV * 1 pair
        """
        if target_pair is None:
            target_pair = self.config.TARGET_PAIR
            
        if verbose:
            print(f"📋 Creating sequences for {target_pair} prediction...")
            print(f"   📊 Data shape: {unified_data.shape}")
            print(f"   📅 Period: {unified_data.index.min()} to {unified_data.index.max()}")
            print(f"   ⚠️  Using OHLCV features only (no technical indicators)")
        
        # Get feature matrix and target column
        feature_matrix = unified_data.values
        
        # Determine target column based on model type
        if self.config.MODEL_TYPE == 'multi':
            target_column = f'{target_pair}_Close_Return'
        else:
            target_column = 'Close_Return'
        
        if target_column not in unified_data.columns:
            raise ValueError(f"Target column {target_column} not found")
        
        target_returns = unified_data[target_column].values
        
        # Calculate number of sequences
        num_sequences = len(unified_data) - self.config.WINDOW_SIZE
        
        if num_sequences <= 0:
            raise ValueError(f"Insufficient data: need at least {self.config.WINDOW_SIZE + 1} records")
        
        if verbose:
            print(f"   📐 Window size: {self.config.WINDOW_SIZE}")
            print(f"   📊 Sequences to create: {num_sequences}")
        
        # Initialize arrays
        num_features = unified_data.shape[1]
        X = np.zeros((num_sequences, self.config.WINDOW_SIZE, num_features), dtype=np.float32)
        y = np.zeros(num_sequences, dtype=np.float32)
        timestamps = []
        
        # Create sequences
        for i in range(num_sequences):
            # Feature sequence (lookback window)
            X[i] = feature_matrix[i:i + self.config.WINDOW_SIZE]
            
            # Target label (direction: 1 if positive return, 0 if negative)
            future_return = target_returns[i + self.config.WINDOW_SIZE]
            y[i] = 1.0 if future_return > 0 else 0.0
            
            # Store timestamp of prediction point
            timestamps.append(unified_data.index[i + self.config.WINDOW_SIZE])
        
        timestamps = pd.DatetimeIndex(timestamps)
        
        if verbose:
            print(f"   📊 Final shapes: X{X.shape}, y{y.shape}")
            print(f"   📊 Class balance: {y.mean():.3f} (1=up, 0=down)")
            print(f"   📅 Sequence period: {timestamps.min()} to {timestamps.max()}")
        
        return X, y, timestamps
    
    def split_temporal_data(self, X, y, timestamps, verbose=True):
        """Split data temporally to prevent data leakage"""
        if verbose:
            print("📅 Splitting data temporally (NO DATA LEAKAGE)...")
        
        # Use config date ranges
        train_start = pd.to_datetime(self.config.TRAIN_START)
        train_end = pd.to_datetime(self.config.TRAIN_END)
        val_start = pd.to_datetime(self.config.VAL_START)
        val_end = pd.to_datetime(self.config.VAL_END)
        test_start = pd.to_datetime(self.config.TEST_START)
        test_end = pd.to_datetime(self.config.TEST_END)
        
        # Create masks
        train_mask = (timestamps >= train_start) & (timestamps <= train_end)
        val_mask = (timestamps >= val_start) & (timestamps <= val_end)
        test_mask = (timestamps >= test_start) & (timestamps <= test_end)
        
        splits = {
            'train': (X[train_mask], y[train_mask], timestamps[train_mask]),
            'val': (X[val_mask], y[val_mask], timestamps[val_mask]),
            'test': (X[test_mask], y[test_mask], timestamps[test_mask])
        }
        
        if verbose:
            for split_name, (X_split, y_split, ts_split) in splits.items():
                print(f"   📊 {split_name.upper()}: {len(y_split)} samples, "
                      f"balance: {y_split.mean():.3f}")
                if len(ts_split) > 0:
                    print(f"      📅 Period: {ts_split.min().date()} to {ts_split.max().date()}")
        
        return splits