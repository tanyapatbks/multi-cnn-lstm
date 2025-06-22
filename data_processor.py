"""
Enhanced Data Processor with proper Volume processing (7SD) and separated Technical Indicators
Updated to prevent data leakage and ensure Train Set statistics only
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def load_currency_data(self):
        """Load currency data from CSV files."""
        print("📥 Loading currency data...")
        data = {}
        
        for pair in self.config.ALL_CURRENCY_PAIRS:
            try:
                # Load from data/ folder
                filename = f"data/{pair}_1H.csv"
                df = pd.read_csv(filename)
                
                # Convert timestamp with proper format handling
                # Format: "13.01.2018 00:00:00.000 GMT+0700"
                try:
                    df['Local time'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
                except ValueError:
                    # Fallback: try different formats
                    try:
                        df['Local time'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f %Z%z')
                    except ValueError:
                        # Final fallback: let pandas infer
                        print(f"   ⚠️ Using automatic date parsing for {pair}")
                        df['Local time'] = pd.to_datetime(df['Local time'], infer_datetime_format=True)
                
                # Convert to UTC and set as index
                if df['Local time'].dt.tz is not None:
                    df['Local time'] = df['Local time'].dt.tz_convert('UTC')
                else:
                    df['Local time'] = df['Local time'].dt.tz_localize('UTC')
                    
                df.set_index('Local time', inplace=True)
                
                # Sort by timestamp
                df.sort_index(inplace=True)
                
                # Verify required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    print(f"❌ Missing required columns in {filename}")
                    print(f"   Available columns: {list(df.columns)}")
                    print(f"   Required columns: {required_cols}")
                    return None
                
                # Check data quality
                total_rows = len(df)
                if total_rows < self.config.MIN_DATA_POINTS:
                    print(f"⚠️ {pair}: Only {total_rows} rows (minimum: {self.config.MIN_DATA_POINTS})")
                
                # Check for missing values
                missing_pct = (df[required_cols].isnull().sum().sum() / (len(df) * len(required_cols))) * 100
                if missing_pct > self.config.MAX_MISSING_DATA_PCT:
                    print(f"⚠️ {pair}: {missing_pct:.1f}% missing data (max allowed: {self.config.MAX_MISSING_DATA_PCT}%)")
                
                data[pair] = df
                print(f"   ✅ {pair}: {len(df)} records loaded ({df.index[0]} to {df.index[-1]})")
                
            except FileNotFoundError:
                print(f"❌ File not found: {filename}")
                print(f"   Please ensure CSV files are in the 'data/' folder")
                return None
            except pd.errors.EmptyDataError:
                print(f"❌ Empty CSV file: {filename}")
                return None
            except Exception as e:
                print(f"❌ Error loading {pair}: {e}")
                print(f"   File: {filename}")
                # Print first few lines for debugging
                try:
                    with open(filename, 'r') as f:
                        print(f"   First 3 lines of {filename}:")
                        for i, line in enumerate(f):
                            if i < 3:
                                print(f"     {line.strip()}")
                            else:
                                break
                except:
                    pass
                return None
        
        print(f"✅ Successfully loaded {len(data)} currency pairs")
        
        # Validate data consistency
        self._validate_data_consistency(data)
        
        return data
    
    def _validate_data_consistency(self, data):
        """Validate loaded data for consistency and quality"""
        print("🔍 Validating data consistency...")
        
        if not data:
            print("❌ No data loaded")
            return False
        
        # Check date ranges
        date_ranges = {}
        for pair, df in data.items():
            date_ranges[pair] = {
                'start': df.index.min(),
                'end': df.index.max(),
                'count': len(df)
            }
            print(f"   📅 {pair}: {date_ranges[pair]['start']} to {date_ranges[pair]['end']} ({date_ranges[pair]['count']} records)")
        
        # Check for sufficient overlap
        all_starts = [info['start'] for info in date_ranges.values()]
        all_ends = [info['end'] for info in date_ranges.values()]
        
        common_start = max(all_starts)
        common_end = min(all_ends)
        
        if common_start >= common_end:
            print("⚠️ Warning: No overlapping time period between all currency pairs")
        else:
            print(f"   ✅ Common time period: {common_start} to {common_end}")
        
        # Check for data gaps
        for pair, df in data.items():
            # Check for large gaps (more than 2 hours)
            time_diffs = df.index.to_series().diff()
            large_gaps = time_diffs > pd.Timedelta(hours=2)
            if large_gaps.any():
                gap_count = large_gaps.sum()
                print(f"   ⚠️ {pair}: {gap_count} time gaps > 2 hours detected")
        
        return True
    
    def preprocess_data(self, raw_data):
        """
        Preprocess currency data with proper Train Set statistics isolation.
        IMPORTANT: Technical Indicators (RSI, MACD) are calculated but NOT included in model input.
        """
        print("🔄 Preprocessing currency data...")
        processed_data = {}
        
        for pair, df in raw_data.items():
            print(f"   Processing {pair}...")
            df_processed = df.copy()
            
            # Step 1: Convert OHLC to percentage change (renamed from _Return to _Changed)
            for col in ['Open', 'High', 'Low', 'Close']:
                df_processed[f'{col}_Changed'] = df_processed[col].pct_change()
            
            # Step 2: Calculate Technical Indicators (for baseline strategies only)
            df_processed = self._calculate_technical_indicators(df_processed)
            
            # Step 3: Remove rows with NaN values
            df_processed.dropna(inplace=True)
            
            processed_data[pair] = df_processed
            print(f"      ✅ {pair}: {len(df_processed)} records after preprocessing")
        
        return processed_data
    
    def _calculate_technical_indicators(self, df):
        """Calculate RSI and MACD for baseline strategies (NOT for model input)"""
        
        # RSI Calculation
        close_delta = df['Close'].diff()
        gain = (close_delta.where(close_delta > 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
        loss = (-close_delta.where(close_delta < 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD Calculation
        exp1 = df['Close'].ewm(span=self.config.MACD_FAST, adjust=False).mean()
        exp2 = df['Close'].ewm(span=self.config.MACD_SLOW, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=self.config.MACD_SIGNAL, adjust=False).mean()
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True, limit=100)
        df.fillna(method='ffill', inplace=True, limit=100)
        
        return df
    
    def get_model_input_data(self, processed_data, loop_info=None):
        """
        Creates the final DataFrame for model input with ONLY OHLCV features.
        Uses Train Set statistics for scaling to prevent data leakage.
        
        Args:
            processed_data: Preprocessed data dictionary
            loop_info: Dictionary containing train/val/test periods (optional for backward compatibility)
        """
        # Backward compatibility: create loop_info from config if not provided
        # Note: This is NOT fake data - it uses the actual config time periods
        # instead of rolling window periods, allowing main_fx.py to work with fixed periods
        if loop_info is None:
            loop_info = {
                'loop': 1,
                'train_start': self.config.TRAIN_START,
                'train_end': self.config.TRAIN_END,
                'val_start': self.config.VAL_START,
                'val_end': self.config.VAL_END
            }
            print("⚠️ Using config periods (main_fx.py compatibility mode - using actual config dates, not rolling window)")
        
        print(f"🎯 Creating model input data for Loop {loop_info.get('loop', 'N/A')}...")
        
        features_list = []
        final_columns = []
        
        # Get train period for statistics calculation
        train_start = pd.to_datetime(loop_info['train_start'], utc=True)
        train_end = pd.to_datetime(loop_info['train_end'], utc=True)
        
        print(f"      📊 Using Train period: {train_start.date()} to {train_end.date()}")
        
        # Collect train data for volume statistics
        train_volume_stats = {}
        
        for pair in self.config.INPUT_CURRENCY_PAIRS:
            df = processed_data[pair].copy()
            
            # Step 1: Process Volume with 7SD capping using ONLY Train Set statistics
            train_mask = (df.index >= train_start) & (df.index <= train_end)
            train_data = df[train_mask]
            
            if len(train_data) == 0:
                print(f"⚠️ No training data for {pair} in specified period")
                continue
            
            # Calculate Volume statistics from Train Set only
            train_volume = train_data['Volume']
            mean_train = train_volume.mean()
            std_train = train_volume.std()
            upper_limit = mean_train + 7 * std_train  # Use 7SD as specified
            
            print(f"      📊 {pair} Volume stats from Train Set:")
            print(f"         Mean: {mean_train:.2f}, SD: {std_train:.2f}, Upper Limit (7SD): {upper_limit:.2f}")
            
            # Apply capping to ALL data (train/val/test)
            df['Volume_Capped'] = np.minimum(df['Volume'], upper_limit)
            
            # Calculate Min-Max scaling parameters from Train Set capped volume
            train_volume_capped = df.loc[train_mask, 'Volume_Capped']
            min_train = train_volume_capped.min()
            max_train = train_volume_capped.max()
            
            # Apply Min-Max scaling to ALL data
            if max_train > min_train:
                df['Volume_Scaled'] = (df['Volume_Capped'] - min_train) / (max_train - min_train)
            else:
                df['Volume_Scaled'] = 0.0
            
            # Step 2: Select ONLY OHLCV features for model input (NO Technical Indicators)
            if self.config.MODEL_TYPE == 'multi':
                # Multi-currency model: include all pairs
                pair_features = df[['Open_Changed', 'High_Changed', 'Low_Changed', 'Close_Changed', 'Volume_Scaled']]
                # Prefix column names for multi-model
                new_cols = {col: f'{pair}_{col}' for col in pair_features.columns}
                pair_features = pair_features.rename(columns=new_cols)
            else:
                # Single-currency model: use only specified pair
                if pair == self.config.MODEL_TYPE:
                    pair_features = df[['Open_Changed', 'High_Changed', 'Low_Changed', 'Close_Changed', 'Volume_Scaled']]
                else:
                    continue
            
            features_list.append(pair_features)
            final_columns.extend(pair_features.columns)
            
            print(f"      ✅ {pair}: Features prepared for model input")
        
        if not features_list:
            print("❌ No features prepared for model")
            return None
        
        # Combine all features
        unified_df = pd.concat(features_list, axis=1, join='inner').dropna()
        
        # Step 3: Apply Standard Scaler to OHLC_Changed features using Train Set statistics
        train_mask_unified = (unified_df.index >= train_start) & (unified_df.index <= train_end)
        train_data_unified = unified_df[train_mask_unified]
        
        if len(train_data_unified) == 0:
            print("❌ No training data available for scaling")
            return None
        
        # Scale OHLC features only (Volume already scaled)
        scaler = StandardScaler()
        
        # Identify OHLC columns (exclude Volume_Scaled columns)
        ohlc_columns = [col for col in unified_df.columns if not col.endswith('_Volume_Scaled')]
        volume_columns = [col for col in unified_df.columns if col.endswith('_Volume_Scaled')]
        
        if ohlc_columns:
            # Fit scaler on train data only
            scaler.fit(train_data_unified[ohlc_columns])
            
            # Transform all data using train statistics
            scaled_ohlc = scaler.transform(unified_df[ohlc_columns])
            scaled_ohlc_df = pd.DataFrame(scaled_ohlc, index=unified_df.index, columns=ohlc_columns)
            
            # Combine scaled OHLC with already scaled Volume
            if volume_columns:
                final_df = pd.concat([scaled_ohlc_df, unified_df[volume_columns]], axis=1)
            else:
                final_df = scaled_ohlc_df
        else:
            final_df = unified_df
        
        # Ensure correct column order
        final_df = final_df[final_columns]
        
        print(f"      🎯 Final model input shape: {final_df.shape}")
        print(f"      📊 Features: {list(final_df.columns)}")
        print(f"      ⚠️ NOTE: RSI/MACD are calculated but NOT included in model input")
        
        return final_df
    
    def get_model_input_data_legacy(self, processed_data):
        """
        Legacy wrapper for backward compatibility with main_fx.py
        Uses config periods instead of loop_info
        """
        return self.get_model_input_data(processed_data, loop_info=None)
    
    def get_technical_indicators_for_baseline(self, processed_data, currency_pair, start_date, end_date):
        """
        Extract technical indicators for baseline strategies only.
        This is separate from model input and used only for RSI/MACD strategies.
        """
        target_df = processed_data[currency_pair]
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        
        period_data = target_df[(target_df.index >= start_dt) & (target_df.index <= end_dt)]
        
        return {
            'RSI': period_data['RSI'],
            'MACD': period_data['MACD'], 
            'MACD_Signal': period_data['MACD_Signal']
        }
    
    def get_price_data_for_period(self, processed_data, target_pair, start_date, end_date):
        """Robustly extracts price data for a given period by filtering the UTC index."""
        target_df = processed_data[target_pair]
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        return target_df[(target_df.index >= start_dt) & (target_df.index <= end_dt)]

# ==================================================================
# SequencePreparator Class - Enhanced for proper Test Set support
# ==================================================================
class SequencePreparator:
    def __init__(self, config):
        self.config = config
    
    def create_sequences_and_splits(self, model_input_data, processed_data, use_test_set=False):
        """
        Creates sequences from input data and splits them into train/eval sets.
        Enhanced to support both Validation and Test Set evaluation.
        """
        print(f"🔢 Creating sequences for target: {self.config.TARGET_PAIR}...")
        
        target_returns = processed_data[self.config.TARGET_PAIR]['Close'].pct_change().reindex(model_input_data.index).fillna(0)
        
        X, y, timestamps = [], [], []
        for i in range(len(model_input_data) - self.config.WINDOW_SIZE):
            X.append(model_input_data.iloc[i:i + self.config.WINDOW_SIZE].values)
            y.append(1 if target_returns.iloc[i + self.config.WINDOW_SIZE] > 0 else 0)
            timestamps.append(model_input_data.index[i + self.config.WINDOW_SIZE])
            
        X, y, timestamps = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.DatetimeIndex(timestamps)
        
        # Define periods with fallback to config defaults
        train_start = pd.to_datetime(getattr(self.config, 'TRAIN_START', '2019-01-01'), utc=True)
        train_end = pd.to_datetime(getattr(self.config, 'TRAIN_END', '2020-12-31'), utc=True)
        
        if use_test_set:
            eval_start = pd.to_datetime(getattr(self.config, 'TEST_START', '2021-02-01'), utc=True)
            eval_end = pd.to_datetime(getattr(self.config, 'TEST_END', '2021-02-28'), utc=True)
            eval_set_name = "Test"
        else:
            eval_start = pd.to_datetime(getattr(self.config, 'VAL_START', '2021-01-01'), utc=True)
            eval_end = pd.to_datetime(getattr(self.config, 'VAL_END', '2021-01-31'), utc=True)
            eval_set_name = "Validation"
        
        # Validate periods
        if train_start >= train_end:
            raise ValueError(f"Invalid train period: {train_start} >= {train_end}")
        if eval_start >= eval_end:
            raise ValueError(f"Invalid eval period: {eval_start} >= {eval_end}")
        
        print(f"      📅 Train: {train_start.date()} to {train_end.date()}")
        print(f"      📅 {eval_set_name}: {eval_start.date()} to {eval_end.date()}")
        
        # Create masks
        train_mask = (timestamps >= train_start) & (timestamps <= train_end)
        eval_mask = (timestamps >= eval_start) & (timestamps <= eval_end)
        
        # Split data
        train_set = (X[train_mask], y[train_mask])
        eval_set = (X[eval_mask], y[eval_mask], timestamps[eval_mask])
        
        print(f"      ✅ Train set size: {len(train_set[0])}")
        print(f"      ✅ {eval_set_name} set size: {len(eval_set[0])}")
        
        # Additional validation
        if len(train_set[0]) == 0:
            raise ValueError("No training data available for the specified period")
        if len(eval_set[0]) == 0:
            raise ValueError(f"No {eval_set_name.lower()} data available for the specified period")
        
        return train_set, eval_set
    
    def print_data_summary(self, model_input_data, processed_data):
        """Print summary of prepared data for debugging"""
        print("🔍 Data Summary:")
        print(f"   Model Input Shape: {model_input_data.shape}")
        print(f"   Date Range: {model_input_data.index[0]} to {model_input_data.index[-1]}")
        print(f"   Features: {list(model_input_data.columns)}")
        
        for pair in self.config.ALL_CURRENCY_PAIRS:
            if pair in processed_data:
                df = processed_data[pair]
                print(f"   {pair}: {len(df)} records")
        
        return True