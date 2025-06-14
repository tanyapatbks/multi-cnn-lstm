"""
CNN-LSTM Model for Multi-Currency Forex Prediction
Simplified version focusing on core model functionality
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os

class CNNLSTMModel:
    """Simplified CNN-LSTM model for forex prediction"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, verbose=True):
        """Build CNN-LSTM model architecture"""
        if verbose:
            print("🏗️  Building CNN-LSTM model...")
        
        model = Sequential([
            # CNN Layer 1 - Extract local patterns
            Conv1D(filters=self.config.CNN_FILTERS_1, 
                   kernel_size=self.config.CNN_KERNEL_SIZE, 
                   padding='same', 
                   activation='relu',
                   input_shape=(self.config.WINDOW_SIZE, self.config.TOTAL_FEATURES),
                   name='cnn_1'),
            BatchNormalization(name='bn_1'),
            
            # CNN Layer 2 - Extract complex patterns
            Conv1D(filters=self.config.CNN_FILTERS_2, 
                   kernel_size=self.config.CNN_KERNEL_SIZE, 
                   padding='same', 
                   activation='relu',
                   name='cnn_2'),
            BatchNormalization(name='bn_2'),
            
            # MaxPooling - Dimensionality reduction
            MaxPooling1D(pool_size=2, strides=2, name='maxpool'),
            
            # LSTM Layer 1 - Learn temporal dependencies
            LSTM(units=self.config.LSTM_UNITS_1, 
                 return_sequences=True, 
                 dropout=self.config.DROPOUT_RATE, 
                 recurrent_dropout=self.config.DROPOUT_RATE,
                 name='lstm_1'),
            BatchNormalization(name='bn_3'),
            
            # LSTM Layer 2 - Final temporal processing
            LSTM(units=self.config.LSTM_UNITS_2, 
                 return_sequences=False, 
                 dropout=self.config.DROPOUT_RATE, 
                 recurrent_dropout=self.config.DROPOUT_RATE,
                 name='lstm_2'),
            BatchNormalization(name='bn_4'),
            
            # Dense Layer - Feature processing
            Dense(units=self.config.DENSE_UNITS, 
                  activation='relu',
                  name='dense'),
            Dropout(self.config.DROPOUT_RATE, name='dropout'),
            
            # Output Layer - Binary classification
            Dense(units=1, 
                  activation='sigmoid',
                  name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        if verbose:
            print(f"   ✅ Model built with {model.count_params():,} parameters")
            self._print_model_summary()
        
        return model
    
    def _print_model_summary(self):
        """Print simplified model architecture"""
        print("   🏗️  Model Architecture:")
        print(f"      Input: ({self.config.WINDOW_SIZE}, {self.config.TOTAL_FEATURES})")
        print(f"      CNN: {self.config.CNN_FILTERS_1} → {self.config.CNN_FILTERS_2} filters")
        print(f"      LSTM: {self.config.LSTM_UNITS_1} → {self.config.LSTM_UNITS_2} units")
        print(f"      Dense: {self.config.DENSE_UNITS} → 1 units")
        print(f"      Output: Binary classification (0=down, 1=up)")
    
    def train_model(self, train_data, val_data, verbose=True):
        """Train the CNN-LSTM model"""
        X_train, y_train, train_timestamps = train_data
        X_val, y_val, val_timestamps = val_data
        
        if verbose:
            print("🚀 Starting model training...")
            print(f"   📊 Training: {X_train.shape[0]} samples")
            print(f"   📊 Validation: {X_val.shape[0]} samples")
            print(f"   📅 Training period: {train_timestamps.min().date()} to {train_timestamps.max().date()}")
            print(f"   📅 Validation period: {val_timestamps.min().date()} to {val_timestamps.max().date()}")
        
        # Verify data integrity
        self._verify_data(X_train, y_train, X_val, y_val)
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )
        
        if verbose:
            self._print_training_summary()
        
        return self.history
    
    def _verify_data(self, X_train, y_train, X_val, y_val):
        """Verify data integrity before training"""
        # Check for NaN values
        if np.isnan(X_train).any():
            raise ValueError("Training data contains NaN values")
        if np.isnan(X_val).any():
            raise ValueError("Validation data contains NaN values")
        
        # Check shapes
        expected_shape = (self.config.WINDOW_SIZE, self.config.TOTAL_FEATURES)
        if X_train.shape[1:] != expected_shape:
            raise ValueError(f"Training data shape mismatch: expected {expected_shape}, got {X_train.shape[1:]}")
        
        # Check class balance
        train_balance = y_train.mean()
        val_balance = y_val.mean()
        print(f"   📊 Class balance - Train: {train_balance:.3f}, Val: {val_balance:.3f}")
    
    def _setup_callbacks(self):
        """Setup improved training callbacks"""
        # Get callback configuration
        callback_config = self.config.get_callback_config() if hasattr(self.config, 'get_callback_config') else {}
        
        # Use config values or defaults
        early_stopping_patience = callback_config.get('early_stopping_patience', 15)
        reduce_lr_patience = callback_config.get('reduce_lr_patience', 7)
        reduce_lr_factor = callback_config.get('reduce_lr_factor', 0.5)
        min_lr = callback_config.get('min_lr', 1e-6)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=early_stopping_patience, 
                restore_best_weights=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=reduce_lr_factor, 
                patience=reduce_lr_patience, 
                min_lr=min_lr, 
                verbose=1
            ),
            ModelCheckpoint(
                f"{self.config.MODELS_PATH}best_model.h5", 
                monitor='val_accuracy', 
                save_best_only=True, 
                verbose=1
            )
        ]
        return callbacks
    
    def _print_training_summary(self):
        """Print training summary"""
        print("\n   ✅ Training completed!")
        
        # Get training metrics
        final_epoch = len(self.history.history['loss'])
        best_val_acc = max(self.history.history['val_accuracy'])
        best_val_loss = min(self.history.history['val_loss'])
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        print(f"   📊 Training Summary:")
        print(f"      Epochs completed: {final_epoch}/{self.config.EPOCHS}")
        print(f"      Best validation accuracy: {best_val_acc:.4f}")
        print(f"      Best validation loss: {best_val_loss:.4f}")
        print(f"      Final training loss: {final_train_loss:.4f}")
        print(f"      Final validation loss: {final_val_loss:.4f}")
    
    def predict(self, X, verbose=False):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        predictions = self.model.predict(X, verbose=0 if not verbose else 1)
        return predictions.flatten()
    
    def evaluate_model_enhanced(self, test_data, verbose=True):
        """Enhanced evaluation with proper binary classification metrics"""
        X_test, y_test, test_timestamps = test_data
        
        if verbose:
            print("📊 Evaluating model with enhanced metrics...")
        
        # Get predictions (probabilities)
        predictions_proba = self.predict(X_test)
        binary_predictions = (predictions_proba > 0.5).astype(int)
        
        # Import necessary libraries
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, log_loss,
            confusion_matrix, classification_report
        )
        
        # Calculate comprehensive metrics
        metrics = {
            # Basic metrics (existing)
            'accuracy': accuracy_score(y_test, binary_predictions),
            'precision': precision_score(y_test, binary_predictions),
            'recall': recall_score(y_test, binary_predictions),
            'f1_score': f1_score(y_test, binary_predictions),
            
            # Advanced binary classification metrics
            'auc_roc': roc_auc_score(y_test, predictions_proba),
            'auc_pr': average_precision_score(y_test, predictions_proba),
            'log_loss': log_loss(y_test, predictions_proba),
            
            # Confusion matrix
            'confusion_matrix': confusion_matrix(y_test, binary_predictions),
            
            # Additional useful metrics
            'true_positives': np.sum((y_test == 1) & (binary_predictions == 1)),
            'true_negatives': np.sum((y_test == 0) & (binary_predictions == 0)),
            'false_positives': np.sum((y_test == 0) & (binary_predictions == 1)),
            'false_negatives': np.sum((y_test == 1) & (binary_predictions == 0)),
            
            # For compatibility
            'predictions': predictions_proba,
            'binary_predictions': binary_predictions
        }
        
        # Calculate additional trading-specific metrics
        # Directional accuracy (important for trading)
        metrics['directional_accuracy'] = metrics['accuracy']
        
        # Matthews Correlation Coefficient (good for imbalanced classes)
        from sklearn.metrics import matthews_corrcoef
        metrics['mcc'] = matthews_corrcoef(y_test, binary_predictions)
        
        if verbose:
            print(f"\n📊 BINARY CLASSIFICATION METRICS:")
            print(f"   Basic Metrics:")
            print(f"      Accuracy: {metrics['accuracy']:.4f}")
            print(f"      Precision: {metrics['precision']:.4f}")
            print(f"      Recall: {metrics['recall']:.4f}")
            print(f"      F1-Score: {metrics['f1_score']:.4f}")
            
            print(f"\n   Advanced Metrics:")
            print(f"      AUC-ROC: {metrics['auc_roc']:.4f} (Area Under ROC Curve)")
            print(f"      AUC-PR: {metrics['auc_pr']:.4f} (Area Under Precision-Recall)")
            print(f"      Log Loss: {metrics['log_loss']:.4f} (Binary Cross-entropy)")
            print(f"      MCC: {metrics['mcc']:.4f} (Matthews Correlation)")
            
            print(f"\n   Confusion Matrix:")
            print(f"      True Positives: {metrics['true_positives']}")
            print(f"      True Negatives: {metrics['true_negatives']}")
            print(f"      False Positives: {metrics['false_positives']}")
            print(f"      False Negatives: {metrics['false_negatives']}")
            
            print(f"\n   Class Balance:")
            print(f"      Test set: {y_test.mean():.3f} (1=up, 0=down)")
            print(f"      Predictions: {binary_predictions.mean():.3f}")
        
        return metrics

    # def evaluate_model(self, test_data, verbose=True):
    #     """Evaluate model performance"""
    #     X_test, y_test, test_timestamps = test_data
        
    #     if verbose:
    #         print("📊 Evaluating model performance...")
        
    #     # Make predictions
    #     predictions = self.predict(X_test)
    #     binary_predictions = (predictions > 0.5).astype(int)
        
    #     # Calculate metrics
    #     accuracy = np.mean(binary_predictions == y_test)
    #     precision = self._calculate_precision(y_test, binary_predictions)
    #     recall = self._calculate_recall(y_test, binary_predictions)
    #     f1_score = self._calculate_f1_score(precision, recall)
        
    #     metrics = {
    #         'accuracy': accuracy,
    #         'precision': precision,
    #         'recall': recall,
    #         'f1_score': f1_score,
    #         'predictions': predictions,
    #         'binary_predictions': binary_predictions
    #     }
        
    #     if verbose:
    #         print(f"   📊 Model Performance:")
    #         print(f"      Accuracy: {accuracy:.4f}")
    #         print(f"      Precision: {precision:.4f}")
    #         print(f"      Recall: {recall:.4f}")
    #         print(f"      F1-Score: {f1_score:.4f}")
    #         print(f"      Test period: {test_timestamps.min().date()} to {test_timestamps.max().date()}")
        
    #     return metrics
    
    def compare_model_metrics(multi_metrics, single_metrics):
        """Compare metrics between multi and single currency models"""
        
        comparison = {
            'model_type': ['Multi-Currency', 'Single-Currency'],
            'accuracy': [multi_metrics['accuracy'], single_metrics['accuracy']],
            'auc_roc': [multi_metrics['auc_roc'], single_metrics['auc_roc']],
            'log_loss': [multi_metrics['log_loss'], single_metrics['log_loss']],
            'f1_score': [multi_metrics['f1_score'], single_metrics['f1_score']],
            'mcc': [multi_metrics['mcc'], single_metrics['mcc']]
        }
        
        import pandas as pd
        df = pd.DataFrame(comparison)
        
        print("\n📊 MODEL COMPARISON - BINARY CLASSIFICATION METRICS:")
        print(df.to_string(index=False))
        
        # Determine winner
        multi_wins = 0
        single_wins = 0
        
        # Higher is better: accuracy, auc_roc, f1_score, mcc
        for metric in ['accuracy', 'auc_roc', 'f1_score', 'mcc']:
            if multi_metrics[metric] > single_metrics[metric]:
                multi_wins += 1
            else:
                single_wins += 1
        
        # Lower is better: log_loss
        if multi_metrics['log_loss'] < single_metrics['log_loss']:
            multi_wins += 1
        else:
            single_wins += 1
        
        print(f"\n🏆 Winner: {'Multi-Currency' if multi_wins > single_wins else 'Single-Currency'} Model")
        print(f"   (Won {max(multi_wins, single_wins)} out of 5 metrics)")
        
        return df
    
    def _calculate_precision(self, y_true, y_pred):
        """Calculate precision"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _calculate_recall(self, y_true, y_pred):
        """Calculate recall"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _calculate_f1_score(self, precision, recall):
        """Calculate F1-score"""
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f"{self.config.MODELS_PATH}trained_model.h5"
        
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            filepath = f"{self.config.MODELS_PATH}trained_model.h5"
        
        self.model = tf.keras.models.load_model(filepath)
        print(f"✅ Model loaded from {filepath}")
        return self.model
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return "Model not built"
        
        return {
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'layers': len(self.model.layers),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape
        }