"""
Checkpoint Manager for Multi-Currency CNN-LSTM Forex Prediction
Streamlined version for step-wise execution without complex features
"""

import pickle
import os
from datetime import datetime

class CheckpointManager:
    """Simplified checkpoint manager for step-wise execution"""
    
    def __init__(self, config):
        self.config = config
        self.checkpoint_file = f"{config.CHECKPOINTS_PATH}checkpoint.pkl"
        
    def save_checkpoint(self, step, data, verbose=True):
        """Save checkpoint for specific step"""
        checkpoint = {
            'step': step,
            'timestamp': datetime.now(),
            'data': data
        }
        
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            if verbose:
                print(f"💾 Checkpoint saved for step: {step}")
                
        except Exception as e:
            print(f"❌ Error saving checkpoint: {str(e)}")
    
    def load_checkpoint(self, verbose=True):
        """Load latest checkpoint"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                if verbose:
                    print(f"💾 Checkpoint loaded from step: {checkpoint['step']}")
                    print(f"   📅 Saved at: {checkpoint['timestamp']}")
                
                return checkpoint
                
            except Exception as e:
                print(f"❌ Error loading checkpoint: {str(e)}")
                return None
        else:
            if verbose:
                print("💾 No checkpoint found")
            return None
    
    def checkpoint_exists(self):
        """Check if checkpoint exists"""
        return os.path.exists(self.checkpoint_file)
    
    def delete_checkpoint(self, verbose=True):
        """Delete existing checkpoint"""
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                if verbose:
                    print("🗑️  Checkpoint deleted")
            except Exception as e:
                print(f"❌ Error deleting checkpoint: {str(e)}")
        else:
            if verbose:
                print("💾 No checkpoint to delete")
    
    def list_available_steps(self):
        """List available execution steps"""
        steps = [
            "1_data_loading",
            "2_data_preprocessing", 
            "3_sequence_creation",
            "4_model_training",
            "5_model_evaluation",
            "6_strategy_testing",
            "7_final_results"
        ]
        return steps
    
    def get_checkpoint_info(self):
        """Get information about current checkpoint"""
        checkpoint = self.load_checkpoint(verbose=False)
        
        if checkpoint is None:
            return "No checkpoint found"
        
        info = {
            'step': checkpoint['step'],
            'timestamp': checkpoint['timestamp'],
            'data_keys': list(checkpoint['data'].keys()) if 'data' in checkpoint else []
        }
        
        return info

class ResultsManager:
    """Simple results manager for saving and loading results"""
    
    def __init__(self, config):
        self.config = config
        
    def save_results(self, results, filename, verbose=True):
        """Save results to file"""
        filepath = f"{self.config.RESULTS_PATH}{filename}"
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
            
            if verbose:
                print(f"💾 Results saved to {filepath}")
                
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
    
    def load_results(self, filename, verbose=True):
        """Load results from file"""
        filepath = f"{self.config.RESULTS_PATH}{filename}"
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    results = pickle.load(f)
                
                if verbose:
                    print(f"💾 Results loaded from {filepath}")
                
                return results
                
            except Exception as e:
                print(f"❌ Error loading results: {str(e)}")
                return None
        else:
            if verbose:
                print(f"❌ Results file not found: {filepath}")
            return None
    
    def save_model_metrics(self, metrics, model_name="cnn_lstm"):
        """Save model performance metrics"""
        filename = f"{model_name}_metrics.pkl"
        self.save_results(metrics, filename)
    
    def save_trading_results(self, trading_results, strategy_name="fixed_holding"):
        """Save trading strategy results"""
        filename = f"{strategy_name}_results.pkl"
        self.save_results(trading_results, filename)
    
    def create_summary_report(self, model_metrics, trading_results, verbose=True):
        """Create a simple summary report"""
        summary = {
            'timestamp': datetime.now(),
            'model_performance': model_metrics,
            'trading_performance': trading_results,
            'config': {
                'currency_pairs': self.config.CURRENCY_PAIRS,
                'window_size': self.config.WINDOW_SIZE,
                'model_architecture': f"CNN({self.config.CNN_FILTERS_1},{self.config.CNN_FILTERS_2})+LSTM({self.config.LSTM_UNITS_1},{self.config.LSTM_UNITS_2})",
                'training_period': f"{self.config.TRAIN_START} to {self.config.TRAIN_END}",
                'validation_period': f"{self.config.VAL_START} to {self.config.VAL_END}"
            }
        }
        
        self.save_results(summary, "experiment_summary.pkl", verbose)
        
        if verbose:
            print("\n📊 Experiment Summary:")
            print("-" * 50)
            if 'accuracy' in model_metrics:
                print(f"Model Accuracy: {model_metrics['accuracy']:.4f}")
            
            if 'performance' in trading_results:
                perf = trading_results['performance']
                print(f"Trading Return: {perf.get('total_return', 0):.4f}")
                print(f"Win Rate: {perf.get('win_rate', 0):.4f}")
                print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.4f}")
        
        return summary
    
    def list_saved_files(self):
        """List all saved result files"""
        if not os.path.exists(self.config.RESULTS_PATH):
            return []
        
        files = [f for f in os.listdir(self.config.RESULTS_PATH) if f.endswith('.pkl')]
        return files