"""
Test Visualization Module
Creates sample visualizations for testing without running full training
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import Config
from visualization import ForexVisualizer

def create_sample_data():
    """Create sample data for testing visualizations"""
    
    # Sample training history
    epochs = 50
    history = {
        'loss': np.random.exponential(0.3, epochs) + 0.2,
        'val_loss': np.random.exponential(0.4, epochs) + 0.3,
        'accuracy': np.random.beta(2, 2, epochs) * 0.3 + 0.5,
        'val_accuracy': np.random.beta(2, 2, epochs) * 0.2 + 0.45
    }
    
    # Make training curves realistic (decreasing loss, increasing accuracy)
    history['loss'] = np.sort(history['loss'])[::-1]
    history['val_loss'] = np.sort(history['val_loss'])[::-1]
    history['accuracy'] = np.sort(history['accuracy'])
    history['val_accuracy'] = np.sort(history['val_accuracy'])
    
    # Sample trading results
    trading_results = {
        'CNN-LSTM Conservative': {
            'performance': {
                'total_return': 0.0234,
                'win_rate': 0.5556,
                'sharpe_ratio': 0.8945,
                'max_drawdown': 0.0156,
                'total_trades': 45
            }
        },
        'CNN-LSTM Moderate': {
            'performance': {
                'total_return': 0.0189,
                'win_rate': 0.5128,
                'sharpe_ratio': 0.7234,
                'max_drawdown': 0.0234,
                'total_trades': 78
            }
        },
        'CNN-LSTM Aggressive': {
            'performance': {
                'total_return': 0.0098,
                'win_rate': 0.4936,
                'sharpe_ratio': 0.4521,
                'max_drawdown': 0.0345,
                'total_trades': 156
            }
        },
        'Buy and Hold': {
            'strategy_name': 'Buy and Hold',
            'total_return': 0.0145,
            'win_rate': 1.0000,
            'sharpe_ratio': 0.0000,
            'max_drawdown': 0.0089,
            'total_trades': 1
        },
        'Random': {
            'strategy_name': 'Random',
            'total_return': -0.0023,
            'win_rate': 0.4800,
            'sharpe_ratio': -0.1234,
            'max_drawdown': 0.0267,
            'total_trades': 50
        }
    }
    
    # Sample multi-currency results
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2021, 12, 31)
    timeline = pd.date_range(start=start_date, end=end_date, freq='M')
    
    multi_currency_results = {}
    currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    for strategy in ['conservative', 'moderate', 'aggressive']:
        multi_currency_results[strategy] = {}
        
        for i, pair in enumerate(currency_pairs):
            # Create realistic cumulative returns
            np.random.seed(42 + i)  # For reproducible results
            
            monthly_returns = np.random.normal(0.001, 0.02, len(timeline))
            if strategy == 'conservative':
                monthly_returns *= 0.5  # More conservative
            elif strategy == 'aggressive':
                monthly_returns *= 1.5  # More volatile
            
            cumulative_returns = np.cumsum(monthly_returns)
            
            multi_currency_results[strategy][pair] = {
                'final_return': cumulative_returns[-1],
                'win_rate': 0.45 + np.random.random() * 0.15,
                'total_trades': np.random.randint(30, 200),
                'sharpe_ratio': np.random.normal(0.5, 0.3),
                'max_drawdown': abs(np.random.normal(0.02, 0.01)),
                'cumulative_returns': cumulative_returns,
                'timestamps': timeline
            }
    
    # Sample currency analysis
    currency_analysis = {}
    for pair in currency_pairs:
        currency_analysis[pair] = {
            'total_return': np.random.normal(0.01, 0.02),
            'win_rate': 0.4 + np.random.random() * 0.2,
            'total_trades': np.random.randint(50, 150),
            'sharpe_ratio': np.random.normal(0.3, 0.4),
            'max_drawdown': abs(np.random.normal(0.02, 0.01))
        }
    
    return history, trading_results, multi_currency_results, currency_analysis

def test_visualizations():
    """Test all visualization functions"""
    print("🎨 Testing Forex Visualization System...")
    
    # Initialize
    config = Config()
    visualizer = ForexVisualizer(config)
    
    # Create sample data
    history, trading_results, multi_currency_results, currency_analysis = create_sample_data()
    
    # Sample model metrics
    eval_metrics = {
        'accuracy': 0.6462,
        'precision': 0.7165,
        'recall': 0.7510,
        'f1_score': 0.7333
    }
    
    print("\n📊 Creating sample visualizations...")
    
    try:
        # Test 1: Training curves
        print("1. Training curves...")
        visualizer.plot_training_curves(history)
        
        # Test 2: Strategy comparison
        print("2. Strategy comparison...")
        visualizer.plot_strategy_comparison(trading_results)
        
        # Test 3: Multi-currency trading
        print("3. Multi-currency trading...")
        visualizer.plot_multi_currency_trading(multi_currency_results)
        
        # Test 4: Currency pair analysis
        print("4. Currency pair analysis...")
        visualizer.plot_currency_pair_analysis(currency_analysis)
        
        # Test 5: Performance summary
        print("5. Performance summary...")
        visualizer.save_performance_summary(eval_metrics, trading_results)
        
        print("\n✅ All visualizations created successfully!")
        print(f"📁 Check the '{config.RESULTS_PATH}' folder for generated charts")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualizations()