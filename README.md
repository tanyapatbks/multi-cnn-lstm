Multi-Currency CNN-LSTM Forex Prediction System
Overview
This is a streamlined version of the multi-currency forex prediction system that focuses on core functionality while removing unnecessary complexity. The system uses CNN-LSTM neural networks to predict forex trends across multiple currency pairs (EURUSD, GBPUSD, USDJPY) and evaluates performance using a Fixed Holding Period Trading Strategy.

Key Features
✅ Clean Architecture: Modular code structure
✅ Core Functionality: CNN-LSTM model training and evaluation
✅ Multi-Currency Support: EURUSD, GBPUSD, USDJPY
✅ Trading Strategy: Fixed Holding Period with risk management
✅ Comprehensive Visualization: Training curves, strategy comparison, multi-currency analysis
✅ Checkpoint System: Resume training from any step
✅ No Data Leakage: Proper temporal data splitting
✅ Model Compatibility: Works with existing .h5 files

Project Structure
multiv_fx/
├── config.py                       # Configuration settings
├── data_processor.py               # Data loading and preprocessing
├── cnn_lstm_model.py              # CNN-LSTM model implementation
├── trading_strategy.py            # Trading strategies
├── checkpoint.py                  # Checkpoint management
├── visualization.py               # Comprehensive visualization
├── main_fx.py                     # Main execution file
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── data/                          # Currency data (CSV files)
│   ├── EURUSD_1H.csv
│   ├── GBPUSD_1H.csv
│   └── USDJPY_1H.csv
├── models/                        # Saved models
│   ├── trained_model.h5           # Main model file
│   └── best_model.h5              # Best validation model
└── results/                       # Output results
    ├── experiment_summary.pkl
    ├── cnn_lstm_metrics.pkl
    ├── fixed_holding_results.pkl
    ├── multi_currency_results.pkl
    ├── currency_analysis.pkl
    ├── training_curves.png
    ├── strategy_comparison.png
    ├── multi_currency_trading.png
    ├── currency_pair_analysis.png
    └── performance_summary.txt
Model Files Explanation
ในโฟลเดอร์ models/ คุณมีไฟล์โมเดล 2 ไฟล์:

trained_model.h5: โมเดลหลักที่บันทึกหลังจากการเทรนเสร็จสิ้น
best_model.h5: โมเดลที่มี validation accuracy ดีที่สุดระหว่างการเทรน (ใช้โดย ModelCheckpoint callback)
ระบบจะใช้ best_model.h5 เป็นหลัก เพราะมีประสิทธิภาพดีกว่า แต่จะ fallback ไปใช้ trained_model.h5 หากไม่พบ

Quick Start
1. Installation
bash
# Install dependencies
pip install -r requirements.txt
2. Data Setup
Ensure your data files are in the data/ directory with format:

csv
Local time,Open,High,Low,Close,Volume
13.01.2018 00:00:00.000 GMT+0700,1.20123,1.20456,1.19987,1.20234,12345
3. Basic Usage
bash
# Run complete pipeline
python main_fx.py

# Start from specific step
python main_fx.py --step 3

# Use test set for final evaluation
python main_fx.py --test

# Start fresh (delete checkpoint)
python main_fx.py --new

# Create visualizations only (from saved results)
python main_fx.py --visualize
Execution Steps
Data Loading & Preprocessing 📚
Load OHLCV data for all currency pairs
Convert to percentage returns (stationary data)
Normalize features using StandardScaler and MinMaxScaler
Sequence Preparation 📋
Create 60-hour sliding windows
Generate binary labels (up/down prediction)
Split data temporally (2018-2020: train, 2021: val, 2022: test)
Model Training 🏗️
Build CNN-LSTM architecture
Train with early stopping and learning rate reduction
Save both trained_model.h5 and best_model.h5
Model Evaluation 📊
Load best performing model
Evaluate on validation/test set
Calculate accuracy, precision, recall, F1-score
Trading Strategy Testing 💼
Apply Fixed Holding Period strategies (Conservative, Moderate, Aggressive)
Compare with baseline strategies (Buy & Hold, Random)
Calculate trading performance metrics
Results Summary & Visualization 📈
Generate comprehensive performance report
Create training curves and strategy comparison charts
Visualize multi-currency trading performance
Save all results and metrics
Visualization Features
The system creates comprehensive visualizations automatically:

📊 Training Analysis
Training vs Validation Loss curves
Training vs Validation Accuracy curves
Best epoch markers and performance indicators
Overfitting analysis
💼 Strategy Performance
Total returns comparison across all strategies
Win rates for each strategy
Sharpe ratios comparison
Maximum drawdown analysis
🌍 Multi-Currency Analysis
Conservative Strategy: Performance across EURUSD (Blue), GBPUSD (Orange), USDJPY (Green)
Moderate Strategy: Performance across all three currency pairs
Aggressive Strategy: Performance across all three currency pairs
Individual currency pair analysis
📈 Generated Charts
All charts are saved as high-resolution PNG files in results/:

training_curves.png - Model training performance
strategy_comparison.png - Strategy performance comparison
multi_currency_trading.png - Multi-currency trading results
currency_pair_analysis.png - Individual currency analysis
performance_summary.txt - Text summary of all results
Configuration
Key parameters in config.py:

python
# Model Architecture
WINDOW_SIZE = 60           # Hours of historical data
CNN_FILTERS_1 = 64         # First CNN layer filters
CNN_FILTERS_2 = 128        # Second CNN layer filters
LSTM_UNITS_1 = 128         # First LSTM layer units
LSTM_UNITS_2 = 64          # Second LSTM layer units

# Trading Strategy Thresholds
THRESHOLDS = {
    'conservative': {'buy': 0.7, 'sell': 0.3},
    'moderate': {'buy': 0.6, 'sell': 0.4},
    'aggressive': {'buy': 0.55, 'sell': 0.45}
}
Model Architecture
Input: (60, 15) - 60 hours × 15 features (3 currencies × 5 OHLCV)
    ↓
CNN Layer 1: 64 filters, kernel_size=3
    ↓
CNN Layer 2: 128 filters, kernel_size=3
    ↓
MaxPooling: pool_size=2
    ↓
LSTM Layer 1: 128 units
    ↓
LSTM Layer 2: 64 units
    ↓
Dense: 32 units
    ↓
Output: 1 unit (sigmoid) - Binary classification
Trading Strategy
Fixed Holding Period Strategy:

Entry: Based on prediction confidence thresholds
Holding Period: 1-3 hours
Exit Conditions:
Stop Loss: -2%
Take Profit: After minimum 1 hour if positive
Time Limit: Maximum 3 hours
Working with Existing Models
หากคุณมีโมเดลเดิมอยู่แล้ว:

python
# ระบบจะโหลดโมเดลตามลำดับความสำคัญ:
# 1. best_model.h5 (ถ้ามี)
# 2. trained_model.h5 (ถ้าไม่มี best_model.h5)

# การโหลดโมเดลด้วยตนเอง:
from cnn_lstm_model import CNNLSTMModel
from config import Config

config = Config()
model_builder = CNNLSTMModel(config)

# โหลดโมเดลเฉพาะ
model_builder.load_model("models/best_model.h5")
Checkpoint System
ระบบ checkpoint ใหม่:

ใช้ไฟล์ checkpoints/checkpoint.pkl
ไม่กระทบ checkpoint เดิม
Resume ได้จากทุกขั้นตอน
bash
# ดูสถานะ checkpoint
python -c "from checkpoint import CheckpointManager; from config import Config; cm = CheckpointManager(Config()); print(cm.get_checkpoint_info())"
Results Output
📈 MODEL PERFORMANCE SUMMARY
Accuracy: 0.5420
Precision: 0.5315
Recall: 0.6108
F1-Score: 0.5686

💼 TRADING STRATEGY PERFORMANCE
Strategy             Trades   Return     Win Rate   Sharpe     Max DD    
CNN-LSTM Conservative 45       0.0234     0.5556     0.8945     0.0156    
CNN-LSTM Moderate     78       0.0189     0.5128     0.7234     0.0234    
CNN-LSTM Aggressive   156      0.0098     0.4936     0.4521     0.0345    
Buy and Hold          1        0.0145     1.0000     0.0000     0.0089    
Random               50        -0.0023    0.4800     -0.1234    0.0267    

🏆 Best performing strategy: CNN-LSTM Conservative (Return: 0.0234)
Performance Expectations
Typical performance metrics:

Model Accuracy: 52-58%
Trading Return: -0.05 to +0.15 (varies by strategy)
Win Rate: 0.45-0.55
Sharpe Ratio: -0.5 to +1.5
Troubleshooting
Common Issues:

Data Loading Error: Check datetime format in CSV files
Model Loading Error:
bash
# ลบ checkpoint และเริ่มใหม่
python main_fx.py --new
Shape Mismatch: Verify all currency pairs have the same time range
Memory Error: Reduce batch size in config.py
File Locations:

Models: models/trained_model.h5, models/best_model.h5
Results: results/*.pkl
Checkpoints: checkpoints/checkpoint.pkl
Data: data/*.csv
Compatibility
✅ Preserves your existing model files (.h5)
✅ Maintains the same core algorithms
✅ Uses separate checkpoint files to avoid conflicts
✅ Produces comparable results to the original system

This streamlined version maintains the scientific rigor of the original system while providing a cleaner, more maintainable codebase for your Master's thesis research.

# multi-cnn-lstm
