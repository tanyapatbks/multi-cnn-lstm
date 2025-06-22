การพยากรณ์อนุกรมเวลาแบบหลายคู่สกุลเงินโดยใช้โครงข่ายประสาทเทียมแบบ CNN-LSTM
Multi-Currency Time Series Forecasting Using CNN-LSTM Networks
🎯 ภาพรวมโปรเจค (Project Overview)
โปรเจคนี้เป็นการพัฒนาระบบพยากรณ์อัตราแลกเปลี่ยนเงินตราต่างประเทศ (Forex) โดยใช้แบบจำลอง Deep Learning แบบผสม (Hybrid Model) ระหว่าง Convolutional Neural Network (CNN) และ Long Short-Term Memory (LSTM)

หัวใจสำคัญ: การสร้างแบบจำลองที่สามารถเรียนรู้ความสัมพันธ์เชิงโครงสร้างและเชิงเวลาจาก หลายคู่สกุลเงินพร้อมกัน (Multi-Currency) ได้แก่ EURUSD, GBPUSD, และ USDJPY

📊 ข้อมูลและการประมวลผล (Data & Processing)
🔹 ข้อมูลที่ใช้
ข้อมูล: OHLCV รายชั่วโมงจากคู่สกุลเงิน 3 คู่: EUR/USD, GBP/USD, USD/JPY
ช่วงเวลา: 2018-2022 (รวม 5 ปี)
ไม่มี Technical Indicators เพิ่มเติม - ใช้เฉพาะ OHLCV
🔹 การประมวลผลข้อมูล
OHLC: แปลงเป็น %Change → Z-score normalization (mean≈0, std≈1)
Volume: 7-SD outlier capping → Min-Max scaling [0,1]
Input Shape:
Multi-currency: (32, 60, 15) = batch_size × time_steps × features (3 pairs × 5 OHLCV)
Single-currency: (32, 60, 5) = batch_size × time_steps × features (1 pair × 5 OHLCV)
🏗️ สถาปัตยกรรมโมเดล (Model Architecture)
Input Layer: (60, 15) หรือ (60, 5)
    ↓
1st Conv1D Layer: 64 filters, kernel_size=3, ReLU
    ↓
2nd Conv1D Layer: 128 filters, kernel_size=3, ReLU
    ↓
MaxPooling1D Layer: pool_size=2
    ↓
1st LSTM Layer: 128 units, dropout=0.2, return_sequences=True
    ↓
2nd LSTM Layer: 64 units, dropout=0.2, return_sequences=False
    ↓
Dense Layer: 64 units, ReLU
    ↓
Output Layer: 1 unit, Sigmoid (ความน่าจะเป็น 0-1)
📈 การทดลองแบบ Rolling Window
🔹 โครงสร้าง 12 Loops
แต่ละ loop เลื่อนเวลาข้างหน้า 1 เดือน:

Loop 1:  Train: 2018-12-01 → 2020-11-30 | Val: 2020-12-01 → 2020-12-31 | Test: 2021-01-01 → 2021-01-31
Loop 2:  Train: 2019-01-01 → 2020-12-31 | Val: 2021-01-01 → 2021-01-31 | Test: 2021-02-01 → 2021-02-28  
Loop 3:  Train: 2019-02-01 → 2021-01-31 | Val: 2021-02-01 → 2021-02-28 | Test: 2021-03-01 → 2021-03-31
...
Loop 12: Train: 2019-11-01 → 2021-10-31 | Val: 2021-11-01 → 2021-11-30 | Test: 2021-12-01 → 2021-12-31
🔹 ไม่เกิด Data Leakage เพราะ:
Temporal order ถูกต้อง (Train → Val → Test)
แต่ละ loop เริ่มต้นใหม่ทั้งหมด
Test periods ไม่ซ้อนทับกัน
🎯 กลยุทธ์การซื้อขาย (Trading Strategies)
🔹 CNN-LSTM Thresholds
Conservative: Buy ≥ 0.7, Sell ≤ 0.3 (Leverage 2.0x)
Moderate: Buy ≥ 0.6, Sell ≤ 0.4 (Leverage 1.0x)
Aggressive: Buy ≥ 0.55, Sell ≤ 0.45 (Leverage 0.5x)
🔹 Baseline Strategies
Buy & Hold: ซื้อและถือตลอดระยะเวลา
RSI-based: ใช้ RSI (14) กับ oversold/overbought levels
MACD-based: ใช้ MACD signal crossovers
🔹 Trading Rules
Holding Period: อย่างน้อย 1 ชั่วโมง, ไม่เกิน 3 ชั่วโมง
Stop Loss: 2%
Position: เพียง 1 ตำแหน่งในแต่ละช่วงเวลา
🚀 การติดตั้งและใช้งาน
ขั้นตอนที่ 1: ติดตั้ง Dependencies
bash
pip install -r requirements.txt
ขั้นตอนที่ 2: เตรียมข้อมูล
วางไฟล์ข้อมูล CSV ในโฟลเดอร์ data/:

data/
├── EURUSD_1H.csv
├── GBPUSD_1H.csv
└── USDJPY_1H.csv
รูปแบบไฟล์ CSV:

csv
DateTime,Open,High,Low,Close,Volume
2018-12-01 00:00:00,1.20137,1.20158,1.20026,1.20106,6885.930
ขั้นตอนที่ 3: รันการทดลอง
🔥 รันการทดลองหลัก 12-Loops (แนะนำ)
bash
# รันด้วย test set evaluation
python run_experiments.py --use-test-set --threshold-choice Moderate

# รันด้วย validation set (สำหรับพัฒนา)
python run_experiments.py --threshold-choice Moderate

# รันแบบครบถ้วน (ใช้เวลานาน)
python run_experiments.py --use-test-set --all-thresholds
🧪 รันทดสอบโมเดลเดียว
bash
# Multi-currency model
python main_fx.py --model multi --target EURUSD --use-test-set

# Single-currency model  
python main_fx.py --model EURUSD --target EURUSD --use-test-set

# เปรียบเทียบหลายโมเดล
python main_fx.py --mode comparison --use-test-set
🔍 เปรียบเทียบ Validation vs Test
bash
python run_experiments.py --mode comparison --threshold-choice Moderate
📊 ผลลัพธ์ที่ได้
หลังจากรันเสร็จ จะได้ไฟล์ผลลัพธ์ในโฟลเดอร์ results/:

🔹 ตารางข้อมูล (CSV Files)
results/rolling_window_12_loops_with_test/
├── EURUSD_Monthly_Return.csv          # ตารางผลตอบแทนรายเดือน
├── EURUSD_Monthly_Sharpe.csv          # ตาราง Sharpe Ratio รายเดือน
├── EURUSD_Summary_Metrics.csv         # ตารางสรุปค่าเฉลี่ย
├── EURUSD_Combined_Results.csv        # ไฟล์รวม 3 ตาราง
├── EURUSD_Performance_Ranking.csv     # อันดับ strategies
├── GBPUSD_*.csv                       # ไฟล์เดียวกันสำหรับ GBPUSD
├── USDJPY_*.csv                       # ไฟล์เดียวกันสำหรับ USDJPY
└── Validation_vs_Test_Comparison.csv  # เปรียบเทียบ val vs test
🔹 กราฟและแผนภูมิ (PNG Files)
Training curves (Loss/Accuracy)
Strategy performance comparison
Rolling window analysis
Currency pair comparison
Performance heatmaps
📋 ตัวอย่างผลลัพธ์
🔹 Monthly Return Table (EURUSD)
Month	Conservative	Moderate	Aggressive	Single-CNN-LSTM	Buy & Hold	RSI	MACD
Jan 2021	2.45	1.83	0.92	1.56	0.98	0.23	1.12
Feb 2021	-1.23	-0.87	-0.44	-0.65	-0.32	-0.15	0.78
...	...	...	...	...	...	...	...
🔹 Summary Metrics Table (EURUSD)
Metric	Conservative	Moderate	Aggressive	Single-CNN-LSTM	Buy & Hold	RSI	MACD
Avg. Total Return (%)	15.6	12.3	8.9	10.2	7.8	4.5	9.1
Avg. Sharpe Ratio	1.23	1.01	0.78	0.89	0.67	0.45	0.82
Avg. Win Rate (%)	65.2	62.1	58.9	60.5	55.2	52.1	57.8
Avg. Max Drawdown (%)	8.5	6.2	4.1	5.8	9.1	7.3	6.9
Avg. Total Trades	45	52	58	48	2	38	41
⚙️ การปรับแต่งพารามิเตอร์
🔹 ไฟล์ config.py
python
# Model Architecture
CNN_FILTERS_1 = 64
CNN_FILTERS_2 = 128
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64

# Training
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Trading
STOP_LOSS_PCT = 2.0
MAX_HOLDING_HOURS = 3
INITIAL_CAPITAL = 10000
🔹 การปรับ Thresholds
python
THRESHOLDS = {
    'Conservative': {'buy': 0.7, 'sell': 0.3, 'leverage': 2.0},
    'Moderate': {'buy': 0.6, 'sell': 0.4, 'leverage': 1.0},
    'Aggressive': {'buy': 0.55, 'sell': 0.45, 'leverage': 0.5}
}
🔧 โครงสร้างโปรเจค
📁 PROJECT ROOT/
├── 📜 config.py                    # ตั้งค่าทั้งหมด
├── 📜 data_processor.py            # ประมวลผลข้อมูล OHLCV
├── 📜 cnn_lstm_model.py            # สถาปัตยกรรม CNN-LSTM
├── 📜 trading_strategy.py          # กลยุทธ์การเทรดและ simulation
├── 📜 rolling_window_experiment.py # การทดลอง 12 loops
├── 📜 visualization.py             # สร้างกราฟและแผนภูมิ
├── 📜 main_fx.py                   # รันทดสอบโมเดลเดียว
├── 📜 run_experiments.py           # 🔥 รันการทดลอง 12 loops
├── 📜 requirements.txt             # รายการ library
├── 📜 README.md                    # เอกสารนี้
│
├── 📁 data/                        # ข้อมูล CSV
│   ├── EURUSD_1H.csv
│   ├── GBPUSD_1H.csv
│   └── USDJPY_1H.csv
│
├── 📁 models/                      # โมเดลที่ฝึกเสร็จ (.h5)
│
└── 📁 results/                     # ผลลัพธ์และรายงาน
    ├── rolling_window_12_loops_with_test/
    ├── single_run_*/
    └── multi_model_comparison/
⏱️ ระยะเวลาการทำงาน
Model training per loop: ~5-10 นาที
Models per loop: 6 โมเดล (3 multi + 3 single)
Total loops: 12 loops
ประมาณการรวม: 6-12 ชั่วโมง
💡 เคล็ดลับ: ใช้ --unattended สำหรับการรันอัตโนมัติ

🏆 ประโยชน์ที่ได้รับ
Robust Evaluation: ผลการทดสอบจากหลายช่วงเวลา
Real-world Simulation: จำลองการใช้งานจริงที่ต้องอัพเดท model
Comprehensive Analysis: ตารางและกราฟที่ครบถ้วน
Academic Rigor: ตรงมาตรฐานงานวิจัย time series
📚 การอ้างอิง (References)
งานวิจัยนี้อ้างอิงจากแนวคิดในเอกสารวิชาการด้าน:

Multi-Currency Time Series Forecasting
CNN-LSTM Hybrid Models
Forex Trading Strategy Development
Rolling Window Validation
👨‍💼 ผู้จัดทำ
ร.ท.ธัญภัทร บุญเกษม ร.น.

รหัสนิสิต: 6670116421
หลักสูตร: วิทยาศาสตร์มหาบัณฑิต สาขาวิศวกรรมคอมพิวเตอร์
สถาบัน: จุฬาลงกรณ์มหาวิทยาลัย
อาจารย์ที่ปรึกษา: ผศ.ดร.พิตติพล คันธวัฒน์
อาจารย์ที่ปรึกษาร่วม: ผศ.ดร.กฤษฎา นิมมานันทน์

🎉 สรุป
โปรเจคนี้สาธิตให้เห็นว่าการใช้ Multi-Currency CNN-LSTM สามารถปรับปรุงประสิทธิภาพการพยากรณ์ Forex ได้ดีกว่าการวิเคราะห์แบบคู่เงินเดียว โดยใช้การทดลองแบบ Rolling Window 12 loops ที่ครอบคลุมและเข้มงวด

ไฮไลท์:

✅ ไม่มี Data Leakage
✅ Realistic Trading Simulation
✅ Comprehensive Evaluation
✅ Ready-to-use Results & Visualizations
