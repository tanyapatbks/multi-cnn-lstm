import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def analyze_volume_outliers(filepath):
    """
    วิเคราะห์ข้อมูลที่โดดผิดปกติ (Outlier) เฉพาะในคอลัมน์ "Volume"
    จากไฟล์ CSV ที่กำหนด
    """
    try:
        # โหลดข้อมูลจากไฟล์ CSV
        df = pd.read_csv(filepath)
        print(f"--- Analyzing Outliers for: {filepath} ---")
    except FileNotFoundError:
        print(f"❌ Error: ไม่พบไฟล์ที่ {filepath}")
        return

    if 'Volume' not in df.columns:
        print("❌ Error: ไม่พบคอลัมน์ 'Volume' ในไฟล์นี้")
        return

    # เลือกเฉพาะคอลัมน์ 'Volume' และตัดค่าที่หายไปทิ้ง
    volume = df['Volume'].dropna()
    total_points = len(volume)
    
    # คำนวณค่าสถิติพื้นฐาน
    mean_volume = volume.mean()
    std_volume = volume.std()
    
    print(f"Total Data Points: {total_points:,}")
    print(f"Mean Volume: {mean_volume:,.2f}")
    print(f"Std Dev of Volume: {std_volume:,.2f}\n")
    
    # วิเคราะห์จำนวน Outlier ตามเกณฑ์ Standard Deviation (SD) ที่ต่างกัน
    print("--- Outlier Analysis by Standard Deviation ---")
    print(f"{'SD Threshold':<15} | {'Upper Limit':<20} | {'Outliers Found':<18} | {'% of Total Data':<18}")
    print("-" * 80)
    
    for k in range(1, 11): # ทดสอบที่เกณฑ์ 1-10 SD
        upper_limit = mean_volume + (k * std_volume)
        
        # นับจำนวนข้อมูลที่สูงเกินขีดจำกัด
        outliers_count = (volume > upper_limit).sum()
        
        # คำนวณเป็นเปอร์เซ็นต์
        percentage_cut = (outliers_count / total_points) * 100 if total_points > 0 else 0
        
        print(f"{k}-SD             | {upper_limit:<20,.2f} | {outliers_count:<18,} | {percentage_cut:<18.4f}%")

if __name__ == '__main__':
    # ==================================================================
    # <<< แก้ไข: เพิ่ม 'data/' เข้าไปหน้าชื่อไฟล์ให้ถูกต้อง >>>
    # ==================================================================
    currency_files = ['data/EURUSD_1H.csv', 'data/GBPUSD_1H.csv', 'data/USDJPY_1H.csv']
    
    for file in currency_files:
        analyze_volume_outliers(file)
        print("\n" + "="*80 + "\n")