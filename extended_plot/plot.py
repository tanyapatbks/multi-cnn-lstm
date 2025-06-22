import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def create_transformation_plots():
    """
    Creates a 3x3 plot showing the data transformation process for each currency pair:
    1. Original OHLC Prices
    2. Percentage Change
    3. Normalized (Standard Scaler) Percentage Change
    """
    files_to_process = {
        "EURUSD": "data/EURUSD_1H.csv",
        "GBPUSD": "data/GBPUSD_1H.csv",
        "USDJPY": "data/USDJPY_1H.csv"
    }
    
    # ==================================================================
    # <<< ส่วนที่เพิ่มเข้ามา: กำหนดสีสำหรับแต่ละคู่เงิน >>>
    # ==================================================================
    currency_colors = {
        "EURUSD": "#3498db",  # สีฟ้า
        "GBPUSD": "#e74c3c",  # สีแดง
        "USDJPY": "#2ecc71"   # สีเขียว
    }

    # สร้าง Figure และ Subplots ขนาด 3 แถว 3 คอลัมน์
    fig, axes = plt.subplots(len(files_to_process), 3, figsize=(25, 15), sharex=True)
    fig.suptitle('Data Transformation Pipeline (2018-2020)', fontsize=22, fontweight='bold')

    # กำหนดหัวข้อสำหรับแต่ละคอลัมน์
    column_titles = ['1. Original OHLC Prices', '2. Percentage Change (Stationary)', '3. Normalized Data (Z-score)']
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title, fontsize=16, fontweight='bold')

    # วนลูปเพื่อวาดกราฟแต่ละคู่เงิน
    for i, (currency, filepath) in enumerate(files_to_process.items()):
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            # --- 1. โหลดและเตรียมข้อมูล ---
            df = pd.read_csv(filepath)
            df.rename(columns={'Local time': 'time'}, inplace=True)
            df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z', utc=True)
            df = df.set_index('time')
            
            start_date = '2018-01-01'
            end_date = '2020-12-31'
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)].copy()

            if df_filtered.empty:
                raise ValueError("No data in the specified date range.")

            # --- 2. คำนวณ % Change และ Normalized Data ---
            ohlc_cols = ['Open', 'High', 'Low', 'Close']
            return_cols = [f'{col}_Return' for col in ohlc_cols]
            
            for col in ohlc_cols:
                df_filtered[f'{col}_Return'] = df_filtered[col].pct_change()
            
            df_filtered.dropna(inplace=True)
            
            scaler = StandardScaler()
            df_normalized_returns = pd.DataFrame(
                scaler.fit_transform(df_filtered[return_cols]),
                index=df_filtered.index,
                columns=return_cols
            )

            # --- 3. วาดกราฟลงบนแต่ละ Subplot ---
            
            # <<< แก้ไข: ดึงสีที่ถูกต้องมาใช้ >>>
            plot_color = currency_colors.get(currency, 'gray')

            # กราฟที่ 1: Original OHLC
            axes[i, 0].plot(df_filtered['Close'], color=plot_color, label=f'{currency} Close Price') # แสดงแค่ราคา Close เพื่อความชัดเจน
            axes[i, 0].set_ylabel(f'{currency} Price')
            axes[i, 0].legend(loc='upper left')
            axes[i, 0].grid(True, linestyle='--', alpha=0.6)

            # กราฟที่ 2: Percentage Change
            axes[i, 1].plot(df_filtered['Close_Return'], color=plot_color, alpha=0.7)
            axes[i, 1].axhline(0, color='black', linewidth=1, linestyle='--')
            axes[i, 1].set_ylabel('% Change')
            axes[i, 1].grid(True, linestyle='--', alpha=0.6)

            # กราฟที่ 3: Normalized Data
            axes[i, 2].plot(df_normalized_returns['Close_Return'], color=plot_color, alpha=0.7)
            axes[i, 2].axhline(0, color='black', linewidth=1, linestyle='--')
            axes[i, 2].set_ylabel('Standardized Value')
            axes[i, 2].grid(True, linestyle='--', alpha=0.6)

        except Exception as e:
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f'Error processing {currency}:\n{e}', ha='center', va='center', color='red')

    # จัดรูปแบบและบันทึก
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = 'data_transformation_stages.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"กราฟ '{save_path}' ถูกสร้างขึ้นเรียบร้อยแล้วครับ")

if __name__ == '__main__':
    create_transformation_plots()