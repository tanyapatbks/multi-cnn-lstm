import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# กำหนดรายชื่อไฟล์และชื่อสำหรับกราฟ
files_to_plot = {
    "EURUSD": "data/EURUSD_1H.csv",
    "GBPUSD": "data/GBPUSD_1H.csv",
    "USDJPY": "data/USDJPY_1H.csv"
}

# ==================================================================
# <<< ส่วนที่แก้ไข: กำหนดสีสำหรับแต่ละคู่เงิน >>>
# ==================================================================
currency_colors = {
    "EURUSD": "#378fca",  # สีฟ้า (Blue)
    "GBPUSD": "#cf3f2f",  # สีแดง (Red)
    "USDJPY": "#1c894a"   # สีเขียว (Green)
}

# สร้าง Figure และ Subplots ขนาด 3 แถว 1 คอลัมน์
fig, axes = plt.subplots(len(files_to_plot), 1, figsize=(20, 24), sharex=True)
fig.suptitle('Volume Data (2018-2020)', fontsize=22, fontweight='bold')

# วนลูปเพื่อวาดกราฟแต่ละคู่เงิน
for i, (currency, filepath) in enumerate(files_to_plot.items()):
    ax = axes[i]
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError

        # โหลดข้อมูลและแปลงเวลาให้ถูกต้อง
        df = pd.read_csv(filepath)
        df.rename(columns={'Local time': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z', utc=True)
        df = df.set_index('time')
        
        # กรองข้อมูลตามช่วงเวลา
        start_date = '2018-01-01'
        end_date = '2020-12-31'
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

        if df_filtered.empty:
            raise ValueError(f"No data found for the period {start_date} to {end_date}")
        
        volume = df_filtered['Volume'].dropna()
        
        # ==================================================================
        # <<< ส่วนที่แก้ไข: เปลี่ยนสีของกราฟตามค่าที่กำหนดไว้ >>>
        # ==================================================================
        plot_color = currency_colors.get(currency, '#2E86C1') # ใช้สีฟ้าเป็นค่าสำรอง
        ax.plot(volume.index, volume, label=f'Actual Volume ({currency})', color=plot_color, alpha=0.8, linewidth=1.2)
            
        # จัดรูปแบบกราฟ
        ax.set_title(f'{currency} Volume (2018-2020)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Volume')
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # จัดรูปแบบแกน Y ให้อ่านง่าย
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        
    except FileNotFoundError:
        ax.text(0.5, 0.5, f'Error: File not found\n{filepath}', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
        ax.set_title(f'{currency} - File Not Found', fontsize=16)
    except Exception as e:
        ax.text(0.5, 0.5, f'An error occurred:\n{e}', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
        ax.set_title(f'{currency} - Error', fontsize=16)

# จัดรูปแบบโดยรวมและบันทึกไฟล์
plt.xlabel('Date', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
save_path = 'volume_visualization_2018-2020.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"กราฟ '{save_path}' ถูกสร้างขึ้นเรียบร้อยแล้วครับ")