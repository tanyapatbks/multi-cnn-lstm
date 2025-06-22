import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# --- การตั้งค่าหลัก ---
# 1. กำหนดรายชื่อไฟล์
files_to_plot = {
    "EURUSD": "data/EURUSD_1H.csv",
    "GBPUSD": "data/GBPUSD_1H.csv",
    "USDJPY": "data/USDJPY_1H.csv"
}

# 2. กำหนดสีสำหรับแต่ละคู่เงิน
currency_colors = {
    "EURUSD": "#3498db",  # สีฟ้า
    "GBPUSD": "#e74c3c",  # สีแดง
    "USDJPY": "#2ecc71"   # สีเขียว
}

# 3. กำหนดช่วงเวลา
start_date = '2018-01-01'
end_date = '2020-12-31'

# --- สร้าง Figure สำหรับกราฟทั้ง 4 ชุด ---
fig1, axes1 = plt.subplots(len(files_to_plot), 1, figsize=(20, 18), sharex=True)
fig1.suptitle(f'Raw Volume Data ({start_date} to {end_date})', fontsize=22, fontweight='bold')

fig2, axes2 = plt.subplots(len(files_to_plot), 1, figsize=(20, 18), sharex=True)
fig2.suptitle('Volume with 7-SD Threshold', fontsize=22, fontweight='bold')

fig3, axes3 = plt.subplots(len(files_to_plot), 1, figsize=(20, 18), sharex=True)
fig3.suptitle('Volume after Capping at 7-SD', fontsize=22, fontweight='bold')

fig4, axes4 = plt.subplots(len(files_to_plot), 1, figsize=(20, 18), sharex=True)
fig4.suptitle('Volume after Capping and Min-Max Scaling', fontsize=22, fontweight='bold')


# --- วนลูปเพื่อโหลดข้อมูลและวาดกราฟ ---
for i, (currency, filepath) in enumerate(files_to_plot.items()):
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # โหลดและเตรียมข้อมูล
        df = pd.read_csv(filepath)
        df.rename(columns={'Local time': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z', utc=True)
        df = df.set_index('time')
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

        if df_filtered.empty:
            raise ValueError("No data in the specified date range.")
            
        volume = df_filtered['Volume'].dropna()
        plot_color = currency_colors.get(currency, 'gray')

        # คำนวณค่าสถิติและค่าต่างๆ
        mean_volume = volume.mean()
        std_volume = volume.std()
        threshold_7sd = mean_volume + (7 * std_volume)
        volume_capped = volume.clip(upper=threshold_7sd)
        
        scaler = MinMaxScaler()
        volume_normalized = scaler.fit_transform(volume_capped.values.reshape(-1, 1)).flatten()

        # --- วาดกราฟลงบนแต่ละ Figure ---
        
        # กราฟ 1: Raw Volume
        axes1[i].plot(volume.index, volume, color=plot_color, alpha=0.8, linewidth=1.2)
        axes1[i].set_title(f'{currency} Raw Volume', fontsize=16)
        axes1[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

        # กราฟ 2: Raw Volume with 7-SD Line
        axes2[i].plot(volume.index, volume, color=plot_color, alpha=0.8, linewidth=1.2)
        axes2[i].axhline(y=threshold_7sd, color='red', linestyle='--', linewidth=2, label=f'7-SD Threshold ({threshold_7sd:,.0f})')
        axes2[i].set_title(f'{currency} Volume with 7-SD Threshold', fontsize=16)
        axes2[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        axes2[i].legend()

        # กราฟ 3: Capped Volume
        axes3[i].plot(volume_capped.index, volume_capped, color=plot_color, alpha=0.8, linewidth=1.2)
        axes3[i].axhline(y=threshold_7sd, color='red', linestyle='--', linewidth=2, label=f'Capping Level ({threshold_7sd:,.0f})')
        axes3[i].set_title(f'{currency} Volume after Capping at 7-SD', fontsize=16)
        axes3[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        axes3[i].legend()

        # กราฟ 4: Normalized Capped Volume
        axes4[i].plot(volume_capped.index, volume_normalized, color=plot_color, alpha=0.8, linewidth=1.2)
        axes4[i].set_title(f'{currency} Volume (Capped & Normalized)', fontsize=16)
        axes4[i].set_ylim(-0.05, 1.05) # กำหนดแกน Y ให้อยู่ในช่วง 0-1
        
    except Exception as e:
        for ax_set in [axes1, axes2, axes3, axes4]:
            ax_set[i].text(0.5, 0.5, f'Error processing {currency}:\n{e}', ha='center', va='center', color='red')
            ax_set[i].set_title(f'{currency} - Error', fontsize=16)


# --- จัดรูปแบบและบันทึกไฟล์ทั้งหมด ---
for fig_obj, filename in [
    (fig1, '1_raw_volume.png'),
    (fig2, '2_volume_with_sd_line.png'),
    (fig3, '3_capped_volume.png'),
    (fig4, '4_normalized_capped_volume.png')
]:
    fig_obj.supxlabel('Date', fontsize=14)
    fig_obj.tight_layout(rect=[0, 0, 1, 0.97])
    fig_obj.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"กราฟ '{filename}' ถูกสร้างขึ้นเรียบร้อยแล้วครับ")