"""
Visualization Module for Multi-Currency CNN-LSTM Forex Prediction
Enhanced with Multi vs Single Model Comparison
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ForexVisualizer:
    """Enhanced visualization for multi vs single model comparison"""
    
    # <<< แก้ไข __init__ ให้รับ unattended_mode >>>
    def __init__(self, config, unattended_mode=False):
        self.config = config
        self.unattended = unattended_mode # เก็บค่าสถานะ
        self.currency_colors = {'EURUSD': "#247dbc", 'GBPUSD': '#ff7f0e', 'USDJPY': '#2ca02c'}
        self.strategy_colors = {
            'Multi-Model': '#e74c3c', 'Single-Model': '#3498db', 'Buy & Hold': '#f39c12',
            'RSI-based': '#9b59b6', 'MACD-based': '#2ecc71'
        }
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)
    
    # --- ฟังก์ชันอื่นๆ จะถูกแก้ไขให้ใช้ self.unattended ---
    
    def plot_comprehensive_comparison(self, all_results, currency_pair, save_path=None):
        if not all_results:
            print(f"⚠️ No results to plot for {currency_pair} comprehensive comparison.")
            return
            
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5], width_ratios=[1, 1])
        fig.suptitle(f'{currency_pair} Comprehensive Trading Strategy Comparison', fontsize=20, fontweight='bold', y=0.98)
        
        # ... (ส่วนของการเตรียมข้อมูลเหมือนเดิม) ...
        strategies, models, confidences, returns, win_rates, sharpe_ratios, max_drawdowns, total_trades = [], [], [], [], [], [], [], []
        model_colors = {
            'Multi-Model': '#e74c3c', 'Single-Model': '#3498db', 'Buy & Hold': '#f39c12',
            'RSI': '#9b59b6', 'MACD': '#2ecc71'
        }
        confidence_patterns = {
            'Conservative': {'hatch': '///', 'alpha': 0.9}, 'Moderate': {'hatch': '...', 'alpha': 0.7}, 'Aggressive': {'hatch': '|||', 'alpha': 0.5}
        }
        for strategy_name, result in all_results.items():
            if isinstance(result, dict):
                perf = result.get('performance', result)
                model, confidence = 'N/A', 'N/A'
                if 'Multi-Model' in strategy_name: model = 'Multi-Model'
                elif currency_pair in strategy_name: model = 'Single-Model'
                elif 'Buy & Hold' in strategy_name: model = 'Buy & Hold'
                elif 'RSI' in strategy_name: model = 'RSI'
                elif 'MACD' in strategy_name: model = 'MACD'
                else: continue
                if 'Conservative' in strategy_name: confidence = 'Conservative'
                elif 'Moderate' in strategy_name: confidence = 'Moderate'
                elif 'Aggressive' in strategy_name: confidence = 'Aggressive'
                strategies.append(strategy_name); models.append(model); confidences.append(confidence); returns.append(perf.get('total_return_pct', 0))
                win_rates.append(perf.get('win_rate', 0)); sharpe_ratios.append(perf.get('sharpe_ratio', 0)); max_drawdowns.append(perf.get('max_drawdown_pct', 0)); total_trades.append(perf.get('total_trades', 0))
        
        # Plot 1: Returns Comparison
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(strategies))
        for i, (model, confidence, ret) in enumerate(zip(models, confidences, returns)):
            color = model_colors.get(model, '#95a5a6')
            pattern = confidence_patterns.get(confidence)
            if pattern: ax1.bar(i, ret, color=color, alpha=pattern['alpha'], edgecolor='black', linewidth=2, hatch=pattern['hatch'])
            else: ax1.bar(i, ret, color=color, alpha=0.8, edgecolor='black', linewidth=2)
            label_y = ret + (max(returns)*0.05) if ret >= 0 else ret - (max(returns)*0.05)
            ax1.text(i, label_y, f'{ret:.1f}%', ha='center', va='bottom' if ret >= 0 else 'top', fontsize=10, fontweight='bold')
        ax1.set_title(f'Total Returns by Strategy - {currency_pair}', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Total Return (%)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x); ax1.set_xticklabels([s.replace(currency_pair, '').replace("CNN-LSTM", "").strip() for s in strategies], rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1); ax1.grid(True, alpha=0.3, axis='y')
        model_handles = [plt.Rectangle((0,0),1,1, fc=color, alpha=0.8, edgecolor='black') for model, color in model_colors.items()]
        ax1.legend(model_handles, model_colors.keys(), loc='upper left', title='Model Type')
        conf_handles = [plt.Rectangle((0,0),1,1, fc='gray', alpha=p['alpha'], hatch=p['hatch'], edgecolor='black') for c, p in confidence_patterns.items()]
        ax1_twin = ax1.twinx(); ax1_twin.set_yticks([]); ax1_twin.legend(conf_handles, confidence_patterns.keys(), loc='upper right', title='Confidence Level')

        colors_list = [model_colors.get(m, '#95a5a6') for m in models]
        xticklabels = [s.replace(currency_pair, '').replace("CNN-LSTM", "").strip() for s in strategies]
        
        # Plot 2: Win Rate
        ax2 = fig.add_subplot(gs[1, 0]); bars2 = ax2.bar(x, win_rates, color=colors_list, alpha=0.7, edgecolor='black')
        ax2.set_title('Win Rates', fontsize=14, fontweight='bold'); ax2.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
        ax2.set_xticks(x); ax2.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=9); ax2.set_ylim(0, max(win_rates) * 1.2 if win_rates else 1)
        ax2.grid(True, alpha=0.3, axis='y');
        for bar, value in zip(bars2, win_rates): ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Sharpe Ratio
        ax3 = fig.add_subplot(gs[1, 1]); bars3 = ax3.bar(x, sharpe_ratios, color=colors_list, alpha=0.7, edgecolor='black')
        ax3.set_title('Sharpe Ratio', fontsize=14, fontweight='bold'); ax3.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax3.set_xticks(x); ax3.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=9); ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax3.grid(True, alpha=0.3, axis='y');
        for bar, value in zip(bars3, sharpe_ratios): ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05 if value >= 0 else bar.get_height() - 0.2, f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
        
        # Plot 4: Maximum Drawdown
        ax4 = fig.add_subplot(gs[2, 0]); bars4 = ax4.bar(x, max_drawdowns, color=colors_list, alpha=0.7, edgecolor='black')
        ax4.set_title('Maximum Drawdown (Risk)', fontsize=14, fontweight='bold'); ax4.set_ylabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
        ax4.set_xticks(x); ax4.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=9); ax4.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars4, max_drawdowns): ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

        # Plot 5: Number of Trades
        ax5 = fig.add_subplot(gs[2, 1]); bars5 = ax5.bar(x, total_trades, color=colors_list, alpha=0.7, edgecolor='black')
        ax5.set_title('Trading Activity', fontsize=14, fontweight='bold'); ax5.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
        ax5.set_xticks(x); ax5.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=9); ax5.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars5, total_trades): ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(total_trades)*0.01, f'{int(value)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # ... (ส่วนของการบันทึกไฟล์เหมือนเดิม) ...
        filename = f"{currency_pair}_model_comparison.png"
        full_path = os.path.join(self.config.RESULTS_PATH, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"📊 Model comparison saved to {full_path}")
        
        # <<< แก้ไขส่วนนี้: ตรวจสอบโหมดก่อนแสดงกราฟ >>>
        if not self.unattended:
            plt.show()
        plt.close(fig) # คืนหน่วยความจำ

    def plot_cumulative_returns_comparison(self, comparison_results, currency_pair, save_path=None):
        if not comparison_results:
            print(f"⚠️ No results to plot for {currency_pair} cumulative returns.")
            return

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(f'{currency_pair} Cumulative Returns Comparison', fontsize=16, fontweight='bold')
        
        # ... (Code for plotting is unchanged) ...
        for strategy_name, result in comparison_results.items():
            if 'portfolio_manager' in result:
                portfolio = result['portfolio_manager']
                capital_history = portfolio.capital_history
                returns = [(cap - portfolio.initial_capital) / portfolio.initial_capital * 100 for cap in capital_history]
                if 'Multi' in strategy_name: color, style, lw = self.strategy_colors['Multi-Model'], '-', 2.5
                elif currency_pair in strategy_name: color, style, lw = self.strategy_colors['Single-Model'], '-', 2.5
                elif 'Buy' in strategy_name: color, style, lw = self.strategy_colors['Buy & Hold'], '--', 2
                elif 'RSI' in strategy_name: color, style, lw = self.strategy_colors['RSI-based'], '-.', 2
                elif 'MACD' in strategy_name: color, style, lw = self.strategy_colors['MACD-based'], ':', 2
                else: color, style, lw = '#95a5a6', '-', 1.5
                ax.plot(range(len(returns)), returns, label=strategy_name, color=color, linestyle=style, linewidth=lw, alpha=0.8)

        ax.set_xlabel('Trading Periods'); ax.set_ylabel('Cumulative Return (%)'); ax.legend(loc='best'); ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filename = f"{currency_pair}_cumulative_returns.png"
        full_path = os.path.join(self.config.RESULTS_PATH, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"📊 Cumulative returns saved to {full_path}")
        
        # <<< แก้ไขส่วนนี้ >>>
        if not self.unattended:
            plt.show()
        plt.close(fig)

    def create_performance_summary_table(self, all_results, save_path=None):
        # ... (This function is mostly unchanged but will benefit from the plt.close(fig) addition)
        currency_tables = {}
        for currency, strategies in all_results.items():
            table_data = []
            for strategy_name, result in strategies.items():
                if isinstance(result, dict) and 'performance' in result:
                    perf = result['performance']
                    table_data.append({'Strategy': strategy_name, 'Total Return (%)': f"{perf.get('total_return_pct', 0):.2f}", 'Win Rate': f"{perf.get('win_rate', 0):.3f}", 'Total Trades': perf.get('total_trades', 0), 'Sharpe Ratio': f"{perf.get('sharpe_ratio', 0):.2f}", 'Max Drawdown (%)': f"{perf.get('max_drawdown_pct', 0):.2f}"})
            if table_data: currency_tables[currency] = pd.DataFrame(table_data)
        if not currency_tables: print("No data for performance summary table."); return

        fig, axes = plt.subplots(len(currency_tables), 1, figsize=(12, 3 * len(currency_tables)), squeeze=False)
        fig.suptitle('Performance Summary Tables by Currency Pair', fontsize=16, fontweight='bold', y=0.98)
        
        for idx, (currency, df) in enumerate(currency_tables.items()):
            ax = axes[idx, 0]
            ax.axis('tight'); ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.8)
            for (i, j), cell in table.get_celld().items():
                if i == 0: cell.set_facecolor('#3498db'), cell.set_text_props(weight='bold', color='white')
                else: cell.set_facecolor('#ecf0f1' if j == 0 else 'white')
            ax.set_title(f'{currency} Performance Metrics', fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4)
        summary_path = save_path or os.path.join(self.config.RESULTS_PATH, "performance_summary_table.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance summary table saved to {summary_path}")
        
        # <<< แก้ไขส่วนนี้ >>>
        if not self.unattended:
            plt.show()
        plt.close(fig)
    
    def plot_training_curves(self, history, save_path=None):
        if not history or 'loss' not in history: print("❌ No training history available."); return
        
        # ... (This function is also mostly unchanged but benefits from plt.close(fig))
        has_accuracy = 'accuracy' in history and 'val_accuracy' in history
        fig, axes = plt.subplots(1, 2 if has_accuracy else 1, figsize=(15, 6), squeeze=False)
        fig.suptitle(f'{self.config.MODEL_TYPE.upper()} Model Training Performance', fontsize=16, fontweight='bold')
        epochs = range(1, len(history['loss']) + 1)
        
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss'); axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        best_epoch = np.argmin(history['val_loss']) + 1; best_val_loss = min(history['val_loss'])
        axes[0, 0].scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5, marker='*', label=f'Best Epoch: {best_epoch}')
        axes[0, 0].set_title('Model Loss'); axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        if has_accuracy:
            axes[0, 1].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy'); axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
            best_acc_epoch = np.argmax(history['val_accuracy']) + 1; best_val_acc = max(history['val_accuracy'])
            axes[0, 1].scatter(best_acc_epoch, best_val_acc, color='green', s=100, zorder=5, marker='*', label=f'Best Epoch: {best_acc_epoch}')
            axes[0, 1].set_title('Model Accuracy'); axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        curves_path = save_path or os.path.join(self.config.RESULTS_PATH, "training_curves.png")
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {curves_path}")
        
        # <<< แก้ไขส่วนนี้ >>>
        if not self.unattended:
            plt.show()
        plt.close(fig)

    def save_performance_summary(self, eval_metrics, trading_results, save_path=None):
        # This function does not show plots, no changes needed here.
        summary_path = save_path or os.path.join(self.config.RESULTS_PATH, "performance_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + f"\nFOREX PREDICTION SYSTEM - PERFORMANCE SUMMARY ({self.config.MODEL_TYPE.upper()})\n" + "="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nModel Type: {self.config.MODEL_TYPE}\n\n")
            f.write("MODEL PERFORMANCE:\n" + "-"*40 + "\n")
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)): f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
            f.write("\nTRADING STRATEGY PERFORMANCE:\n" + "-"*40 + "\n")
            header = f"{'Strategy':<45} {'Return%':<10} {'Win Rate':<10} {'Sharpe':<10} {'Trades':<8} {'Max DD%':<10}\n"
            f.write(header + "-" * (len(header)+5) + "\n")
            sorted_results = sorted(trading_results.items(), key=lambda item: item[1].get('performance', {}).get('total_return_pct', -9e9), reverse=True)
            for name, result in sorted_results:
                perf = result.get('performance', result)
                f.write(f"{name:<45} {perf.get('total_return_pct', 0):<10.2f} "
                       f"{perf.get('win_rate', 0):<10.4f} {perf.get('sharpe_ratio', 0):<10.2f} "
                       f"{perf.get('total_trades', 0):<8} {perf.get('max_drawdown_pct', 0):<10.2f}\n")
        print(f"📄 Performance summary saved to {summary_path}")