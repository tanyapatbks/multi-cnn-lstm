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

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ForexVisualizer:
    """Enhanced visualization for multi vs single model comparison"""
    
    def __init__(self, config):
        self.config = config
        self.currency_colors = {
            'EURUSD': "#247dbc",  # Blue
            'GBPUSD': '#ff7f0e',  # Orange  
            'USDJPY': '#2ca02c'   # Green
        }
        
        self.strategy_colors = {
            'Multi-Model': '#e74c3c',      # Red
            'Single-Model': '#3498db',     # Blue
            'Buy & Hold': '#f39c12',       # Orange
            'RSI-based': '#9b59b6',        # Purple
            'MACD-based': '#2ecc71'        # Green
        }
        
        # Ensure results directory exists
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)
    
    def plot_comprehensive_comparison(self, all_results, currency_pair, save_path=None):
        """Plot comprehensive comparison of all strategies for a currency pair"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5], width_ratios=[1, 1])
        
        # Main title
        fig.suptitle(f'{currency_pair} Comprehensive Trading Strategy Comparison', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # Prepare data
        strategies = []
        models = []
        confidences = []
        returns = []
        win_rates = []
        sharpe_ratios = []
        max_drawdowns = []
        total_trades = []
        
        # Define colors for different model types
        model_colors = {
            'Multi-Model': '#e74c3c',   # Red
            'Single-Model': '#3498db',  # Blue
            'Buy & Hold': '#f39c12',    # Orange
            'RSI': '#9b59b6',           # Purple
            'MACD': '#2ecc71'           # Green
        }
        
        # Define patterns for confidence levels
        confidence_patterns = {
            'Conservative': {'hatch': '///', 'alpha': 0.9},
            'Moderate': {'hatch': '...', 'alpha': 0.7},
            'Aggressive': {'hatch': '|||', 'alpha': 0.5}
        }
        
        # Extract data from results
        for strategy_name, result in all_results.items():
            if isinstance(result, dict):
                perf = result.get('performance', result)
                
                # Determine model type and confidence level
                if 'Multi-Model' in strategy_name:
                    model = 'Multi-Model'
                    if 'Conservative' in strategy_name:
                        confidence = 'Conservative'
                    elif 'Moderate' in strategy_name:
                        confidence = 'Moderate'
                    elif 'Aggressive' in strategy_name:
                        confidence = 'Aggressive'
                    else:
                        confidence = 'N/A'
                elif currency_pair in strategy_name:
                    model = 'Single-Model'
                    if 'Conservative' in strategy_name:
                        confidence = 'Conservative'
                    elif 'Moderate' in strategy_name:
                        confidence = 'Moderate'
                    elif 'Aggressive' in strategy_name:
                        confidence = 'Aggressive'
                    else:
                        confidence = 'N/A'
                elif 'Buy & Hold' in strategy_name:
                    model = 'Buy & Hold'
                    confidence = 'N/A'
                elif 'RSI' in strategy_name:
                    model = 'RSI'
                    confidence = 'N/A'
                elif 'MACD' in strategy_name:
                    model = 'MACD'
                    confidence = 'N/A'
                else:
                    continue
                
                strategies.append(strategy_name)
                models.append(model)
                confidences.append(confidence)
                returns.append(perf.get('total_return_pct', 0))
                win_rates.append(perf.get('win_rate', 0))
                sharpe_ratios.append(perf.get('sharpe_ratio', 0))
                max_drawdowns.append(perf.get('max_drawdown_pct', 0))
                total_trades.append(perf.get('total_trades', 0))
        
        # Plot 1: Returns Comparison (Main plot)
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(strategies))
        bars = []
        
        for i, (strategy, model, confidence, ret) in enumerate(zip(strategies, models, confidences, returns)):
            color = model_colors.get(model, '#95a5a6')
            
            if confidence != 'N/A' and confidence in confidence_patterns:
                pattern = confidence_patterns[confidence]
                bar = ax1.bar(i, ret, color=color, alpha=pattern['alpha'], 
                             edgecolor='black', linewidth=2, hatch=pattern['hatch'])
            else:
                bar = ax1.bar(i, ret, color=color, alpha=0.8, 
                             edgecolor='black', linewidth=2)
            bars.append(bar)
            
            # Add value labels
            label_y = ret + 1 if ret > 0 else ret - 1
            ax1.text(i, label_y, f'{ret:.1f}%', ha='center', va='bottom' if ret > 0 else 'top',
                    fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Trading Strategy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Return (%)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Total Returns by Strategy - {currency_pair}', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.replace(currency_pair, '').strip() for s in strategies], 
                           rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add legend for models
        model_handles = []
        for model, color in model_colors.items():
            model_handles.append(plt.Rectangle((0,0),1,1, fc=color, alpha=0.8, edgecolor='black'))
        ax1.legend(model_handles, model_colors.keys(), loc='upper left', title='Model Type')
        
        # Add confidence level legend
        conf_handles = []
        for conf, pattern in confidence_patterns.items():
            conf_handles.append(plt.Rectangle((0,0),1,1, fc='gray', alpha=pattern['alpha'], 
                                            hatch=pattern['hatch'], edgecolor='black'))
        ax1_twin = ax1.twinx()
        ax1_twin.set_yticks([])
        ax1_twin.legend(conf_handles, confidence_patterns.keys(), loc='upper right', title='Confidence Level')
        
        # Plot 2: Win Rate
        ax2 = fig.add_subplot(gs[1, 0])
        colors_list = [model_colors.get(m, '#95a5a6') for m in models]
        bars2 = ax2.bar(x, win_rates, color=colors_list, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Win Rates by Strategy', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.replace(currency_pair, '').strip() for s in strategies], 
                           rotation=45, ha='right', fontsize=9)
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars2, win_rates):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', 
                           va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Plot 4: Number of Trades
        bars4 = axes[1, 1].bar(strategies, total_trades, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Total Trades', fontweight='bold')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, total_trades):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(value)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Model comparison saved to {save_path}")
        else:
            filename = f"{currency_pair}_model_comparison.png"
            plt.savefig(f"{self.config.RESULTS_PATH}{filename}", dpi=300, bbox_inches='tight')
            print(f"📊 Model comparison saved to {self.config.RESULTS_PATH}{filename}")
        
        plt.show()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Sharpe Ratio
        ax3 = fig.add_subplot(gs[1, 1])
        bars3 = ax3.bar(x, sharpe_ratios, color=colors_list, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax3.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.replace(currency_pair, '').strip() for s in strategies], 
                           rotation=45, ha='right', fontsize=9)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax3.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good (>1.0)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars3, sharpe_ratios):
            height = bar.get_height()
            label_y = height + 0.05 if height > 0 else height - 0.05
            ax3.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # Plot 4: Maximum Drawdown
        ax4 = fig.add_subplot(gs[2, 0])
        bars4 = ax4.bar(x, max_drawdowns, color=colors_list, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Maximum Drawdown (Risk)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([s.replace(currency_pair, '').strip() for s in strategies], 
                           rotation=45, ha='right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars4, max_drawdowns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 5: Number of Trades
        ax5 = fig.add_subplot(gs[2, 1])
        bars5 = ax5.bar(x, total_trades, color=colors_list, alpha=0.7, edgecolor='black')
        ax5.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
        ax5.set_title('Trading Activity', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([s.replace(currency_pair, '').strip() for s in strategies], 
                           rotation=45, ha='right', fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars5, total_trades):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(value)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            filename = f"{currency_pair}_comprehensive_comparison.png"
            plt.savefig(f"{self.config.RESULTS_PATH}{filename}", dpi=300, bbox_inches='tight')
            print(f"📊 Comprehensive comparison saved to {self.config.RESULTS_PATH}{filename}")
        
        plt.show()
        
        # Create summary statistics table
        self._create_summary_table(strategies, models, confidences, returns, win_rates, 
                                  sharpe_ratios, max_drawdowns, total_trades, currency_pair)
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', 
                           va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Plot 4: Number of Trades
        bars4 = axes[1, 1].bar(strategies, total_trades, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Total Trades', fontweight='bold')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, total_trades):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(value)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Model comparison saved to {save_path}")
        else:
            filename = f"{currency_pair}_model_comparison.png"
            plt.savefig(f"{self.config.RESULTS_PATH}{filename}", dpi=300, bbox_inches='tight')
            print(f"📊 Model comparison saved to {self.config.RESULTS_PATH}{filename}")
        
        plt.show()
    
    def plot_cumulative_returns_comparison(self, comparison_results, currency_pair, save_path=None):
        """Plot cumulative returns over time for different strategies"""
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(f'{currency_pair} Cumulative Returns Comparison', fontsize=16, fontweight='bold')
        
        # Plot each strategy's capital history
        for strategy_name, result in comparison_results.items():
            if 'portfolio_manager' in result:
                portfolio = result['portfolio_manager']
                capital_history = portfolio.capital_history
                
                # Create time series (assuming daily trading)
                returns = [(capital_history[i] - portfolio.initial_capital) / portfolio.initial_capital * 100 
                          for i in range(len(capital_history))]
                
                # Determine line style and color
                if 'Multi' in strategy_name:
                    color = self.strategy_colors['Multi-Model']
                    linestyle = '-'
                    linewidth = 2.5
                elif 'Single' in strategy_name or currency_pair in strategy_name:
                    color = self.strategy_colors['Single-Model']
                    linestyle = '-'
                    linewidth = 2.5
                elif 'Buy' in strategy_name:
                    color = self.strategy_colors['Buy & Hold']
                    linestyle = '--'
                    linewidth = 2
                elif 'RSI' in strategy_name:
                    color = self.strategy_colors['RSI-based']
                    linestyle = '-.'
                    linewidth = 2
                elif 'MACD' in strategy_name:
                    color = self.strategy_colors['MACD-based']
                    linestyle = ':'
                    linewidth = 2
                else:
                    color = '#95a5a6'
                    linestyle = '-'
                    linewidth = 1.5
                
                ax.plot(range(len(returns)), returns, 
                       label=strategy_name, 
                       color=color, 
                       linestyle=linestyle,
                       linewidth=linewidth,
                       alpha=0.8)
        
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        # Add performance zones
        ax.axhspan(0, 100, alpha=0.1, color='green', label='Profit Zone')
        ax.axhspan(-100, 0, alpha=0.1, color='red', label='Loss Zone')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            filename = f"{currency_pair}_cumulative_returns.png"
            plt.savefig(f"{self.config.RESULTS_PATH}{filename}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_performance_summary_table(self, all_results, save_path=None):
        """Create comprehensive performance summary tables"""
        # Prepare data for each currency pair
        currency_tables = {}
        
        for currency in self.config.CURRENCY_PAIRS:
            if currency in all_results:
                table_data = []
                currency_results = all_results[currency]
                
                for strategy_name, result in currency_results.items():
                    if isinstance(result, dict):
                        if 'performance' in result:
                            perf = result['performance']
                            row = {
                                'Strategy': strategy_name,
                                'Total Return (%)': f"{perf.get('total_return_pct', 0):.2f}",
                                'Win Rate': f"{perf.get('win_rate', 0):.3f}",
                                'Total Trades': perf.get('total_trades', 0),
                                'Sharpe Ratio': f"{perf.get('sharpe_ratio', 0):.2f}",
                                'Max Drawdown (%)': f"{perf.get('max_drawdown_pct', 0):.2f}"
                            }
                        else:
                            row = {
                                'Strategy': result.get('strategy_name', strategy_name),
                                'Total Return (%)': f"{result.get('total_return_pct', 0):.2f}",
                                'Win Rate': f"{result.get('win_rate', 0):.3f}",
                                'Total Trades': result.get('total_trades', 0),
                                'Sharpe Ratio': f"{result.get('sharpe_ratio', 0):.2f}",
                                'Max Drawdown (%)': f"{result.get('max_drawdown_pct', 0):.2f}"
                            }
                        table_data.append(row)
                
                currency_tables[currency] = pd.DataFrame(table_data)
        
        # Create figure with subplots for each currency
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Performance Summary Tables by Currency Pair', fontsize=16, fontweight='bold')
        
        for idx, (currency, df) in enumerate(currency_tables.items()):
            ax = axes[idx]
            ax.axis('tight')
            ax.axis('off')
            
            # Create table
            table = ax.table(cellText=df.values,
                           colLabels=df.columns,
                           cellLoc='center',
                           loc='center')
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8)
            
            # Color header
            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_facecolor('#3498db')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if j == 0:  # Strategy column
                        cell.set_facecolor('#ecf0f1')
                    cell.set_text_props(ha='center')
            
            ax.set_title(f'{currency} Performance Metrics', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.config.RESULTS_PATH}performance_summary_table.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Also save as Excel file for easy access
        excel_path = save_path.replace('.png', '.xlsx') if save_path else f"{self.config.RESULTS_PATH}performance_summary.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            for currency, df in currency_tables.items():
                df.to_excel(writer, sheet_name=currency, index=False)
        print(f"📊 Performance summary Excel saved to {excel_path}")
    
    def plot_training_curves(self, history, save_path=None):
        """Plot training and validation curves with best epoch marker"""
        if not history or 'loss' not in history:
            print("❌ No training history available for plotting")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        model_type = self.config.MODEL_TYPE.upper()
        fig.suptitle(f'{model_type} Model Training Performance', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Plot 1: Loss curves
        axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
        axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
        
        # Find and mark best epoch (lowest validation loss)
        best_epoch = np.argmin(history['val_loss']) + 1
        best_val_loss = min(history['val_loss'])
        
        axes[0].scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5, 
                       marker='*', label=f'Best Epoch: {best_epoch}')
        axes[0].annotate(f'Best: {best_val_loss:.4f}', 
                        xy=(best_epoch, best_val_loss),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        axes[0].set_title('Model Loss During Training')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        if 'accuracy' in history and 'val_accuracy' in history:
            axes[1].plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Training Accuracy', alpha=0.8)
            axes[1].plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy', alpha=0.8)
            
            # Mark best validation accuracy
            best_val_acc_epoch = np.argmax(history['val_accuracy']) + 1
            best_val_acc = max(history['val_accuracy'])
            
            axes[1].scatter(best_val_acc_epoch, best_val_acc, color='red', s=100, zorder=5,
                           marker='*', label=f'Best Val Acc: {best_val_acc:.4f}')
            axes[1].annotate(f'Best: {best_val_acc:.4f}', 
                            xy=(best_val_acc_epoch, best_val_acc),
                            xytext=(10, -10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            axes[1].set_title('Model Accuracy During Training')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Add overfitting indicator
            final_gap = history['accuracy'][-1] - history['val_accuracy'][-1]
            gap_color = 'red' if final_gap > 0.1 else 'orange' if final_gap > 0.05 else 'green'
            axes[1].text(0.02, 0.98, f'Overfitting Gap: {final_gap:.3f}', 
                        transform=axes[1].transAxes, fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=gap_color, alpha=0.3),
                        verticalalignment='top')
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training curves saved to {save_path}")
        else:
            plt.savefig(f"{self.config.RESULTS_PATH}training_curves.png", dpi=300, bbox_inches='tight')
            print(f"📊 Training curves saved to {self.config.RESULTS_PATH}training_curves.png")
        
        plt.show()
    
    def save_performance_summary(self, eval_metrics, trading_results, save_path=None):
        """Save a text summary of performance"""
        if save_path is None:
            save_path = f"{self.config.RESULTS_PATH}performance_summary.txt"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"FOREX PREDICTION SYSTEM - PERFORMANCE SUMMARY ({self.config.MODEL_TYPE.upper()})\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: {self.config.MODEL_TYPE}\n")
            f.write(f"Training Period: {self.config.TRAIN_START} to {self.config.TRAIN_END}\n")
            f.write(f"Validation Period: {self.config.VAL_START} to {self.config.VAL_END}\n")
            f.write(f"Test Period: {self.config.TEST_START} to {self.config.TEST_END}\n\n")
            
            # Model Performance
            f.write("MODEL PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {eval_metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {eval_metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall: {eval_metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-Score: {eval_metrics.get('f1_score', 0):.4f}\n\n")
            
            # Trading Performance
            f.write("TRADING STRATEGY PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Strategy':<25} {'Return%':<10} {'Win Rate':<10} {'Sharpe':<10} {'Trades':<8} {'Max DD%':<10}\n")
            f.write("-" * 83 + "\n")
            
            for strategy_name, result in trading_results.items():
                if 'performance' in result:
                    perf = result['performance']
                    f.write(f"{strategy_name:<25} {perf.get('total_return_pct', 0):<10.2f} "
                           f"{perf.get('win_rate', 0):<10.4f} {perf.get('sharpe_ratio', 0):<10.2f} "
                           f"{perf.get('total_trades', 0):<8} {perf.get('max_drawdown_pct', 0):<10.2f}\n")
                else:
                    f.write(f"{result.get('strategy_name', strategy_name):<25} "
                           f"{result.get('total_return_pct', 0):<10.2f} {result.get('win_rate', 0):<10.4f} "
                           f"{result.get('sharpe_ratio', 0):<10.2f} {result.get('total_trades', 0):<8} "
                           f"{result.get('max_drawdown_pct', 0):<10.2f}\n")
        
        print(f"📄 Performance summary saved to {save_path}")

def display_saved_results(config):
    """Display results from saved files"""
    results_manager = None
    try:
        from checkpoint import ResultsManager
        results_manager = ResultsManager(config)
    except ImportError:
        print("❌ Could not import ResultsManager")
        return
    
    # Load saved results
    model_metrics = results_manager.load_results("cnn_lstm_metrics.pkl", verbose=False)
    trading_results = results_manager.load_results("fixed_holding_results.pkl", verbose=False)
    
    if model_metrics and trading_results:
        visualizer = ForexVisualizer(config)
        
        # Load training history if available
        try:
            import pickle
            with open(f"{config.CHECKPOINTS_PATH}checkpoint.pkl", 'rb') as f:
                checkpoint = pickle.load(f)
                history = checkpoint.get('data', {}).get('history', {})
        except:
            history = None
        
        # Create visualizations
        if history:
            visualizer.plot_training_curves(history)
        visualizer.save_performance_summary(model_metrics, trading_results)
    else:
        print("❌ Could not load saved results for visualization")