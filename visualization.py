import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict

plt.style.use('seaborn-v0_8-darkgrid')

class ForexVisualizer:
    def __init__(self, config, unattended_mode=False):
        self.config = config
        self.unattended = unattended_mode
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)

    def plot_training_curves(self, history):
        """Plots training curves and saves them to the model's specific result folder."""
        try:
            if not history or 'loss' not in history.history:
                print("⚠️ No training history available for plotting.")
                return
            
            hist_dict = history.history
            fig, ax = plt.subplots(1, 2, figsize=(15, 5), squeeze=True)
            fig.suptitle(f'Model Training Performance: {self.config.MODEL_TYPE} (Target: {self.config.TARGET_PAIR})', 
                        fontsize=16, fontweight='bold')
            epochs = range(1, len(hist_dict['loss']) + 1)
            
            # Loss plot
            ax[0].plot(epochs, hist_dict['loss'], 'b-', label='Training Loss', linewidth=2)
            ax[0].plot(epochs, hist_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax[0].set_title('Model Loss', fontsize=14)
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            
            # Accuracy plot
            if 'accuracy' in hist_dict:
                ax[1].plot(epochs, hist_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
                ax[1].plot(epochs, hist_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
                ax[1].set_title('Model Accuracy', fontsize=14)
                ax[1].legend()
                ax[1].grid(True, alpha=0.3)
                ax[1].set_xlabel('Epoch')
                ax[1].set_ylabel('Accuracy')
            else:
                ax[1].text(0.5, 0.5, 'No Accuracy Data Available', 
                          ha='center', va='center', transform=ax[1].transAxes, fontsize=12)
                ax[1].set_title('Model Accuracy', fontsize=14)
            
            save_path = os.path.join(self.config.RESULTS_PATH, "training_curves.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Training curves saved to {save_path}")
            
            if not self.unattended:
                plt.show()
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ Error plotting training curves: {e}")

    def create_loop_summary_csv(self, loop_results):
        print("💾 Generating enhanced summary CSV for the loop...")
        df_data = []
        for name, result in loop_results.items():
            perf = result 
            df_data.append({
                'Strategy': name,
                'Total Return (%)': perf.get('total_return_pct', 0),
                'Sharpe Ratio': perf.get('sharpe_ratio', 0),
                'Win Rate': perf.get('win_rate', 0),
                'Max Drawdown (%)': perf.get('max_drawdown_pct', 0),
                'Total Trades': perf.get('total_trades', 0),
            })
        
        if not df_data:
            print("⚠️ No data to create summary CSV.")
            return

        df = pd.DataFrame(df_data).sort_values(by="Total Return (%)", ascending=False)
        save_path = os.path.join(self.config.RESULTS_PATH, "enhanced_loop_summary.csv")
        df.to_csv(save_path, index=False, float_format='%.4f')
        print(f"✅ Enhanced Loop Summary CSV saved to {save_path}")
        print(f"📈 Total strategies evaluated: {len(df)}")
        
        # Generate and save detailed analysis
        try:
            analyzer = PerformanceAnalyzer(loop_results)
            report_text = analyzer.generate_performance_report()
            report_path = os.path.join(self.config.RESULTS_PATH, "detailed_analysis_report.txt")
            with open(report_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Detailed analysis saved to {report_path}")
        except Exception as e:
            print(f"⚠️ Could not generate detailed analysis: {e}")

    def plot_loop_comparison_graph(self, all_results, currency_pair):
        try:
            # Filter results for the specific currency pair
            pair_results = {k: v for k, v in all_results.items() if f'({currency_pair})' in k}
            
            if not pair_results:
                print(f"⚠️ No results found for {currency_pair}")
                return
            
            # Create comparison plot
            strategies = list(pair_results.keys())
            returns = [pair_results[s].get('total_return_pct', 0) for s in strategies]
            sharpe_ratios = [pair_results[s].get('sharpe_ratio', 0) for s in strategies]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Returns comparison
            bars1 = ax1.bar(range(len(strategies)), returns, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
            ax1.set_title(f'{currency_pair} - Total Returns Comparison', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Total Return (%)')
            ax1.set_xticks(range(len(strategies)))
            ax1.set_xticklabels([s.replace(f' ({currency_pair})', '') for s in strategies], rotation=45, ha='right')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, returns):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.3,
                        f'{value:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
            
            # Sharpe ratio comparison
            bars2 = ax2.bar(range(len(strategies)), sharpe_ratios, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
            ax2.set_title(f'{currency_pair} - Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_xticks(range(len(strategies)))
            ax2.set_xticklabels([s.replace(f' ({currency_pair})', '') for s in strategies], rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars2, sharpe_ratios):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height >= 0 else height - 0.05,
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.tight_layout()
            save_path = os.path.join(self.config.RESULTS_PATH, f"{currency_pair}_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 {currency_pair} comparison saved to {save_path}")
            
            if not self.unattended:
                plt.show()
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ Error creating comparison graph for {currency_pair}: {e}")

    def plot_threshold_comparison_summary(self, all_results):
        try:
            print("📊 Creating threshold comparison summary...")
            
            # Group results by threshold type
            threshold_data = {'Conservative': [], 'Moderate': [], 'Aggressive': []}
            
            for strategy_name, performance in all_results.items():
                if 'CNN-LSTM' in strategy_name:
                    if 'Conservative' in strategy_name:
                        threshold_data['Conservative'].append(performance.get('total_return_pct', 0))
                    elif 'Moderate' in strategy_name:
                        threshold_data['Moderate'].append(performance.get('total_return_pct', 0))
                    elif 'Aggressive' in strategy_name:
                        threshold_data['Aggressive'].append(performance.get('total_return_pct', 0))
            
            # Calculate averages
            avg_returns = {}
            for threshold, returns in threshold_data.items():
                avg_returns[threshold] = np.mean(returns) if returns else 0
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            thresholds = list(avg_returns.keys())
            returns = list(avg_returns.values())
            colors = ['red', 'blue', 'green']
            
            bars = ax.bar(thresholds, returns, color=colors, alpha=0.7)
            ax.set_title('Average Returns by Threshold Strategy', fontsize=16, fontweight='bold')
            ax.set_ylabel('Average Total Return (%)')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, returns):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.3,
                       f'{value:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
            
            save_path = os.path.join(self.config.RESULTS_PATH, "threshold_comparison_summary.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Threshold comparison saved to {save_path}")
            
            if not self.unattended:
                plt.show()
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ Error creating threshold comparison: {e}")


class PerformanceAnalyzer:
    def __init__(self, results_dict: Dict):
        """Initialize with results dictionary from main_fx.py"""
        self.results = results_dict
        try:
            self.df = self._create_results_dataframe()
        except Exception as e:
            print(f"⚠️ Could not create results dataframe: {e}")
            self.df = pd.DataFrame()
        
    def _create_results_dataframe(self) -> pd.DataFrame:
        data = []
        for strategy_name, metrics in self.results.items():
            # Parse strategy information
            strategy_info = self._parse_strategy_name(strategy_name)
            row = {
                'Full_Name': strategy_name,
                'Strategy_Type': strategy_info['type'],
                'Threshold': strategy_info['threshold'],
                'Currency': strategy_info['currency'],
                'Model_Family': strategy_info['family'],
                **metrics  # Unpack all performance metrics
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _parse_strategy_name(self, name: str) -> Dict:
        info = {
            'type': 'Unknown',
            'threshold': 'None',
            'currency': 'Unknown',
            'family': 'Unknown'
        }
        
        try:
            if 'CNN-LSTM' in name:
                info['family'] = 'CNN-LSTM'
                if 'Multi' in name:
                    info['type'] = 'Multi-Currency'
                else:
                    info['type'] = 'Single-Currency'
                
                # Extract threshold
                if 'Conservative' in name:
                    info['threshold'] = 'Conservative'
                elif 'Moderate' in name:
                    info['threshold'] = 'Moderate'
                elif 'Aggressive' in name:
                    info['threshold'] = 'Aggressive'
                
            elif 'Buy & Hold' in name:
                info['family'] = 'Buy & Hold'
                info['type'] = 'Baseline'
            elif 'RSI' in name:
                info['family'] = 'RSI'
                info['type'] = 'Technical'
            elif 'MACD' in name:
                info['family'] = 'MACD'
                info['type'] = 'Technical'
            
            # Extract currency
            for currency in ['EURUSD', 'GBPUSD', 'USDJPY']:
                if currency in name:
                    info['currency'] = currency
                    break
                    
        except Exception as e:
            print(f"⚠️ Error parsing strategy name '{name}': {e}")
        
        return info
    
    def generate_performance_report(self) -> str:
        if self.df.empty:
            return "No data available for performance analysis."
        
        try:
            report = []
            report.append("COMPREHENSIVE PERFORMANCE ANALYSIS REPORT")
            report.append("="*60)
            report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Total Strategies Analyzed: {len(self.df)}")
            report.append("")
            
            # Overall statistics
            report.append("OVERALL PERFORMANCE STATISTICS:")
            report.append("-" * 40)
            
            if 'total_return_pct' in self.df.columns:
                avg_return = self.df['total_return_pct'].mean()
                best_strategy = self.df.loc[self.df['total_return_pct'].idxmax(), 'Full_Name']
                best_return = self.df['total_return_pct'].max()
                worst_strategy = self.df.loc[self.df['total_return_pct'].idxmin(), 'Full_Name']
                worst_return = self.df['total_return_pct'].min()
                
                report.append(f"Average Return: {avg_return:.2f}%")
                report.append(f"Best Performer: {best_strategy} ({best_return:.2f}%)")
                report.append(f"Worst Performer: {worst_strategy} ({worst_return:.2f}%)")
                report.append("")
            
            # Strategy family comparison
            if 'Model_Family' in self.df.columns:
                report.append("PERFORMANCE BY STRATEGY FAMILY:")
                report.append("-" * 40)
                family_stats = self.df.groupby('Model_Family')['total_return_pct'].agg(['mean', 'std', 'count'])
                for family, stats in family_stats.iterrows():
                    report.append(f"{family}: Avg={stats['mean']:.2f}%, Std={stats['std']:.2f}%, Count={stats['count']}")
                report.append("")
            
            # Threshold analysis for CNN-LSTM strategies
            cnn_lstm_data = self.df[self.df['Model_Family'] == 'CNN-LSTM']
            if not cnn_lstm_data.empty and 'Threshold' in cnn_lstm_data.columns:
                report.append("CNN-LSTM THRESHOLD ANALYSIS:")
                report.append("-" * 40)
                threshold_stats = cnn_lstm_data.groupby('Threshold')['total_return_pct'].agg(['mean', 'std', 'count'])
                for threshold, stats in threshold_stats.iterrows():
                    report.append(f"{threshold}: Avg={stats['mean']:.2f}%, Std={stats['std']:.2f}%, Count={stats['count']}")
                report.append("")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"Error generating performance report: {e}"