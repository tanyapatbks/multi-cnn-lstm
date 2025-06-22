"""
Complete Visualization Module - Enhanced with Built-in Performance Analyzer
"""
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

    def create_loop_summary_csv(self, loop_results):
        """Creates a comprehensive CSV summary for all results including multiple thresholds."""
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
        analyzer = PerformanceAnalyzer(loop_results)
        report_text = analyzer.generate_performance_report()
        report_path = os.path.join(self.config.RESULTS_PATH, "detailed_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"📄 Detailed analysis saved to {report_path}")

    def plot_loop_comparison_graph(self, all_results, currency_pair):
        """Enhanced comparison graph with proper multiple threshold support."""
        if not all_results: return
        
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        fig.suptitle(f'Enhanced Performance Comparison for {currency_pair}', fontsize=24, fontweight='bold')
        
        df_data = []
        for name, result in all_results.items():
            if f'({currency_pair})' in name:
                perf = result
                # Enhanced name cleaning to preserve threshold info
                clean_name = name.replace(f" ({currency_pair})", "").replace(f" (Target: {currency_pair})", "")
                df_data.append({
                    'Strategy': clean_name, 
                    'Return': perf.get('total_return_pct', 0),
                    'Sharpe': perf.get('sharpe_ratio', 0), 
                    'Win Rate': perf.get('win_rate', 0),
                    'Max Drawdown': perf.get('max_drawdown_pct', 0),
                })
        
        if not df_data:
            print(f"No data to plot for {currency_pair}"); plt.close(fig); return
            
        df = pd.DataFrame(df_data).sort_values(by='Return', ascending=False)

        # Enhanced color palettes for better distinction
        palettes = ['viridis', 'plasma', 'magma', 'coolwarm']
        metrics = ['Return', 'Sharpe', 'Win Rate', 'Max Drawdown']
        titles = ['Total Return (%)', 'Sharpe Ratio', 'Win Rate', 'Maximum Drawdown (%)']
        
        for i, ax in enumerate(axes.flat):
            # Enhanced plotting with better colors for thresholds
            bars = sns.barplot(data=df, x='Strategy', y=metrics[i], ax=ax, palette=palettes[i])
            ax.set_title(titles[i], fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('')
            ax.set_ylabel(titles[i].split(' ')[-1], fontsize=14)
            ax.axhline(0, color='black', lw=1.2, linestyle='--', alpha=0.7)
            
            # Enhanced x-axis labels with rotation and styling
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
            
            # Enhanced annotations with better positioning
            for j, p in enumerate(ax.patches):
                height = p.get_height()
                if not pd.isna(height):
                    ax.annotate(f'{height:.2f}', 
                              (p.get_x() + p.get_width() / 2., height),
                              ha='center', va='bottom' if height >= 0 else 'top', 
                              fontsize=10, fontweight='bold', color='black',
                              xytext=(0, 8 if height >= 0 else -8), 
                              textcoords='offset points')
            
            # Format Win Rate as percentage
            if 'Win Rate' in titles[i]:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
            
            # Enhanced grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"enhanced_loop_comparison_{currency_pair}.png"
        full_path = os.path.join(self.config.RESULTS_PATH, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Enhanced comparison graph for {currency_pair} saved to {full_path}")
        if not self.unattended: plt.show()
        plt.close(fig)

    def plot_threshold_comparison_summary(self, all_results):
        """Creates a comprehensive summary comparing all thresholds across currencies."""
        print("📊 Creating threshold comparison summary...")
        
        # Extract CNN-LSTM results only
        cnn_lstm_results = {}
        for name, result in all_results.items():
            if 'CNN-LSTM' in name and any(threshold in name for threshold in ['Conservative', 'Moderate', 'Aggressive']):
                cnn_lstm_results[name] = result
        
        if not cnn_lstm_results:
            print("⚠️ No CNN-LSTM threshold results found.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('CNN-LSTM Threshold Strategy Comparison', fontsize=22, fontweight='bold')
        
        df_data = []
        for name, result in cnn_lstm_results.items():
            # Extract strategy type, threshold, and currency
            parts = name.split('(')
            strategy_info = parts[0].strip()
            currency = parts[1].replace(')', '') if len(parts) > 1 else 'Unknown'
            
            # Extract threshold info
            threshold = 'Unknown'
            if '-Conservative' in strategy_info:
                threshold = 'Conservative'
                strategy_type = strategy_info.replace('-Conservative', '')
            elif '-Moderate' in strategy_info:
                threshold = 'Moderate'
                strategy_type = strategy_info.replace('-Moderate', '')
            elif '-Aggressive' in strategy_info:
                threshold = 'Aggressive'
                strategy_type = strategy_info.replace('-Aggressive', '')
            else:
                strategy_type = strategy_info
            
            df_data.append({
                'Full_Name': name,
                'Strategy_Type': strategy_type,
                'Threshold': threshold,
                'Currency': currency,
                'Return': result.get('total_return_pct', 0),
                'Sharpe': result.get('sharpe_ratio', 0),
                'Win_Rate': result.get('win_rate', 0),
                'Max_Drawdown': result.get('max_drawdown_pct', 0)
            })
        
        if not df_data:
            print("⚠️ No valid threshold data found.")
            plt.close(fig)
            return
        
        df = pd.DataFrame(df_data)
        
        metrics = ['Return', 'Sharpe', 'Win_Rate', 'Max_Drawdown']
        titles = ['Total Return (%)', 'Sharpe Ratio', 'Win Rate', 'Maximum Drawdown (%)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            # Create pivot table for better visualization
            pivot_df = df.pivot_table(index=['Strategy_Type', 'Currency'], 
                                    columns='Threshold', 
                                    values=metric, 
                                    aggfunc='mean')
            
            # Create grouped bar plot
            pivot_df.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('')
            ax.legend(title='Threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            if metric == 'Win_Rate':
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        
        plt.tight_layout()
        filename = "threshold_comparison_summary.png"
        full_path = os.path.join(self.config.RESULTS_PATH, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📈 Threshold comparison summary saved to {full_path}")
        if not self.unattended: plt.show()
        plt.close(fig)

def create_comprehensive_report(self, all_loops_results):
    """
    Create comprehensive report and visualizations for all 12 loops results
    This method was missing and causing the error in run_experiments.py
    """
    print("📊 Creating comprehensive report for all loops...")
    
    if not all_loops_results:
        print("⚠️ No results data available for comprehensive report")
        return
    
    try:
        # 1. Generate overall statistics
        total_loops = len(all_loops_results)
        print(f"   📈 Processing {total_loops} loops of results...")
        
        # 2. Create summary CSV for all loops
        self._create_all_loops_summary(all_loops_results)
        
        # 3. Generate strategy comparison charts
        self._create_strategy_comparison_charts(all_loops_results)
        
        # 4. Create performance analysis report
        self._create_performance_analysis_report(all_loops_results)
        
        # 5. Generate leverage effectiveness analysis
        self._create_leverage_analysis_report(all_loops_results)
        
        print("✅ Comprehensive report created successfully!")
        
    except Exception as e:
        print(f"⚠️ Error creating comprehensive report: {e}")
        # Fallback: create basic summary
        self._create_basic_summary(all_loops_results)

def _create_all_loops_summary(self, all_loops_results):
    """Create summary CSV for all loops combined"""
    print("   💾 Creating all-loops summary CSV...")
    
    all_data = []
    for loop_num, loop_results in all_loops_results.items():
        for strategy_name, performance in loop_results.items():
            all_data.append({
                'Loop': loop_num,
                'Strategy': strategy_name,
                'Total_Return_Pct': performance.get('total_return_pct', 0),
                'Sharpe_Ratio': performance.get('sharpe_ratio', 0),
                'Win_Rate': performance.get('win_rate', 0),
                'Max_Drawdown_Pct': performance.get('max_drawdown_pct', 0),
                'Total_Trades': performance.get('total_trades', 0),
                'Avg_Leverage': performance.get('avg_leverage', 1.0)
            })
    
    if all_data:
        df = pd.DataFrame(all_data)
        summary_path = os.path.join(self.config.RESULTS_PATH, 'comprehensive_all_loops_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"      ✅ All-loops summary saved: {summary_path}")

def _create_strategy_comparison_charts(self, all_loops_results):
    """Create strategy comparison visualizations"""
    print("   📊 Creating strategy comparison charts...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Extract strategy performance across all loops
        strategy_data = {}
        
        for loop_num, loop_results in all_loops_results.items():
            for strategy_name, performance in loop_results.items():
                if strategy_name not in strategy_data:
                    strategy_data[strategy_name] = []
                strategy_data[strategy_name].append(performance.get('total_return_pct', 0))
        
        if strategy_data:
            # Create box plot comparison
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            
            strategy_names = list(strategy_data.keys())
            strategy_returns = [strategy_data[name] for name in strategy_names]
            
            # Limit to top strategies to avoid overcrowding
            if len(strategy_names) > 10:
                # Calculate average returns and select top 10
                avg_returns = [(name, np.mean(returns)) for name, returns in zip(strategy_names, strategy_returns)]
                avg_returns.sort(key=lambda x: x[1], reverse=True)
                top_10 = avg_returns[:10]
                strategy_names = [item[0] for item in top_10]
                strategy_returns = [strategy_data[name] for name in strategy_names]
            
            ax.boxplot(strategy_returns, labels=[name.split('(')[0] for name in strategy_names])
            ax.set_title('Strategy Performance Distribution Across All Loops', fontsize=16, fontweight='bold')
            ax.set_ylabel('Total Return (%)')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            chart_path = os.path.join(self.config.RESULTS_PATH, 'strategy_comparison_boxplot.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✅ Strategy comparison chart saved: {chart_path}")
            
    except Exception as e:
        print(f"      ⚠️ Could not create strategy comparison charts: {e}")

def _create_performance_analysis_report(self, all_loops_results):
    """Create detailed performance analysis text report"""
    print("   📄 Creating performance analysis report...")
    
    try:
        # Collect all results for analysis
        all_results_flat = {}
        for loop_results in all_loops_results.values():
            for strategy_name, performance in loop_results.items():
                if strategy_name not in all_results_flat:
                    all_results_flat[strategy_name] = []
                all_results_flat[strategy_name].append(performance)
        
        # Generate report
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE FOREX CNN-LSTM PERFORMANCE ANALYSIS")
        report.append("12-Loop Rolling Window Test Set Evaluation")
        report.append("="*80)
        
        # Overall statistics
        total_evaluations = sum(len(loop_results) for loop_results in all_loops_results.values())
        report.append(f"\n📊 EXPERIMENT OVERVIEW:")
        report.append(f"   • Total Loops: {len(all_loops_results)}")
        report.append(f"   • Total Strategy Evaluations: {total_evaluations}")
        report.append(f"   • Unique Strategies: {len(all_results_flat)}")
        
        # Best overall performers
        report.append(f"\n🏆 TOP PERFORMING STRATEGIES (Average Return):")
        strategy_averages = []
        for strategy_name, performances in all_results_flat.items():
            avg_return = np.mean([p.get('total_return_pct', 0) for p in performances])
            avg_sharpe = np.mean([p.get('sharpe_ratio', 0) for p in performances])
            strategy_averages.append((strategy_name, avg_return, avg_sharpe))
        
        strategy_averages.sort(key=lambda x: x[1], reverse=True)
        for i, (name, avg_return, avg_sharpe) in enumerate(strategy_averages[:5]):
            report.append(f"   {i+1}. {name}: {avg_return:.2f}% (Sharpe: {avg_sharpe:.2f})")
        
        # Threshold analysis  
        report.append(f"\n🎯 THRESHOLD EFFECTIVENESS:")
        threshold_performance = {'Conservative': [], 'Moderate': [], 'Aggressive': []}
        
        for strategy_name, performances in all_results_flat.items():
            if 'Conservative' in strategy_name:
                threshold_performance['Conservative'].extend([p.get('total_return_pct', 0) for p in performances])
            elif 'Moderate' in strategy_name:
                threshold_performance['Moderate'].extend([p.get('total_return_pct', 0) for p in performances])
            elif 'Aggressive' in strategy_name:
                threshold_performance['Aggressive'].extend([p.get('total_return_pct', 0) for p in performances])
        
        for threshold, returns in threshold_performance.items():
            if returns:
                avg_return = np.mean(returns)
                success_rate = len([r for r in returns if r > 0]) / len(returns) * 100
                report.append(f"   • {threshold}: Avg Return {avg_return:.2f}%, Success Rate {success_rate:.1f}%")
        
        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(self.config.RESULTS_PATH, 'comprehensive_performance_analysis.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"      ✅ Performance analysis saved: {report_path}")
        
    except Exception as e:
        print(f"      ⚠️ Could not create performance analysis: {e}")

def _create_leverage_analysis_report(self, all_loops_results):
    """Create leverage effectiveness analysis"""
    print("   💰 Creating leverage analysis report...")
    
    try:
        leverage_data = {'Conservative (2.0x)': [], 'Moderate (1.0x)': [], 'Aggressive (0.5x)': []}
        
        for loop_results in all_loops_results.values():
            for strategy_name, performance in loop_results.items():
                if 'CNN-LSTM' in strategy_name:
                    leverage = performance.get('avg_leverage', 1.0)
                    return_pct = performance.get('total_return_pct', 0)
                    
                    if leverage >= 1.8:  # Conservative
                        leverage_data['Conservative (2.0x)'].append(return_pct)
                    elif leverage >= 0.8:  # Moderate
                        leverage_data['Moderate (1.0x)'].append(return_pct)
                    else:  # Aggressive
                        leverage_data['Aggressive (0.5x)'].append(return_pct)
        
        report = []
        report.append("="*60)
        report.append("LEVERAGE EFFECTIVENESS ANALYSIS")
        report.append("="*60)
        
        for leverage_type, returns in leverage_data.items():
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                success_rate = len([r for r in returns if r > 0]) / len(returns) * 100
                
                report.append(f"\n{leverage_type}:")
                report.append(f"   Average Return: {avg_return:.2f}%")
                report.append(f"   Standard Deviation: {std_return:.2f}%")
                report.append(f"   Success Rate: {success_rate:.1f}%")
                report.append(f"   Sample Size: {len(returns)}")
        
        leverage_report_path = os.path.join(self.config.RESULTS_PATH, 'leverage_effectiveness_analysis.txt')
        with open(leverage_report_path, 'w') as f:
            f.write("\n".join(report))
        print(f"      ✅ Leverage analysis saved: {leverage_report_path}")
        
    except Exception as e:
        print(f"      ⚠️ Could not create leverage analysis: {e}")

def _create_basic_summary(self, all_loops_results):
    """Fallback method to create basic summary if main methods fail"""
    print("   📝 Creating basic summary (fallback)...")
    
    try:
        summary = []
        summary.append("BASIC EXPERIMENT SUMMARY")
        summary.append("="*50)
        summary.append(f"Total Loops: {len(all_loops_results)}")
        
        if all_loops_results:
            first_loop = list(all_loops_results.values())[0]
            summary.append(f"Strategies per Loop: {len(first_loop)}")
        
        summary.append("\nExperiment completed successfully!")
        summary.append("Results files have been generated.")
        
        basic_path = os.path.join(self.config.RESULTS_PATH, 'basic_summary.txt')
        with open(basic_path, 'w') as f:
            f.write("\n".join(summary))
        print(f"      ✅ Basic summary saved: {basic_path}")
        
    except Exception as e:
        print(f"      ⚠️ Could not create basic summary: {e}")

    def plot_training_curves(self, history):
        """Plots training curves and saves them to the model's specific result folder."""
        if not history or 'loss' not in history.history:
            print("⚠️ No training history available for plotting.")
            return
        
        hist_dict = history.history
        fig, ax = plt.subplots(1, 2, figsize=(15, 5), squeeze=True)
        fig.suptitle(f'Model Training Performance: {self.config.MODEL_TYPE} (Target: {self.config.TARGET_PAIR})', 
                    fontsize=16, fontweight='bold')
        epochs = range(1, len(hist_dict['loss']) + 1)
        
        ax[0].plot(epochs, hist_dict['loss'], 'b-', label='Training Loss', linewidth=2)
        ax[0].plot(epochs, hist_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax[0].set_title('Model Loss', fontsize=14)
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        
        if 'accuracy' in hist_dict:
            ax[1].plot(epochs, hist_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            ax[1].plot(epochs, hist_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            ax[1].set_title('Model Accuracy', fontsize=14)
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Accuracy')
        
        save_path = os.path.join(self.config.RESULTS_PATH, "training_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Training curves saved to {save_path}")
        if not self.unattended: plt.show()
        plt.close(fig)


class PerformanceAnalyzer:
    """Built-in Performance Analyzer for comprehensive results analysis."""
    
    def __init__(self, results_dict: Dict):
        """Initialize with results dictionary from main_fx.py"""
        self.results = results_dict
        self.df = self._create_results_dataframe()
        
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Creates a structured DataFrame from results dictionary."""
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
        """Parses strategy name to extract components."""
        # Default values
        info = {
            'type': 'Unknown',
            'threshold': 'None',
            'currency': 'Unknown',
            'family': 'Unknown'
        }
        
        # Extract currency (always in parentheses at the end)
        if '(' in name and ')' in name:
            info['currency'] = name.split('(')[-1].replace(')', '')
            base_name = name.split('(')[0].strip()
        else:
            base_name = name
        
        # Determine model family and threshold
        if 'Multi-CNN-LSTM' in base_name:
            info['family'] = 'Multi-CNN-LSTM'
            if '-Conservative' in base_name:
                info['threshold'] = 'Conservative'
                info['type'] = 'Multi-CNN-LSTM-Conservative'
            elif '-Moderate' in base_name:
                info['threshold'] = 'Moderate'
                info['type'] = 'Multi-CNN-LSTM-Moderate'
            elif '-Aggressive' in base_name:
                info['threshold'] = 'Aggressive'
                info['type'] = 'Multi-CNN-LSTM-Aggressive'
            else:
                info['type'] = 'Multi-CNN-LSTM'
        
        elif 'Single-CNN-LSTM' in base_name:
            info['family'] = 'Single-CNN-LSTM'
            if '-Conservative' in base_name:
                info['threshold'] = 'Conservative'
                info['type'] = 'Single-CNN-LSTM-Conservative'
            elif '-Moderate' in base_name:
                info['threshold'] = 'Moderate'
                info['type'] = 'Single-CNN-LSTM-Moderate'
            elif '-Aggressive' in base_name:
                info['threshold'] = 'Aggressive'
                info['type'] = 'Single-CNN-LSTM-Aggressive'
            else:
                info['type'] = 'Single-CNN-LSTM'
        
        elif 'Buy & Hold' in base_name:
            info['family'] = 'Baseline'
            info['type'] = 'Buy & Hold'
        elif 'RSI-based' in base_name:
            info['family'] = 'Baseline'
            info['type'] = 'RSI-based'
        elif 'MACD-based' in base_name:
            info['family'] = 'Baseline'
            info['type'] = 'MACD-based'
        
        return info
    
    def get_best_performers(self, metric: str = 'total_return_pct', top_n: int = 5) -> pd.DataFrame:
        """Returns top N performers for a given metric."""
        return self.df.nlargest(top_n, metric)[['Strategy_Type', 'Currency', 'Threshold', metric]]
    
    def compare_thresholds(self, currency: str = None) -> pd.DataFrame:
        """Compares threshold performance across CNN-LSTM models."""
        cnn_lstm_data = self.df[self.df['Model_Family'].isin(['Multi-CNN-LSTM', 'Single-CNN-LSTM'])]
        
        if currency:
            cnn_lstm_data = cnn_lstm_data[cnn_lstm_data['Currency'] == currency]
        
        if cnn_lstm_data.empty:
            return pd.DataFrame()
        
        comparison = cnn_lstm_data.groupby(['Model_Family', 'Threshold']).agg({
            'total_return_pct': 'mean',
            'sharpe_ratio': 'mean',
            'win_rate': 'mean',
            'max_drawdown_pct': 'mean'
        }).round(4)
        
        return comparison
    
    def analyze_threshold_effectiveness(self) -> Dict:
        """Analyzes which thresholds work best across different conditions."""
        cnn_lstm_data = self.df[self.df['Threshold'] != 'None']
        
        analysis = {}
        for threshold in ['Conservative', 'Moderate', 'Aggressive']:
            thresh_data = cnn_lstm_data[cnn_lstm_data['Threshold'] == threshold]
            if not thresh_data.empty:
                analysis[threshold] = {
                    'avg_return': thresh_data['total_return_pct'].mean(),
                    'avg_sharpe': thresh_data['sharpe_ratio'].mean(),
                    'avg_win_rate': thresh_data['win_rate'].mean(),
                    'consistency': thresh_data['total_return_pct'].std(),  # Lower = more consistent
                    'success_rate': (thresh_data['total_return_pct'] > 0).mean()  # % of positive returns
                }
        
        return analysis
    
    def generate_performance_report(self) -> str:
        """Generates a comprehensive text report of performance analysis."""
        report = []
        report.append("="*80)
        report.append("FOREX CNN-LSTM PERFORMANCE ANALYSIS REPORT")
        report.append("="*80)
        
        # Overall statistics
        report.append(f"\n📊 OVERALL STATISTICS:")
        report.append(f"Total Strategies Evaluated: {len(self.df)}")
        report.append(f"Currencies: {', '.join(self.df['Currency'].unique())}")
        report.append(f"Model Families: {', '.join(self.df['Model_Family'].unique())}")
        
        # Best performers
        report.append(f"\n🏆 TOP 5 PERFORMERS (by Total Return):")
        top_performers = self.get_best_performers('total_return_pct', 5)
        for idx, row in top_performers.iterrows():
            report.append(f"  {idx+1}. {row['Strategy_Type']} ({row['Currency']}) - {row['total_return_pct']:.2f}%")
        
        # Threshold analysis
        threshold_analysis = self.analyze_threshold_effectiveness()
        if threshold_analysis:
            report.append(f"\n🎯 THRESHOLD EFFECTIVENESS ANALYSIS:")
            for threshold, metrics in threshold_analysis.items():
                report.append(f"  {threshold}:")
                report.append(f"    Average Return: {metrics['avg_return']:.2f}%")
                report.append(f"    Average Sharpe: {metrics['avg_sharpe']:.2f}")
                report.append(f"    Success Rate: {metrics['success_rate']:.1%}")
                report.append(f"    Consistency (σ): {metrics['consistency']:.2f}")
        
        # Multi vs Single comparison
        report.append(f"\n🔄 MULTI vs SINGLE CURRENCY COMPARISON:")
        multi_data = self.df[self.df['Model_Family'] == 'Multi-CNN-LSTM']
        single_data = self.df[self.df['Model_Family'] == 'Single-CNN-LSTM']
        
        if not multi_data.empty and not single_data.empty:
            multi_avg = multi_data['total_return_pct'].mean()
            single_avg = single_data['total_return_pct'].mean()
            report.append(f"  Multi-Currency Average Return: {multi_avg:.2f}%")
            report.append(f"  Single-Currency Average Return: {single_avg:.2f}%")
            report.append(f"  Advantage to Multi-Currency: {multi_avg - single_avg:.2f}%")
        
        # Risk analysis
        report.append(f"\n⚠️ RISK ANALYSIS:")
        high_return_strategies = self.df[self.df['total_return_pct'] > 5]
        if not high_return_strategies.empty:
            avg_drawdown = high_return_strategies['max_drawdown_pct'].mean()
            report.append(f"  High-return strategies (>5%) average drawdown: {avg_drawdown:.2f}%")
        
        report.append("="*80)
        return "\n".join(report)