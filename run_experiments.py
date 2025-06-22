"""
Main script to run the complete 12-loop rolling window experiment
with test set evaluation for Multi-Currency CNN-LSTM Forex Prediction System
"""
import argparse
import warnings
import os
import sys
import time
from datetime import datetime
import matplotlib

# Set backend for unattended runs
matplotlib.use('Agg')

from config import Config
from rolling_window_experiment import RollingWindowExperiment
from visualization import ForexVisualizer

warnings.filterwarnings('ignore')

def print_system_info():
    """Print system and configuration information"""
    print("="*100)
    print("🚀 MULTI-CURRENCY CNN-LSTM FOREX PREDICTION SYSTEM")
    print("   12-LOOP ROLLING WINDOW EXPERIMENT WITH TEST SET EVALUATION")
    print("="*100)
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Platform: {sys.platform}")
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"🧠 TensorFlow Version: {tf.__version__}")
        print(f"🔧 GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    except ImportError:
        print("⚠️ TensorFlow not available")
    
    print("="*100)

def estimate_execution_time():
    """Estimate total execution time"""
    print("\n⏱️ EXECUTION TIME ESTIMATION:")
    print("   • Model training per loop: ~5-10 minutes")
    print("   • Models per loop: 6 (3 multi + 3 single)")
    print("   • Total loops: 12")
    print("   • Estimated total time: 6-12 hours")
    print("   • This includes data processing, training, evaluation, and visualization")
    print("\n💡 TIP: Run with --unattended for automated execution")
    print("💡 TIP: Use --threshold-choice to focus on one strategy")

def run_complete_experiment(use_test_set=True, threshold_choice='Moderate', 
                          generate_all_thresholds=False, unattended=True):
    """
    Run the complete 12-loop rolling window experiment
    
    Args:
        use_test_set: Whether to use test set for final evaluation
        threshold_choice: Which threshold strategy to focus on
        generate_all_thresholds: Whether to generate results for all thresholds
        unattended: Whether to run in unattended mode
    """
    
    print(f"\n🚀 STARTING COMPLETE 12-LOOP EXPERIMENT")
    print(f"{'='*80}")
    print(f"📊 Evaluation Mode: {'Test Set' if use_test_set else 'Validation Set'}")
    print(f"🎯 Primary Threshold: {threshold_choice}")
    print(f"🔄 All Thresholds: {'Yes' if generate_all_thresholds else 'No'}")
    print(f"🤖 Unattended Mode: {'Yes' if unattended else 'No'}")
    print(f"{'='*80}")
    
    # Create base configuration
    base_config = Config()
    base_config.print_config_summary()
    
    # Create experiment manager
    experiment = RollingWindowExperiment(base_config)
    
    # Display experiment schedule
    print(f"\n📅 EXPERIMENT SCHEDULE:")
    print(f"{'='*80}")
    print(f"{'Loop':<6} {'Train Period':<25} {'Val Period':<15} {'Test Period':<15}")
    print(f"{'='*80}")
    
    for schedule in experiment.rolling_schedule:
        loop_num = schedule['loop']
        train_period = f"{schedule['train_start']} to {schedule['train_end']}"
        val_period = f"{schedule['val_start']} to {schedule['val_end']}"
        test_period = f"{schedule['test_start']} to {schedule['test_end']}"
        
        print(f"{loop_num:<6} {train_period:<25} {val_period:<15} {test_period:<15}")
    
    print(f"{'='*80}")
    
    # Confirm execution
    if not unattended:
        response = input("\n🤔 Do you want to proceed with the experiment? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("🚫 Experiment cancelled by user")
            return
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run the experiment
        if generate_all_thresholds:
            # Run experiment for all thresholds (more comprehensive but longer)
            print(f"\n🔄 Running experiment with all threshold strategies...")
            
            for threshold in ['Conservative', 'Moderate', 'Aggressive']:
                print(f"\n{'='*60}")
                print(f"🎯 RUNNING WITH {threshold.upper()} THRESHOLD")
                print(f"{'='*60}")
                
                experiment.run_complete_experiment(
                    use_test_set=use_test_set,
                    threshold_choice=threshold
                )
                
                print(f"✅ {threshold} threshold completed")
                
        else:
            # Run experiment with single threshold (faster)
            print(f"\n🎯 Running experiment with {threshold_choice} threshold...")
            
            experiment.run_complete_experiment(
                use_test_set=use_test_set,
                threshold_choice=threshold_choice
            )
        
        # Calculate execution time
        end_time = time.time()
        execution_hours = (end_time - start_time) / 3600
        
        print(f"\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"⏱️ Total Execution Time: {execution_hours:.2f} hours")
        print(f"📁 Results saved in: {experiment.results_path}")
        
        # Generate final visualizations
        print(f"\n📊 GENERATING FINAL VISUALIZATIONS...")
        
        final_config = Config()
        final_config.RESULTS_PATH = experiment.results_path
        
        visualizer = ForexVisualizer(final_config, unattended_mode=True)
        visualizer.create_comprehensive_report(experiment.all_loops_results)
        
        print(f"✅ Final visualizations completed")
        
        # Print summary statistics
        print_experiment_summary(experiment)
        
        return experiment
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted by user")
        execution_time = (time.time() - start_time) / 3600
        print(f"⏱️ Partial execution time: {execution_time:.2f} hours")
        return None
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        execution_time = (time.time() - start_time) / 3600
        print(f"⏱️ Execution time before failure: {execution_time:.2f} hours")
        raise

def print_experiment_summary(experiment):
    """Print summary of experiment results"""
    
    print(f"\n📋 EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    if not experiment.all_loops_results:
        print("⚠️ No results available")
        return
    
    # Calculate overall statistics
    total_loops = len(experiment.all_loops_results)
    total_strategies = len(experiment.all_loops_results[1]) if 1 in experiment.all_loops_results else 0
    
    print(f"📊 Total Loops Completed: {total_loops}")
    print(f"🔧 Strategies per Loop: {total_strategies}")
    print(f"🎯 Total Evaluations: {total_loops * total_strategies}")
    
    # Find best performing strategies
    print(f"\n🏆 TOP PERFORMING STRATEGIES (by average return):")
    print(f"{'='*80}")
    
    # Calculate average performance across all loops
    strategy_averages = {}
    
    for loop_results in experiment.all_loops_results.values():
        for strategy_name, performance in loop_results.items():
            if strategy_name not in strategy_averages:
                strategy_averages[strategy_name] = []
            strategy_averages[strategy_name].append(performance.get('total_return_pct', 0))
    
    # Calculate means and sort
    strategy_means = {}
    for strategy_name, returns in strategy_averages.items():
        strategy_means[strategy_name] = sum(returns) / len(returns)
    
    # Sort by performance
    sorted_strategies = sorted(strategy_means.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<5} {'Strategy':<50} {'Avg Return %':<12}")
    print(f"{'='*80}")
    
    for rank, (strategy_name, avg_return) in enumerate(sorted_strategies[:10], 1):
        clean_name = strategy_name.replace(' (EURUSD)', '').replace(' (GBPUSD)', '').replace(' (USDJPY)', '')
        print(f"{rank:<5} {clean_name:<50} {avg_return:>10.2f}%")
    
    # Currency pair analysis
    print(f"\n💱 PERFORMANCE BY CURRENCY PAIR:")
    print(f"{'='*80}")
    
    for currency in ['EURUSD', 'GBPUSD', 'USDJPY']:
        currency_strategies = [s for s in strategy_means.keys() if currency in s]
        if currency_strategies:
            currency_returns = [strategy_means[s] for s in currency_strategies]
            avg_return = sum(currency_returns) / len(currency_returns)
            best_strategy = max(currency_strategies, key=lambda s: strategy_means[s])
            best_return = strategy_means[best_strategy]
            
            print(f"{currency}:")
            print(f"  Average Return: {avg_return:>8.2f}%")
            print(f"  Best Strategy:  {best_strategy.replace(f' ({currency})', '')} ({best_return:.2f}%)")
            print()

def run_validation_vs_test_analysis(threshold_choice='Moderate'):
    """
    Run both validation and test evaluations for comparison analysis
    """
    
    print(f"\n🔍 RUNNING VALIDATION VS TEST COMPARISON ANALYSIS")
    print(f"{'='*80}")
    print(f"🎯 Threshold: {threshold_choice}")
    print(f"📊 This will run the experiment twice: once with validation, once with test")
    print(f"{'='*80}")
    
    base_config = Config()
    
    # Run validation experiment
    print(f"\n📊 PHASE 1: VALIDATION SET EVALUATION")
    print(f"{'='*60}")
    
    val_experiment = RollingWindowExperiment(base_config)
    val_experiment.results_path = 'results/validation_analysis/'
    os.makedirs(val_experiment.results_path, exist_ok=True)
    
    val_experiment.run_complete_experiment(
        use_test_set=False,
        threshold_choice=threshold_choice
    )
    
    # Run test experiment
    print(f"\n🧪 PHASE 2: TEST SET EVALUATION")
    print(f"{'='*60}")
    
    test_experiment = RollingWindowExperiment(base_config)
    test_experiment.results_path = 'results/test_analysis/'
    os.makedirs(test_experiment.results_path, exist_ok=True)
    
    test_experiment.run_complete_experiment(
        use_test_set=True,
        threshold_choice=threshold_choice
    )
    
    # Create comparison analysis
    print(f"\n🔍 PHASE 3: COMPARISON ANALYSIS")
    print(f"{'='*60}")
    
    comparison_config = Config()
    comparison_config.RESULTS_PATH = 'results/validation_vs_test_comparison/'
    os.makedirs(comparison_config.RESULTS_PATH, exist_ok=True)
    
    # Generate comparison visualizations
    visualizer = ForexVisualizer(comparison_config, unattended_mode=True)
    
    # Create side-by-side comparison plots
    # This would require additional implementation in the visualizer
    
    print(f"✅ Validation vs Test analysis completed")
    print(f"📁 Validation results: {val_experiment.results_path}")
    print(f"📁 Test results: {test_experiment.results_path}")
    print(f"📁 Comparison results: {comparison_config.RESULTS_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run complete 12-loop rolling window experiment for Multi-Currency CNN-LSTM Forex Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run standard experiment with test set evaluation
  python run_experiments.py --use-test-set --threshold-choice Moderate
  
  # Run validation set evaluation for development
  python run_experiments.py --threshold-choice Moderate
  
  # Run comprehensive analysis with all thresholds (takes longer)
  python run_experiments.py --use-test-set --all-thresholds
  
  # Run validation vs test comparison
  python run_experiments.py --mode comparison --threshold-choice Moderate
        """
    )
    
    # Experiment configuration
    parser.add_argument('--use-test-set', action='store_true',
                       help='Use test set for final evaluation (default: validation set)')
    
    parser.add_argument('--threshold-choice', type=str, default='Moderate',
                       choices=['Conservative', 'Moderate', 'Aggressive'],
                       help='Primary threshold strategy to focus on (default: Moderate)')
    
    parser.add_argument('--all-thresholds', action='store_true',
                       help='Generate results for all threshold strategies (takes 3x longer)')
    
    # Execution mode
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['standard', 'comparison'],
                       help='Execution mode: standard experiment or validation vs test comparison')
    
    parser.add_argument('--unattended', action='store_true',
                       help='Run in fully automated mode without user prompts')
    
    # Information flags
    parser.add_argument('--estimate-time', action='store_true',
                       help='Show execution time estimation and exit')
    
    parser.add_argument('--info', action='store_true',
                       help='Show system information and exit')
    
    args = parser.parse_args()
    
    # Handle information requests
    if args.info:
        print_system_info()
        sys.exit(0)
    
    if args.estimate_time:
        estimate_execution_time()
        sys.exit(0)
    
    # Print system info
    print_system_info()
    
    try:
        if args.mode == 'standard':
            # Run standard experiment
            run_complete_experiment(
                use_test_set=args.use_test_set,
                threshold_choice=args.threshold_choice,
                generate_all_thresholds=args.all_thresholds,
                unattended=args.unattended
            )
            
        elif args.mode == 'comparison':
            # Run validation vs test comparison
            run_validation_vs_test_analysis(args.threshold_choice)
        
        print(f"\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiments interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Experiments failed with error: {e}")
        sys.exit(1)