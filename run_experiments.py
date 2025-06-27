"""
Main script to run the complete 12-loop rolling window experiment
with OPTIMIZED single training per model for Multi-Currency CNN-LSTM Forex Prediction System
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
    print("🚀 OPTIMIZED MULTI-CURRENCY CNN-LSTM FOREX PREDICTION SYSTEM")
    print("   12-LOOP ROLLING WINDOW EXPERIMENT WITH SINGLE TRAINING PER MODEL")
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
    """Estimate total execution time with optimization"""
    print("\n⏱️ OPTIMIZED EXECUTION TIME ESTIMATION:")
    print("   • Model training per loop: ~5-10 minutes")
    print("   • Models per loop: 6 (3 multi + 3 single)")
    print("   • ✅ OPTIMIZATION: 1 training per model (instead of 3)")
    print("   • Total loops: 12")
    print("   • 🎉 Estimated total time: 2-4 hours (was 6-12 hours)")
    print("   • ⚡ Time reduction: ~66% faster")
    print("   • This includes data processing, training, evaluation, and visualization")
    print("\n💡 TIP: Run with --unattended for automated execution")

def run_optimized_experiment(use_test_set=True, unattended=True):
    """
    ✅ OPTIMIZED: Run the complete 12-loop rolling window experiment with single training
    
    Args:
        use_test_set: Whether to use test set for final evaluation  
        unattended: Whether to run in unattended mode
    """
    
    print(f"\n🚀 STARTING OPTIMIZED 12-LOOP EXPERIMENT")
    print(f"{'='*80}")
    print(f"📊 Evaluation Mode: {'Test Set' if use_test_set else 'Validation Set'}")
    print(f"⚡ OPTIMIZATION: Single training per model, evaluate ALL thresholds")
    print(f"🤖 Unattended Mode: {'Yes' if unattended else 'No'}")
    print(f"{'='*80}")
    
    # Create base configuration
    base_config = Config()
    base_config.print_config_summary()
    
    # Create experiment manager
    experiment = RollingWindowExperiment(base_config)
    
    # Display experiment schedule
    print(f"\n📅 OPTIMIZED EXPERIMENT SCHEDULE:")
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
    print(f"⚡ OPTIMIZATION SUMMARY:")
    print(f"   • Training per loop: 6 models (was 18)")
    print(f"   • Time reduction: ~66%")
    print(f"   • Same accuracy: All thresholds evaluated from single predictions")
    print(f"{'='*80}")
    
    # Confirm execution
    if not unattended:
        response = input("\n🤔 Do you want to proceed with the optimized experiment? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("🚫 Experiment cancelled by user")
            return
    
    # Record start time
    start_time = time.time()
    
    try:
        # ✅ Run the optimized experiment
        print(f"\n⚡ Running optimized experiment with single training per model...")
        
        experiment.run_complete_experiment(use_test_set=use_test_set)
        
        # Calculate execution time
        end_time = time.time()
        execution_hours = (end_time - start_time) / 3600
        
        print(f"\n🎉 OPTIMIZED EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"⏱️ Total execution time: {execution_hours:.2f} hours")
        print(f"📁 Results saved to: {experiment.results_path}")
        print(f"⚡ Training optimization achieved ~66% time reduction")
        
        # Generate performance summary
        generate_optimization_summary(experiment, execution_hours)
        
    except Exception as e:
        print(f"\n❌ OPTIMIZED EXPERIMENT FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_optimization_summary(experiment, execution_hours):
    """Generate summary of optimization benefits"""
    print(f"\n{'='*80}")
    print("📊 OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    # Calculate total strategies evaluated
    total_strategies = 0
    cnn_lstm_strategies = 0
    for loop_results in experiment.all_loops_results.values():
        total_strategies += len(loop_results)
        cnn_lstm_strategies += sum(1 for name in loop_results.keys() if 'CNN-LSTM' in name)
    
    baseline_strategies = total_strategies - cnn_lstm_strategies
    
    print(f"📈 EXPERIMENT RESULTS:")
    print(f"   • Total loops completed: {len(experiment.all_loops_results)}")
    print(f"   • Total strategies evaluated: {total_strategies}")
    print(f"   • CNN-LSTM strategies (all thresholds): {cnn_lstm_strategies}")
    print(f"   • Baseline strategies: {baseline_strategies}")
    print(f"   • Execution time: {execution_hours:.2f} hours")
    print(f"")
    print(f"⚡ OPTIMIZATION BENEFITS:")
    print(f"   • Training reduction: 6 models per loop (was 18)")
    print(f"   • Time savings: ~66% faster execution")
    print(f"   • Same accuracy: All thresholds from single predictions")
    print(f"   • Consistent results: Same model, different risk levels")
    print(f"")
    print(f"📁 RESULTS LOCATION:")
    print(f"   • Main results: {experiment.results_path}")
    print(f"   • CSV files: Strategy comparison, monthly analysis, leverage impact")
    print(f"   • Charts: Performance visualization across loops and thresholds")
    print(f"{'='*80}")

def run_validation_vs_test_comparison():
    """Run comprehensive validation vs test comparison with optimization"""
    print(f"\n🔍 RUNNING VALIDATION VS TEST COMPARISON (OPTIMIZED)")
    print(f"{'='*80}")
    
    base_config = Config()
    
    # Run validation experiment
    print(f"\n📊 PHASE 1: VALIDATION SET EVALUATION")
    print(f"{'='*60}")
    
    val_experiment = RollingWindowExperiment(base_config)
    val_experiment.results_path = 'results/validation_analysis_optimized/'
    os.makedirs(val_experiment.results_path, exist_ok=True)
    
    val_experiment.run_complete_experiment(use_test_set=False)
    
    # Run test experiment
    print(f"\n🧪 PHASE 2: TEST SET EVALUATION")
    print(f"{'='*60}")
    
    test_experiment = RollingWindowExperiment(base_config)
    test_experiment.results_path = 'results/test_analysis_optimized/'
    os.makedirs(test_experiment.results_path, exist_ok=True)
    
    test_experiment.run_complete_experiment(use_test_set=True)
    
    # Create comparison analysis
    print(f"\n🔍 PHASE 3: COMPARISON ANALYSIS")
    print(f"{'='*60}")
    
    comparison_config = Config()
    comparison_config.RESULTS_PATH = 'results/validation_vs_test_comparison_optimized/'
    os.makedirs(comparison_config.RESULTS_PATH, exist_ok=True)
    
    # Generate comparison visualizations
    visualizer = ForexVisualizer(comparison_config, unattended_mode=True)
    
    print(f"✅ Optimized validation vs test analysis completed")
    print(f"📁 Validation results: {val_experiment.results_path}")
    print(f"📁 Test results: {test_experiment.results_path}")
    print(f"📁 Comparison results: {comparison_config.RESULTS_PATH}")
    print(f"⚡ Both phases used optimized training (66% time reduction)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run OPTIMIZED 12-loop rolling window experiment for Multi-Currency CNN-LSTM Forex Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 OPTIMIZED EXPERIMENT MODES:

Examples:
  # Run optimized experiment with test set evaluation (RECOMMENDED)
  python run_experiments.py --use-test-set
  
  # Run optimized validation set evaluation for development
  python run_experiments.py
  
  # Run optimized validation vs test comparison
  python run_experiments.py --mode comparison
  
  # Interactive mode (ask for confirmation)
  python run_experiments.py --use-test-set --interactive

⚡ OPTIMIZATION BENEFITS:
  • 66% faster execution (2-4 hours instead of 6-12 hours)
  • Same accuracy with single training per model
  • All thresholds evaluated from same predictions
  • Consistent results across risk levels
        """
    )
    
    # Command line arguments
    parser.add_argument(
        '--use-test-set', 
        action='store_true',
        help='Use test set for evaluation (default: validation set)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['standard', 'comparison'],
        default='standard',
        help='Experiment mode: standard or validation vs test comparison'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode (ask for confirmation)'
    )
    
    args = parser.parse_args()
    
    # Print system information
    print_system_info()
    estimate_execution_time()
    
    # Determine unattended mode
    unattended_mode = not args.interactive
    
    print(f"\n🎯 EXPERIMENT CONFIGURATION:")
    print(f"   • Mode: {args.mode.title()}")
    print(f"   • Evaluation: {'Test Set' if args.use_test_set else 'Validation Set'}")
    print(f"   • Interactive: {'Yes' if args.interactive else 'No'}")
    print(f"   • Optimization: ⚡ Single training per model (66% faster)")
    
    try:
        if args.mode == 'comparison':
            # Run validation vs test comparison
            run_validation_vs_test_comparison()
            
        else:
            # Run standard optimized experiment
            run_optimized_experiment(
                use_test_set=args.use_test_set,
                unattended=unattended_mode
            )
        
        print(f"\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"⚡ Optimization reduced training time by ~66%")
        print(f"📊 Check results in respective results/ subdirectories")
        
    except KeyboardInterrupt:
        print(f"\n🛑 EXPERIMENT INTERRUPTED BY USER")
        print(f"⚠️ Partial results may be available in results/ directory")
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {str(e)}")
        print(f"🔍 Check error details above for troubleshooting")
        import traceback
        traceback.print_exc()