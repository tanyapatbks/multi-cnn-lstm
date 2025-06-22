import pandas as pd
import os
import argparse
from config import Config
from data_processor import DataProcessor
from trading_strategy import TechnicalIndicatorStrategies, SimpleBaselineStrategies
from checkpoint import ResultsManager

def evaluate_all_baselines(config_base: Config, use_test_set=False):
    all_baseline_results = {}
    
    master_processor = DataProcessor(config_base)
    raw_data = master_processor.load_currency_data(verbose=False)
    if raw_data is None: return
    processed_data = master_processor.preprocess_data(raw_data, verbose=False)

    print(f"\n📊 CALCULATING BASELINE STRATEGIES...")
    for currency_pair in ['EURUSD', 'GBPUSD', 'USDJPY']:
        print(f"--- For {currency_pair} ---")
        run_config = Config(model_type=currency_pair)
        
        eval_set_start = config_base.TEST_START if use_test_set else config_base.VAL_START
        eval_set_end = config_base.TEST_END if use_test_set else config_base.VAL_END

        price_data = master_processor.get_price_data_for_period(processed_data, currency_pair, eval_set_start, eval_set_end)
        
        if price_data.empty: continue

        eval_prices = price_data['Close_Price']
        eval_timestamps = price_data.index
        technical_indicators = master_processor.get_technical_indicators(processed_data, eval_timestamps, currency_pair)
        
        tech_strategy = TechnicalIndicatorStrategies(run_config)
        baseline_strategy = SimpleBaselineStrategies(run_config)
        
        all_baseline_results[f'Buy & Hold ({currency_pair})'] = baseline_strategy.buy_and_hold(eval_prices, eval_timestamps, currency_pair)
        all_baseline_results[f'RSI-based Trading ({currency_pair})'] = tech_strategy.rsi_strategy(eval_prices, eval_timestamps, technical_indicators, currency_pair)
        all_baseline_results[f'MACD-based Trading ({currency_pair})'] = tech_strategy.macd_strategy(eval_prices, eval_timestamps, technical_indicators, currency_pair)

    final_results_path = 'results/baseline_runs/'
    os.makedirs(final_results_path, exist_ok=True)
    results_manager = ResultsManager(Config())
    results_manager.RESULTS_PATH = final_results_path
    results_manager.save_results(all_baseline_results, "baseline_results.pkl", verbose=False)
    print(f"\n✅ All baseline results saved to {final_results_path}baseline_results.pkl")
    
    print("\n--- Baseline Performance Summary ---")
    summary_data = []
    for name, result in all_baseline_results.items():
        perf = result.get('performance', {})
        summary_data.append({
            'Strategy': name, 'Return (%)': perf.get('total_return_pct', 0),
            'Sharpe Ratio': perf.get('sharpe_ratio', 0), 'Win Rate': perf.get('win_rate', 0)})
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Baseline Strategies Evaluation.')
    args = parser.parse_args()
    base_config = Config()
    print(f"📊 STARTING BASELINE CALCULATION | PERIOD: {base_config.VAL_START} to {base_config.VAL_END}")
    evaluate_all_baselines(base_config)
    print(f"\n🎉 BASELINE CALCULATION COMPLETED.")