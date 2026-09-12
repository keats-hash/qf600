import os
import pandas as pd
import numpy as np

def compute_realized_performance(input_path: str):
    print("Loading monthly portfolio returns...")
    df = pd.read_parquet(input_path)
    df['rebalance_date'] = pd.to_datetime(df['rebalance_date'])
    df = df.sort_values('rebalance_date').reset_index(drop=True)

    # 1. Realized Monthly Spread Calculation
    # Strategy Return = Long Leg (Decile 10) - Short Leg (Decile 1)
    df['Realized_Spread'] = df['Long_Leg'] - df['Short_Leg']

    # 2. Compute Summary Statistics (Annualized)
    num_months = len(df)
    ann_factor = 12

    # Mean Returns
    mean_long = df['Long_Leg'].mean() * ann_factor
    mean_short = df['Short_Leg'].mean() * ann_factor
    mean_spread = df['Realized_Spread'].mean() * ann_factor

    # Volatility (Standard Deviation)
    vol_long = df['Long_Leg'].std() * np.sqrt(ann_factor)
    vol_short = df['Short_Leg'].std() * np.sqrt(ann_factor)
    vol_spread = df['Realized_Spread'].std() * np.sqrt(ann_factor)

    # Sharpe Ratios (Assuming 0% risk-free rate for baseline)
    sharpe_long = mean_long / vol_long if vol_long != 0 else 0
    sharpe_spread = mean_spread / vol_spread if vol_spread != 0 else 0

    # 3. Print Performance Table
    print("\n========================================================")
    print("      STEP 4: REALIZED STRATEGY PERFORMANCE SUMMARY      ")
    print("========================================================")
    metrics_df = pd.DataFrame({
        'Metric': ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'],
        'Long Leg (D10)': [f"{mean_long*100:.2f}%", f"{vol_long*100:.2f}%", f"{sharpe_long:.2f}"],
        'Short Leg (D1)': [f"{mean_short*100:.2f}%", f"{vol_short*100:.2f}%", "-"],
        'Long-Short Spread': [f"{mean_spread*100:.2f}%", f"{vol_spread*100:.2f}%", f"{sharpe_spread:.2f}"]
    })
    print(metrics_df.to_string(index=False))
    print("========================================================\n")

    # Display sample individual realized months
    print("Sample Realized Monthly Returns:")
    sample = df[['rebalance_date', 'Long_Leg', 'Short_Leg', 'Realized_Spread']].head(5).copy()
    sample['Long_Leg'] = sample['Long_Leg'].map('{:.2%}'.format)
    sample['Short_Leg'] = sample['Short_Leg'].map('{:.2%}'.format)
    sample['Realized_Spread'] = sample['Realized_Spread'].map('{:.2%}'.format)
    print(sample.to_string(index=False))

if __name__ == "__main__":
    processed_dir = '../data/processed' if os.path.exists('../data/processed') else '.'
    input_file = os.path.join(processed_dir, "momentum_strategy_returns.parquet")
    
    compute_realized_performance(input_file)