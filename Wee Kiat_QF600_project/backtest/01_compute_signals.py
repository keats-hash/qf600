import os
import numpy as np
import pandas as pd

def compute_momentum_signals(input_path: str, output_path: str):
    print("Loading master panel...")
    df = pd.read_parquet(input_path)
    
    # 1. Parse UNIX millisecond timestamps and sort properly
    df['rebalance_date'] = pd.to_datetime(df['rebalance_date'], unit='ms')
    
    # 2. Integrate delisting returns to prevent survivorship bias
    df['ret'] = df['ret'].fillna(0)
    df['dlret'] = df['dlret'].fillna(0)
    df['true_total_return'] = (1 + df['ret']) * (1 + df['dlret']) - 1
    df['log_ret'] = np.log(1 + df['true_total_return'])
    
    # Sort chronologically per stock
    df = df.sort_values(['permno', 'rebalance_date']).reset_index(drop=True)
    
    # 3. Compute 12-1 Trailing Momentum (Window: t-12 to t-2)
    print("Computing 12-1 momentum signals...")
    df['mom_log'] = (
        df.groupby('permno')['log_ret']
        .transform(lambda x: x.shift(2).rolling(window=11, min_periods=11).sum())
    )
    
    df['momentum_12_1_raw'] = np.exp(df['mom_log']) - 1
    
    # 4. Filter for valid signals
    valid_panel = df.dropna(subset=['momentum_12_1_raw']).copy()
    
    # 4. Filter for valid signals
    valid_panel = df.dropna(subset=['momentum_12_1_raw']).copy()
    
    # 5. Point-in-time cross-sectional winsorization (1st and 99th Percentiles)
    print("Winsorizing signal outliers point-in-time...")
    valid_panel['momentum_12_1'] = valid_panel.groupby('rebalance_date')['momentum_12_1_raw'].transform(
        lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
    )
    
    print(f"Calculated signals for {len(valid_panel)} stock-month observations.")
    print(f"Coverage: {valid_panel['rebalance_date'].nunique()} month-ends.")
    
    # 6. Save Processed Dataset
    valid_panel.to_parquet(output_path, index=False)
    print(f"Saved dataset with signals to: {output_path}")

if __name__ == "__main__":
    processed_dir = '../data/processed' if os.path.exists('../data/processed') else '.'
    
    input_file = os.path.join(processed_dir, "backtest_master_panel.parquet")
    output_file = os.path.join(processed_dir, "backtest_panel_with_signals.parquet")
    
    compute_momentum_signals(
        input_path=input_file,
        output_path=output_file
    )