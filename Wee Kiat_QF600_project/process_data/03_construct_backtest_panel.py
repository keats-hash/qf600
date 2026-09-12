import os
import pandas as pd

# Define input/output directory
processed_dir = '../data/processed' if os.path.exists('../data/processed') else '.'

# 1. Load PIT constituents and WRDS returns
pit_path = os.path.join(processed_dir, 'sp500_constituents_pit.parquet')
wrds_path = os.path.join(processed_dir, 'wrds_crsp_monthly_returns.parquet')

pit_df = pd.read_parquet(pit_path)
wrds_df = pd.read_parquet(wrds_path)

# Convert Unix millisecond timestamps if present
for col in ['rebalance_date', 'snapshot_date']:
    if pd.api.types.is_numeric_dtype(pit_df[col]):
        pit_df[col] = pd.to_datetime(pit_df[col], unit='ms')

wrds_df['date'] = pd.to_datetime(wrds_df['date'])

# Standardize date formats to Year-Month period for robust merging
pit_df['ym'] = pit_df['rebalance_date'].dt.to_period('M')
wrds_df['ym'] = wrds_df['date'].dt.to_period('M')

# --- CRITICAL FIX 1: Deduplicate PIT Universe on (rebalance_date, permno) ---
# Prevents historical dual-class/ticker swaps (like C and TRV) from duplicating rows
pit_df = pit_df.sort_values(['rebalance_date', 'permno']).drop_duplicates(
    subset=['rebalance_date', 'permno'], keep='first'
).reset_index(drop=True)

# --- CRITICAL FIX 2: Deduplicate CRSP Panel before shifting ---
# Ensures purely sequential month-to-month calculation for forward returns
wrds_df = wrds_df.drop_duplicates(subset=['permno', 'ym']).sort_values(['permno', 'date']).reset_index(drop=True)

# 2. Compute 1-month forward return (holding return for t+1) in full CRSP panel
wrds_df['fwd_return'] = wrds_df.groupby('permno')['total_return'].shift(-1)

# 3. Merge PIT universe with CRSP market data using PERMNO (prevents symbology bias)
panel = pd.merge(
    pit_df[['rebalance_date', 'snapshot_date', 'ticker', 'permno', 'ym']],
    wrds_df[['permno', 'ym', 'price', 'ret', 'dlret', 'total_return', 'fwd_return']],
    on=['permno', 'ym'],
    how='left'  # Left join preserves constituents and avoids coverage loss
)

# Audit missing return entries
missing_returns = panel['total_return'].isna().sum()
if missing_returns > 0:
    print(f"Warning: {missing_returns} constituent-month records missing CRSP returns.")

# 4. Sort cleanly by rebalance date and PERMNO
panel = panel.sort_values(['rebalance_date', 'permno']).reset_index(drop=True)

# 5. Save final analysis-ready dataset to ../data/processed/
output_path = os.path.join(processed_dir, 'backtest_master_panel.parquet')
panel.to_parquet(output_path, index=False)

print(f"Master Backtesting Panel Created: {len(panel)} rows across {panel['rebalance_date'].nunique()} month-ends.")
print(f"Saved panel to: {output_path}")