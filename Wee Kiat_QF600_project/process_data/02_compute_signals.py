import os
import pandas as pd
import wrds

# Define input/output directory
processed_dir = '../data/processed' if os.path.exists('../data/processed') else '.'

# 1. Load PIT universe panel
input_parquet_path = os.path.join(processed_dir, 'sp500_constituents_pit.parquet')
universe_df = pd.read_parquet(input_parquet_path)

# Convert Unix millisecond timestamps to datetime if present
for col in ['rebalance_date', 'snapshot_date']:
    if pd.api.types.is_numeric_dtype(universe_df[col]):
        universe_df[col] = pd.to_datetime(universe_df[col], unit='ms')

# Extract unique PERMNOs as native Python ints to avoid np.int64 string formatting
clean_permnos = [int(x) for x in universe_df['permno'].dropna().unique()]
permnos_formatted = f"({', '.join(map(str, clean_permnos))})"

# Calculate start date with a 13-month lookback buffer for 12-1 momentum
min_rebal_date = universe_df['rebalance_date'].min()
query_start_date = (min_rebal_date - pd.DateOffset(months=13)).strftime('%Y-%m-%d')
end_date = universe_df['rebalance_date'].max().strftime('%Y-%m-%d')

print(f"Unique PERMNOs: {len(clean_permnos)} | Query Range: {query_start_date} to {end_date}")

# 2. Connect to WRDS
db = wrds.Connection(
    wrds_username="",
    api_key=""
)
# 3. Query CRSP Monthly Stock File directly using clean PERMNO string
sql_query = f"""
SELECT 
    a.permno,
    a.date,
    ABS(a.altprc) AS price,
    COALESCE(a.ret, 0) AS ret,
    COALESCE(d.dlret, 0) AS dlret
FROM crsp.msf AS a
LEFT JOIN crsp.msedelist AS d
    ON a.permno = d.permno
   AND a.date = d.dlstdt
WHERE a.permno IN {permnos_formatted}
  AND a.date BETWEEN '{query_start_date}' AND '{end_date}'
ORDER BY a.permno, a.date;
"""

print("Executing WRDS SQL Query...")
crsp_data = db.raw_sql(sql_query)
db.close()

# 4. Compute true total return (adjusting for delisting returns)
crsp_data['total_return'] = (1 + crsp_data['ret']) * (1 + crsp_data['dlret']) - 1

# 5. Save raw WRDS price/return panel
output_parquet_path = os.path.join(processed_dir, 'wrds_crsp_monthly_returns.parquet')
crsp_data.to_parquet(output_parquet_path, index=False)
print(f"Saved WRDS returns data to {output_parquet_path}")