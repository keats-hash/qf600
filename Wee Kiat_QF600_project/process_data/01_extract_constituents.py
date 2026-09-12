import os
import pandas as pd
import wrds

# 1. Load Constituents CSV from ../data
data_dir = '../data' if os.path.exists('../data') else '.'
csv_path = os.path.join(data_dir, 'S&P 500 Historical Components & Changes (Updated).csv')

df_const = pd.read_csv(csv_path)
df_const['date'] = pd.to_datetime(df_const['date'])
df_const = df_const.sort_values('date').reset_index(drop=True)

# 2. Define Month-End Rebalance Schedule (2000 to 2024)
rebalance_dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='ME')

# 3. Build Point-in-Time Mapping Panel
pit_records = []

for me in rebalance_dates:
    valid_snapshots = df_const[df_const['date'] <= me]
    latest_snapshot = valid_snapshots.iloc[-1]
    
    tickers = [t.strip() for t in str(latest_snapshot['tickers']).split(',') if t.strip()]
    
    for ticker in tickers:
        pit_records.append({
            'rebalance_date': me,
            'snapshot_date': latest_snapshot['date'],
            'ticker': ticker
        })

pit_df = pd.DataFrame(pit_records)
pit_df['snapshot_date_str'] = pit_df['snapshot_date'].dt.strftime('%Y-%m-%d')

# 4. Extract unique (ticker, snapshot_date) pairs for WRDS query
unique_pairs = pit_df[['ticker', 'snapshot_date_str']].drop_duplicates()

# Format as SQL VALUES tuple string: ('AAPL', '2000-01-31'::date), ('MSFT', '2000-01-31'::date)...
values_clause = ",\n".join([
    f"('{row.ticker}', '{row.snapshot_date_str}'::date)" 
    for row in unique_pairs.itertuples()
])

# 5. Connect to WRDS and map to PERMNOs
db = wrds.Connection(
    wrds_username="",
    api_key=""
)
permno_map_query = f"""
WITH pit_inputs(ticker, snapshot_date) AS (
    VALUES {values_clause}
)
SELECT DISTINCT
    p.ticker,
    p.snapshot_date::text AS snapshot_date_str,
    s.permno
FROM pit_inputs AS p
INNER JOIN crsp.stocknames AS s
    ON p.ticker = s.ticker
   AND p.snapshot_date >= s.st_date
   AND p.snapshot_date <= s.end_date;
"""

print("Resolving Point-in-Time tickers to CRSP PERMNOs...")
mapped_permnos = db.raw_sql(permno_map_query)
db.close()

# 6. Merge PERMNOs back to main PIT panel
pit_df = pd.merge(
    pit_df,
    mapped_permnos,
    on=['ticker', 'snapshot_date_str'],
    how='left'
).drop(columns=['snapshot_date_str'])

# Audit unmapped tickers
unmapped_count = pit_df['permno'].isna().sum()
if unmapped_count > 0:
    print(f"Warning: {unmapped_count} constituent entries failed to resolve to a PERMNO.")

# 7. Save to Parquet under ../data/processed/
output_dir = '../data/processed'
os.makedirs(output_dir, exist_ok=True)

parquet_path = os.path.join(output_dir, 'sp500_constituents_pit.parquet')
pit_df.to_parquet(parquet_path, index=False)

print(f"Point-In-Time panel created: {len(pit_df)} row entries with PERMNOs.")