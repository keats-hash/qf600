import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_benchmark_and_capm(portfolio_parquet_path: str, master_panel_parquet_path: str, rf_monthly: float = 0.003):
    """
    Executes Step 5 (Benchmark Comparison) and Step 6 (CAPM Alpha Isolation).
    """
    # 1. Load Strategy Returns & Master Panel
    strat_df = pd.read_parquet(portfolio_parquet_path)
    panel_df = pd.read_parquet(master_panel_parquet_path)
    
    strat_df['rebalance_date'] = pd.to_datetime(strat_df['rebalance_date'])
    panel_df['rebalance_date'] = pd.to_datetime(panel_df['rebalance_date'])

    # 2. Compute Benchmark Return (Equal-Weighted Market Proxy)
    # Map forward returns cleanly with a 35-day gap limit
    panel_df = panel_df.sort_values(['permno', 'rebalance_date'])
    panel_df['next_date'] = panel_df.groupby('permno')['rebalance_date'].shift(-1)
    panel_df['holding_return_t1'] = panel_df.groupby('permno')['total_return'].shift(-1)
    
    invalid_gap = (panel_df['next_date'] - panel_df['rebalance_date']).dt.days > 35
    panel_df.loc[invalid_gap, 'holding_return_t1'] = np.nan

    market_df = panel_df.groupby('rebalance_date')['holding_return_t1'].mean().reset_index()
    market_df.rename(columns={'holding_return_t1': 'Market_Return'}, inplace=True)

    # 3. Merge Strategy Returns with Market Returns
    df = pd.merge(strat_df, market_df, on='rebalance_date').dropna()

    # 4. STEP 5: Metric Comparison Math
    ann_factor = 12
    
    # Strategy vs Benchmark Excess Return
    # A L/S spread is already a zero-cost excess return, no need to subtract RF.
    df['Strategy_Excess'] = df['Long_Short_Spread'] 
    df['Market_Excess'] = df['Market_Return'] - rf_monthly

    # Performance Summaries
    # Arithmetic annualization for zero-cost L/S spread
    strat_ann_ret = df['Long_Short_Spread'].mean() * ann_factor
    
    # Geometric annualization for fully funded market proxy
    mkt_ann_ret = (1 + df['Market_Return']).prod() ** (ann_factor / len(df)) - 1
    
    strat_vol = df['Long_Short_Spread'].std() * np.sqrt(ann_factor)
    mkt_vol = df['Market_Return'].std() * np.sqrt(ann_factor)

    # Sharpe Ratios
    strat_sharpe = strat_ann_ret / strat_vol if strat_vol != 0 else 0
    mkt_sharpe = (mkt_ann_ret - (rf_monthly * ann_factor)) / mkt_vol if mkt_vol != 0 else 0

    print("========================================================")
    print("      STEP 5: BENCHMARK PERFORMANCE COMPARISON          ")
    print("========================================================")
    comp_table = pd.DataFrame({
        'Metric': ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'],
        'L/S Momentum Strategy': [f"{strat_ann_ret:.2%}", f"{strat_vol:.2%}", f"{strat_sharpe:.2f}"],
        'S&P 500 / Market Proxy': [f"{mkt_ann_ret:.2%}", f"{mkt_vol:.2%}", f"{mkt_sharpe:.2f}"]
    })
    print(comp_table.to_string(index=False))

    # 5. STEP 6: CAPM Regression (Alpha Isolation)
    X = sm.add_constant(df['Market_Excess'])
    y = df['Strategy_Excess']
    
    model = sm.OLS(y, X).fit()
    
    alpha_monthly = model.params['const']
    alpha_annualized = alpha_monthly * 12
    beta = model.params['Market_Excess']
    p_value_alpha = model.pvalues['const']
    t_stat_alpha = model.tvalues['const']

    print("\n========================================================")
    print("      STEP 6: CAPM ALPHA REGRESSION RESULTS             ")
    print("========================================================")
    print(f"Annualized Alpha (\u03b1):  {alpha_annualized:.2%}")
    print(f"Market Beta (\u03b2):        {beta:.4f}")
    print(f"Alpha t-statistic:     {t_stat_alpha:.2f}")
    print(f"Alpha p-value:         {p_value_alpha:.4f}")
    print("--------------------------------------------------------")
    if p_value_alpha < 0.05 and t_stat_alpha > 1.96:
        print("DECISION: Reject H0. The strategy generates statistically significant Alpha.")
    else:
        print("DECISION: Fail to reject H0. Returns are driven by market exposure/noise.")
    print("========================================================\n")

if __name__ == "__main__":
    run_benchmark_and_capm(
        portfolio_parquet_path="../data/processed/momentum_strategy_returns.parquet",
        master_panel_parquet_path="../data/processed/backtest_master_panel.parquet"
    )