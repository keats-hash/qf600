(.venv) nwk@Mac1 backtest % python3 04_evaluate_performance.py
========================================================
      STEP 5: BENCHMARK PERFORMANCE COMPARISON          
========================================================
               Metric L/S Momentum Strategy S&P 500 / Market Proxy
    Annualized Return                -2.66%                 10.20%
Annualized Volatility                24.83%                 17.61%
         Sharpe Ratio                 -0.11                   0.37

========================================================
      STEP 6: CAPM ALPHA REGRESSION RESULTS             
========================================================
Annualized Alpha (α):  3.06%
Market Beta (β):        -0.7408
Alpha t-statistic:     0.70
Alpha p-value:         0.4836
--------------------------------------------------------
DECISION: Fail to reject H0. Returns are driven by market exposure/noise.
========================================================