## Project Title

Does Our Strategy Have Real Alpha? — A Long-Short Momentum Backtest on S&P 500 Constituents

## Course Context

QF600 · Asset Pricing · Group Homework 

## Objective

Backtest a long-short momentum strategy on S&P 500 constituents, using point-in-time index membership to avoid survivorship bias, then use a CAPM regression to separate genuine skill (alpha, α) from simple market exposure (beta, β).

## Background / Theory

The CAPM regression form is:

$R_p - R_f = \alpha + \beta(R_M - R_f) + \varepsilon$

- **α (alpha)** = risk-adjusted excess return — the part of performance not explained by the market. This is what represents "skill."
- **β (beta)** = the strategy's sensitivity to market movements — this is just riding the market, not skill.
- The CAPM's prediction is that α = 0. A statistically significant, positive α would suggest the strategy adds real value beyond market exposure.
- A long-short construction is a stronger test than long-only: it's closer to market-neutral by design, so a significant α is more convincing evidence of skill rather than just equity-market exposure.

## Data Assets

### Owned data

- **S&P 500 historical constituents**: `S_P_500_Historical_Components_&_Changes_(Updated).csv`
    - Format: one row per membership-change date, with a `date` column and a `tickers` column (comma-separated list of all tickers in the index as of that date)
    - Coverage: 1996-01-02 to 2026-08-18 (2,720 rows)
    - Ticker count grows from 487 (1996) to 503 (present) as the index composition rules evolved
    - **This is point-in-time data** — it correctly reflects who was in/out of the index at each date, which is what allows us to avoid survivorship bias

### Data still needed

- **Historical prices** for every unique ticker that appears anywhere in the constituents file (not just current members) — needed to compute returns for the momentum signal and the strategy's realized returns
- **Market proxy prices**: S&P 500 index itself (e.g., ^GSPC or SPY) for the CAPM regression's R_M
- **Risk-free rate**: 3-month T-bill (e.g., FRED ticker `DGS3MO`, or `^IRX`), converted to a monthly rate to match rebalancing frequency

## Universe Selection

- **Assets**: S&P 500 constituents, using the point-in-time membership file above (dynamic universe, not a static list)
- **Period**: Suggest 2000–2024 (long enough for a robust test, avoids the thinnest early years of the data), monthly frequency
- Rationale: dynamic, point-in-time universe avoids survivorship bias, which is the standard critique of naive backtests using "today's" index list

## Strategy Selection

**Long-short 12-1 momentum**, rebalanced monthly:

- At each month-end, look up S&P 500 membership as of that date (most recent membership-change date on or before month-end)
- For each eligible stock with sufficient price history, compute trailing 12-month return, **skipping the most recent month** (the "12-1" convention — avoids short-term reversal effects)
- Rank all eligible stocks by this momentum score
- **Long** the top decile (or top N), **equal-weighted**
- **Short** the bottom decile (or bottom N), equal-weighted
- Hold for one month, then re-rank and rebalance against that month's (possibly changed) membership list

## Backtest Requirements

Compute and report the following, compared against a buy-and-hold benchmark:

- Total/annualized return
- Volatility (annualized standard deviation)
- Sharpe ratio
- Maximum drawdown

**Benchmark definition**: buy-and-hold equal-weighted (or cap-weighted) basket of all current S&P 500 constituents, no momentum-based rebalancing — isolates the effect of the momentum _selection_ rather than just being invested in large-cap US equities.

## Alpha Test (CAPM Regression)

- Regress the strategy's monthly excess returns on the market's monthly excess returns: **R_strategy − R_f = α + β(R_M − R_f) + ε**
- Report: α (coefficient), β (coefficient), standard errors, t-statistics, p-values, and R²
- State clearly whether α is positive and statistically significant (e.g., |t-stat| > ~2, p < 0.05)


## 📈 Performance Results (2000–2024)

### Benchmark Performance Comparison

| Metric | L/S Momentum Strategy | S&P 500 / Market Proxy |
| :--- | :--- | :--- |
| **Annualized Return** | -2.66% | 10.20% |
| **Annualized Volatility** | 24.83% | 17.61% |
| **Sharpe Ratio** | -0.11 | 0.37 |

### CAPM Alpha Regression Results

| Parameter | Estimate |
| :--- | :--- |
| **Annualized Alpha (α)** | 3.06% |
| **Market Beta (β)** | -0.7408 |
| **Alpha t-statistic** | 0.70 |
| **Alpha p-value** | 0.4836 |

> **Decision:** Fail to reject H0. Returns are driven by market exposure and statistical noise rather than genuine, repeatable alpha.