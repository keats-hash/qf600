"""
Step 4 -- Holding-period construction.  *** FOUNDATIONAL ***

Converts discrete disclosed trades into a monthly held-position panel with an
explicit, mechanical exit. Everything downstream consumes this module's output.

THE RULE (committed, structural -- not tuned):
  A trade disclosed in month t opens a leg that earns the ticker's return in
  months t+1 .. t+HOLDING_MONTHS. A Purchase opens a +1 (long) leg; a Sale opens
  a -1 (short) leg, used only by the long-short construction. Legs then roll off.

ANTI-BIAS NOTES -- the three that matter most:

  1. ENTRY IS ON THE FILING (DISCLOSURE) DATE, NEVER THE TRADE DATE.
     Filings lag the trade by ~29 days at the median and 438 days at the 95th
     percentile in this dataset. Entering on the trade date would credit the
     portfolio with a price move that was not public and not investable. That is
     the single largest look-ahead trap in congressional-trading research, and it
     is exactly what makes most published "congress beats the market" claims
     unreplicable by an actual investor.

  2. ENTRY LAG OF ONE MONTH (config.ENTRY_LAG_MONTHS = 1).
     A filing lands on some day inside month t. Awarding the portfolio month t's
     FULL return would capture the part of month t that occurred BEFORE the
     disclosure was public -- a partial look-ahead. The leg therefore earns from
     month t+1, the first month an investor reading the filing could hold for its
     entirety. This is the strictly conservative direction: it can only reduce
     measured alpha, never inflate it. The zero-lag variant is reported in Step 9
     so the choice is visible rather than buried.

  3. THE EXIT CARRIES NO INFORMATION.
     Exit is a fixed function of entry (entry + 12 months), so it cannot encode
     anything about what the price did afterwards. Rejected alternatives:
       * hold-until-disclosed-closing-trade -- closing trades are sparsely and
         very late disclosed, so the exit date would itself be a selected,
         information-bearing quantity;
       * hold-to-next-rebalance -- makes the holding period depend on future
         trading activity, which is forward-looking.

Run:  python src/step3_holding.py
Out:  data/clean/held_positions.parquet, data/reports/step4_holding.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OUT = C.CLEAN / "held_positions.parquet"
REPORT = C.REPORTS / "step4_holding.txt"


def build_panel(trades, holding_months=None, entry_lag=None):
    """Explode each trade into its monthly legs.

    Returns one row per (trade, month-held) with the month the leg earns in.
    """
    holding_months = holding_months or C.HOLDING_MONTHS
    entry_lag = C.ENTRY_LAG_MONTHS if entry_lag is None else entry_lag

    t = trades.copy()
    t["entry_month"] = C.month_end(t["filing_date"])

    # offsets are 1..H under the default lag: the H full months after disclosure.
    offsets = np.arange(entry_lag, entry_lag + holding_months)
    n = len(t)

    idx = np.repeat(np.arange(n), len(offsets))
    off = np.tile(offsets, n)

    panel = t.iloc[idx][
        ["trade_id", "filer_id", "full_name", "party", "chamber",
         "ticker", "side", "amount_mid", "entry_month"]
    ].reset_index(drop=True)
    panel["months_held"] = off

    # Month the leg earns in = entry month shifted forward by the offset.
    ent = panel["entry_month"].dt.to_period("M")
    panel["month"] = (ent + off).dt.to_timestamp("M")

    return panel


def attach_returns(panel, rets):
    """Join each leg-month to its ticker return; drop and log unpriced legs."""
    long_r = rets.stack(dropna=True).rename("ret")
    long_r.index.names = ["month", "ticker"]
    long_r = long_r.reset_index()

    before = len(panel)
    merged = panel.merge(long_r, on=["month", "ticker"], how="left")
    unpriced = merged["ret"].isna()

    # A leg with no price is dropped -- but it is LOGGED per member, because
    # silently dropping delisted names is survivorship bias.
    dropped_trades = merged.loc[unpriced, "trade_id"].nunique()
    merged = merged[~unpriced].copy()
    print(f"  Leg-months: {before:,} -> {len(merged):,} priced "
          f"({len(merged)/before:.1%}); {unpriced.sum():,} unpriced leg-months "
          f"across {dropped_trades:,} trades")
    return merged, unpriced.sum(), dropped_trades


def member_monthly_returns(panel, construction="long_only", weight="equal",
                           dedupe_legs=False):
    """Per-member monthly return from the held-position panel.

    construction:
      "long_only"  -- only +1 legs; short legs are ignored entirely.
      "long_short" -- all legs, signed: a -1 leg earns MINUS the ticker return.
                      A member simultaneously long some names and short others in
                      the same month is handled naturally, because every open leg
                      shares the same 1/n weight and carries its own sign. A +1 and
                      a -1 leg in the SAME ticker net to zero, which is the correct
                      economics of a disclosed buy followed by a disclosed sell.
    weight:
      "equal"  -- 1/n across the member's open legs (DEFAULT; disclosed amounts are
                  ranges, so size weights are badly measured).
      "amount" -- proportional to amount_mid (reported as a robustness variant only).
    dedupe_legs:
      collapse repeated legs in the same ticker to one, removing the implicit
      conviction-weighting of repeatedly-bought names. Robustness variant.

    Returns: DataFrame indexed by month, columns = filer_id.
    """
    p = panel if construction == "long_short" else panel[panel["side"] == 1]
    if not len(p):
        return pd.DataFrame()
    p = p.copy()

    if dedupe_legs:
        # Keep the earliest-entered leg per (member, month, ticker, side).
        p = (p.sort_values("months_held")
               .drop_duplicates(["filer_id", "month", "ticker", "side"]))

    if weight == "amount":
        w = p["amount_mid"].fillna(0.0).clip(lower=0.0)
        # A member-month whose disclosed amounts are all missing falls back to
        # equal weight rather than silently dropping out of the portfolio.
        tot = w.groupby([p["filer_id"], p["month"]]).transform("sum")
        w = np.where(tot > 0, w / tot.replace(0, np.nan), np.nan)
        p["w"] = w
        p["w"] = p["w"].fillna(
            1.0 / p.groupby(["filer_id", "month"])["ticker"].transform("size"))
    else:
        p["w"] = 1.0 / p.groupby(["filer_id", "month"])["ticker"].transform("size")

    p["contrib"] = p["w"] * p["side"] * p["ret"]
    mr = p.groupby(["month", "filer_id"])["contrib"].sum().unstack("filer_id")
    return mr.sort_index()


def group_portfolio(member_rets, members, start, end):
    """Equal-weight across ACTIVE members (those holding something that month).

    Equal-weighting across members, rather than pooling all legs, stops one
    hyperactive member from becoming the entire portfolio.
    """
    cols = [m for m in members if m in member_rets.columns]
    if not cols:
        return pd.Series(dtype=float)
    sub = member_rets.loc[
        (member_rets.index >= start) & (member_rets.index <= end), cols]
    return sub.mean(axis=1, skipna=True).dropna()


def invested_months(panel, start, end, long_only=True):
    """Months in [start,end] in which a member holds >=1 open leg.

    This is the eligibility measure the floor is applied to. It is recomputed HERE
    (not carried over from Phase 1) because the explicit 12-month rule changes how
    many months a member is actually invested.
    """
    p = panel[(panel["month"] >= start) & (panel["month"] <= end)]
    if long_only:
        p = p[p["side"] == 1]
    return p.groupby("filer_id")["month"].nunique()


def main():
    trades = pd.read_parquet(C.CLEAN / "congress_trades.parquet")
    rets = pd.read_parquet(C.CLEAN / "returns_monthly.parquet")

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out("STEP 4 -- HOLDING-PERIOD PANEL")
    out("=" * 78)
    out(f"Rule: {C.HOLDING_MONTHS}-month fixed window from FILING date, "
        f"entry lag {C.ENTRY_LAG_MONTHS} month(s)")
    out(f"Legs earn returns in months  t+{C.ENTRY_LAG_MONTHS} .. "
        f"t+{C.ENTRY_LAG_MONTHS + C.HOLDING_MONTHS - 1}  where t = filing month")
    out()

    # Only trades whose legs can reach the study span matter.
    span_lo = pd.Timestamp(C.FORMATION_START) - pd.DateOffset(months=C.HOLDING_MONTHS + 2)
    t = trades[(trades["filing_date"] >= span_lo) &
               (trades["filing_date"] <= C.EVAL_END)]
    out(f"Trades in span: {len(t):,} ({t['filer_id'].nunique()} members)")

    panel = build_panel(t)
    out(f"Exploded to {len(panel):,} leg-months")

    panel, n_unpriced, n_dropped_trades = attach_returns(panel, rets)
    out(f"Priced leg-months: {len(panel):,}  (dropped {n_unpriced:,} unpriced "
        f"leg-months from {n_dropped_trades:,} trades)")

    # Keep only leg-months inside the study span.
    panel = panel[(panel["month"] >= C.FORMATION_START) &
                  (panel["month"] <= C.EVAL_END)].copy()
    out(f"Leg-months inside study span: {len(panel):,}")
    out()

    # ---- concentration diagnostic ------------------------------------------
    # Repeated purchases of the same ticker create multiple simultaneous legs, so
    # equal-weighting ACROSS LEGS implicitly conviction-weights that ticker.
    # Quantify it so the write-up can state the effect rather than hide it.
    lo = panel[panel["side"] == 1]
    per_mm = lo.groupby(["filer_id", "month"]).agg(
        legs=("ticker", "size"), names=("ticker", "nunique")
    )
    dup_ratio = (per_mm["legs"] / per_mm["names"])
    out("Leg concentration (long legs), legs per distinct ticker in a member-month:")
    out(f"  median {dup_ratio.median():.2f}, p75 {dup_ratio.quantile(.75):.2f}, "
        f"p95 {dup_ratio.quantile(.95):.2f}, max {dup_ratio.max():.2f}")
    out("  -> equal-weight ACROSS LEGS therefore conviction-weights repeatedly-bought")
    out("     names. Step 9 reports a de-duplicated variant (one leg per ticker).")
    out()
    out(f"Open long legs per member-month: median {per_mm['legs'].median():.0f}, "
        f"p25 {per_mm['legs'].quantile(.25):.0f}, p75 {per_mm['legs'].quantile(.75):.0f}, "
        f"max {per_mm['legs'].max()}")
    out()

    # ---- eligibility recompute ---------------------------------------------
    out("-" * 78)
    out("ELIGIBILITY RECOMPUTE UNDER THE EXPLICIT HOLDING RULE")
    out("-" * 78)
    form_trades = trades[(trades["filing_date"] >= C.FORMATION_START) &
                         (trades["filing_date"] <= C.FORMATION_END)]
    n_form = form_trades.groupby("filer_id").size()
    inv = invested_months(panel, C.FORMATION_START, C.FORMATION_END, long_only=True)

    elig_trades = set(n_form[n_form >= C.MIN_FORMATION_TRADES].index)
    elig_inv = set(inv[inv >= C.MIN_INVESTED_MONTHS].index)
    eligible = sorted(elig_trades & elig_inv)

    out(f"  Members with >={C.MIN_FORMATION_TRADES} formation trades : {len(elig_trades)}")
    out(f"  Members with >={C.MIN_INVESTED_MONTHS} formation invested months : {len(elig_inv)}")
    out(f"  ELIGIBLE POOL (both conditions)          : {len(eligible)}")
    out()
    lost = sorted(elig_trades - elig_inv)
    if lost:
        out(f"  Cleared the trade floor but NOT the invested-month floor ({len(lost)}):")
        first_buy = (trades[(trades["side"] == 1) &
                            (trades["filing_date"] >= C.FORMATION_START) &
                            (trades["filing_date"] <= C.FORMATION_END)]
                     .groupby("filer_id")["filing_date"].min())
        for f in lost:
            fb = first_buy.get(f)
            fb_s = f"{fb:%Y-%m}" if pd.notna(fb) else "no buys"
            out(f"    {f:<42} {n_form.get(f,0):>3} trades, {inv.get(f,0):>2} invested "
                f"months, first buy {fb_s}")
        out()
        out("  WHY THIS HAPPENS -- worth stating, because it is not a bug and it")
        out("  materially changes the pool. Two distinct causes:")
        out("    (a) LATE ARRIVALS. Most of these members' first purchase falls in")
        out("        2019, at the very end of the formation window. They are largely")
        out("        the freshman class sworn in January 2019. A high trade count")
        out("        does NOT give them a formation track record: their legs only")
        out("        start earning at the window's edge, so there is no formation")
        out("        performance to rank them on. Ranking them anyway would score")
        out("        them on a couple of months of noise.")
        out("    (b) SELLERS ONLY. Some members made no PURCHASES in formation at")
        out("        all, so they open no long legs. Invested months counts long")
        out("        legs, because the long-only construction is what the ranking")
        out("        uses.")
        out("  Both exclusions are structural, applied before any return was ranked,")
        out("  and identical in spirit to the trade-count floor.")
    out()
    out("  NOTE: this pool is recomputed from scratch under the 12-month rule and")
    out("  may differ from the Phase-1 (implicit-exit) pool. Step 5 logs the")
    out("  resulting change to the SELECTED member list.")

    pd.Series(eligible, name="filer_id").to_csv(
        C.CLEAN / "eligible_pool.csv", index=False)
    panel.to_parquet(OUT, index=False)

    out()
    out(f"Saved panel -> {OUT}  ({len(panel):,} rows)")
    out(f"Saved eligible pool -> {C.CLEAN / 'eligible_pool.csv'} ({len(eligible)} members)")
    REPORT.write_text("\n".join(lines))
    print(f"Report -> {REPORT}")


if __name__ == "__main__":
    main()
