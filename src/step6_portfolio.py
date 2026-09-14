"""
Step 7 -- Portfolio return series.

Builds the evaluation-window monthly return series for every group, in both
constructions, and saves them for the regression step.

CONSTRUCTIONS (both always reported; long-only leads the write-up by decision of
2026-09-14, because it is the construction directly comparable to the Phase-1
baseline):

  long_only  -- equal-weight across each member's open LONG legs, then
                equal-weight across active members.
  long_short -- long the open purchase legs, short the open sale legs, netted
                within the member-month. If members have genuine directional
                timing, beta should fall toward zero here. If beta STAYS HIGH,
                the "buys" were simply market exposure wearing a stock-picking
                costume -- which is itself the finding.

WEIGHTING: equal-weight is the default because disclosed amounts are RANGES, so
midpoint sizes are badly measured. Midpoint-size weighting is a reported
robustness variant (Step 9), never the headline.

Run:  python src/step6_portfolio.py
Out:  data/clean/portfolio_returns.parquet, data/reports/step7_portfolios.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from step3_holding import (build_panel, attach_returns, member_monthly_returns,
                           group_portfolio)

REPORT = C.REPORTS / "step7_portfolios.txt"


def trackB_panel(trades_insec, rets):
    """Held-position panel built ONLY from on-committee, in-sector trades."""
    p = build_panel(trades_insec)
    p, _, _ = attach_returns(p, rets)
    return p[(p["month"] >= C.EVAL_START) & (p["month"] <= C.EVAL_END)].copy()


def series_for(panel, members, construction, weight="equal", dedupe=False):
    mr = member_monthly_returns(panel, construction=construction,
                                weight=weight, dedupe_legs=dedupe)
    if mr.empty:
        return pd.Series(dtype=float)
    return group_portfolio(mr, members, C.EVAL_START, C.EVAL_END)


def build_all(weight="equal", dedupe=False, entry_lag=None, quiet=False):
    """Every group x construction series. Reused by Step 9 for variants."""
    rets = pd.read_parquet(C.CLEAN / "returns_monthly.parquet")
    selected = pd.read_csv(C.CLEAN / "trackA_selected.csv")["filer_id"].tolist()
    pool = pd.read_csv(C.CLEAN / "eligible_pool.csv")["filer_id"].tolist()

    if entry_lag is None:
        panel = pd.read_parquet(C.CLEAN / "held_positions.parquet")
    else:
        # Robustness only: rebuild the panel under a different entry lag.
        trades = pd.read_parquet(C.CLEAN / "congress_trades.parquet")
        t = trades[trades["filing_date"] <= C.EVAL_END]
        panel = build_panel(t, entry_lag=entry_lag)
        panel, _, _ = attach_returns(panel, rets)
        panel = panel[(panel["month"] >= C.FORMATION_START) &
                      (panel["month"] <= C.EVAL_END)]

    out = {}
    for con in ("long_only", "long_short"):
        out[("trackA_selected", con)] = series_for(panel, selected, con, weight, dedupe)
        # CONTROL GROUP: the members the ranking REJECTED. Reported so the selected
        # group's alpha can be read against the pool it was drawn from, rather than
        # against nothing.
        out[("trackA_pool_all", con)] = series_for(panel, pool, con, weight, dedupe)
        rejected = [m for m in pool if m not in selected]
        out[("trackA_rejected", con)] = series_for(panel, rejected, con, weight, dedupe)

    # ---- Track B -----------------------------------------------------------
    bpath = C.CLEAN / "trackB_groups.parquet"
    if bpath.exists():
        insec = pd.read_parquet(bpath)
        bpanel = trackB_panel(insec, rets)
        if len(bpanel):
            # Per-committee groups: a trade may legitimately appear under TWO
            # committees (a member sitting on two panels that both oversee the
            # ticker's sector), so this merge can duplicate rows.
            key_of = insec.drop_duplicates(
                ["trade_id", "committee_key"])[["trade_id", "committee_key"]]
            bp = bpanel.merge(key_of, on="trade_id", how="left")
            for ck, sub in bp.groupby("committee_key"):
                mem = sorted(sub["filer_id"].unique())
                for con in ("long_only", "long_short"):
                    out[(f"trackB_{ck}", con)] = series_for(sub, mem, con, weight, dedupe)

            # COMBINED portfolio uses the UNDUPLICATED panel. Using `bp` here would
            # double-weight exactly those trades that fall under two committees,
            # silently over-weighting the most heavily-overseen positions.
            allmem = sorted(bpanel["filer_id"].unique())
            for con in ("long_only", "long_short"):
                out[("trackB_combined", con)] = series_for(bpanel, allmem, con,
                                                           weight, dedupe)

    return {k: v for k, v in out.items() if len(v) > 0}


def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out("STEP 7 -- PORTFOLIO RETURN SERIES (evaluation window)")
    out("=" * 78)
    out(f"Window: {C.EVAL_START[:7]} .. {C.EVAL_END[:7]}   "
        f"Weighting: equal (default)   Headline: {C.HEADLINE_CONSTRUCTION}")
    out()

    series = build_all()
    df = pd.DataFrame({f"{g}|{c}": s for (g, c), s in series.items()})
    df.index.name = "month"
    df.to_parquet(C.CLEAN / "portfolio_returns.parquet")

    out(f"{'group':<28} {'construction':<12} {'N':>4} {'mean%/yr':>9} "
        f"{'sd%/yr':>8} {'min':>8} {'max':>8}")
    out("-" * 78)
    for (g, c), s in sorted(series.items()):
        out(f"{g:<28} {c:<12} {len(s):>4} {s.mean()*12*100:>8.1f}% "
            f"{s.std()*np.sqrt(12)*100:>7.1f}% {s.min()*100:>7.1f}% {s.max()*100:>7.1f}%")

    out()
    out("Coverage check -- months with no active member are simply absent, so a")
    out("group with N < 60 was not holding anything in some evaluation months:")
    for (g, c), s in sorted(series.items()):
        if len(s) < 60 and c == "long_only":
            missing = 60 - len(s)
            out(f"  {g:<28} {len(s):>3}/60 months ({missing} with no open position)")

    out()
    out(f"Saved -> {C.CLEAN / 'portfolio_returns.parquet'}  "
        f"({df.shape[1]} series x {df.shape[0]} months)")
    REPORT.write_text("\n".join(lines))
    print(f"Report -> {REPORT}")


if __name__ == "__main__":
    main()
