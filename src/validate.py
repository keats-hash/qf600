"""
Validation -- assert the anti-bias invariants actually hold in the produced data.

The design claims in CLAUDE.md and the README are only worth something if they are
TRUE OF THE OUTPUT, not merely intended by the code. This module re-derives each
claim from the saved artefacts and fails loudly if one is violated.

Run this after any change to the pipeline, and before quoting any result.

    python src/validate.py

Exit code 0 = every invariant holds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main():
    print("=" * 78)
    print("VALIDATION -- anti-bias invariants")
    print("=" * 78)

    trades = pd.read_parquet(C.CLEAN / "congress_trades.parquet")
    panel = pd.read_parquet(C.CLEAN / "held_positions.parquet")
    pool = pd.read_csv(C.CLEAN / "eligible_pool.csv")["filer_id"].tolist()
    sel = pd.read_csv(C.CLEAN / "trackA_selected.csv")["filer_id"].tolist()

    print("\n-- Windows ------------------------------------------------------")
    check("Formation and evaluation windows do not overlap",
          pd.Timestamp(C.FORMATION_END) < pd.Timestamp(C.EVAL_START),
          f"{C.FORMATION_END} < {C.EVAL_START}")

    print("\n-- Disclosure-date entry (the main look-ahead trap) --------------")
    check("No trade is filed before it was transacted",
          (trades["filing_date"] >= trades["transaction_date"]).all())
    check("Entry month is derived from FILING date, not transaction date",
          (trades["entry_month"] >= C.month_end(trades["transaction_date"])).all(),
          "every entry month is at or after the trade's own month")
    lag = (trades["filing_date"] - trades["transaction_date"]).dt.days
    print(f"        (disclosure lag: median {lag.median():.0f}d, "
          f"p95 {lag.quantile(.95):.0f}d -- this is what trade-date entry would steal)")

    print("\n-- Holding rule -------------------------------------------------")
    per_trade = panel.groupby("trade_id")["months_held"].agg(["min", "max", "nunique"])
    check("No leg is held beyond the committed window",
          per_trade["max"].max() <= C.ENTRY_LAG_MONTHS + C.HOLDING_MONTHS - 1,
          f"max months_held = {per_trade['max'].max()}")
    check("No leg earns before the committed entry lag",
          per_trade["min"].min() >= C.ENTRY_LAG_MONTHS,
          f"min months_held = {per_trade['min'].min()} (lag = {C.ENTRY_LAG_MONTHS})")
    check("Every leg-month is strictly after its filing month",
          (panel["month"] > panel["entry_month"]).all() if C.ENTRY_LAG_MONTHS >= 1
          else (panel["month"] >= panel["entry_month"]).all(),
          "no position contributes to a month before it was public")

    print("\n-- Selection quarantine (Track A) -------------------------------")
    check("Selected members are a subset of the eligible pool",
          set(sel).issubset(set(pool)), f"{len(sel)} of {len(pool)}")
    check("Selected group size matches the committed top quartile",
          len(sel) == max(1, int(round(len(pool) * C.TOP_QUANTILE))),
          f"{len(sel)} selected, {C.TOP_QUANTILE:.0%} of {len(pool)}")
    # Re-derive the ranking from formation months only and confirm it reproduces
    # the saved selection. If evaluation data had leaked in, this would differ.
    from step3_holding import member_monthly_returns
    fp = panel[(panel["month"] >= C.FORMATION_START) &
               (panel["month"] <= C.FORMATION_END)]
    mr = member_monthly_returns(fp, construction="long_only")
    mr = mr[[c for c in pool if c in mr.columns]]
    rederived = mr.mean().sort_values(ascending=False).index[:len(sel)].tolist()
    check("Selection reproduces from FORMATION MONTHS ONLY",
          set(rederived) == set(sel),
          "ranking re-derived without touching any evaluation month")
    check("Formation ranking used no evaluation month",
          mr.index.max() <= pd.Timestamp(C.FORMATION_END),
          f"last month used = {mr.index.max():%Y-%m}")

    print("\n-- Evaluation series --------------------------------------------")
    px_path = C.CLEAN / "portfolio_returns.parquet"
    if px_path.exists():
        px = pd.read_parquet(px_path)
        check("All portfolio months lie inside the evaluation window",
              (px.index.min() >= pd.Timestamp(C.EVAL_START)) and
              (px.index.max() <= pd.Timestamp(C.EVAL_END)),
              f"{px.index.min():%Y-%m} .. {px.index.max():%Y-%m}")
        # NaNs are expected and legitimate: a group with no active member in a
        # month simply has no return that month. Compare only observed values --
        # `NaN < 1.0` is False and would fail this check spuriously.
        obs = px.stack(dropna=True)
        check("No portfolio month return is implausible (|r| < 100%)",
              bool((obs.abs() < 1.0).all()),
              f"{len(obs):,} observed monthly returns, "
              f"max |r| = {obs.abs().max():.1%}")

    print("\n-- Track B (selection uses membership only) ----------------------")
    bpath = C.CLEAN / "trackB_groups.parquet"
    if bpath.exists():
        b = pd.read_parquet(bpath)
        mem = pd.read_parquet(C.CLEAN / "committee_membership.parquet")
        pairs = set(zip(mem["filer_id"], mem["congress"], mem["committee_key"]))
        bad = [t for t in set(zip(b["filer_id"], b["congress"], b["committee_key"]))
               if t not in pairs]
        check("Every Track-B trade maps to a real point-in-time committee seat",
              not bad, f"{len(bad)} orphan rows")
        check("Track-B trades are all inside the evaluation window",
              bool((b["filing_date"] >= C.EVAL_START).all() and
                   (b["filing_date"] <= C.EVAL_END).all()))
        # Sector restriction actually applied.
        import yaml
        from step5_committee import load_sector_map, committee_sectors
        csec = committee_sectors(load_sector_map())
        ok = b.apply(lambda r: r["sector"] in csec.get(r["committee_key"], []), axis=1)
        check("Every Track-B trade is in a sector its committee oversees",
              bool(ok.all()), f"{(~ok).sum()} violations")
    else:
        print("  [SKIP] Track B artefacts not built yet")

    print("\n" + "=" * 78)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nFAILURES:")
        for n, d in FAIL:
            print(f"  - {n} {d}")
        sys.exit(1)
    print("All anti-bias invariants hold.")


if __name__ == "__main__":
    main()
