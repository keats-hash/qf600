"""
Step 5 -- Track A selection (performance / formation ranking).

Ranks the eligible pool by long-only mean monthly return over the FORMATION window
and selects the top quartile. The question this track answers is whether past
winners PERSIST (skill) or were simply LUCKY (noise): if formation performance
carries no information, the selected group's evaluation-window alpha should be
indistinguishable from zero.

ANTI-BIAS NOTE -- this is the step where data snooping would enter if it were
going to. Three guards, all enforced in code rather than by good intentions:

  1. Ranking uses ONLY leg-months whose return month falls inside the formation
     window. A hard assertion at the end re-checks that no evaluation month
     touched the ranking, and raises if one did.
  2. The eligibility floor (>=10 formation trades, >=12 invested months) and the
     top-quartile cutoff were fixed in Step 2 from COUNTS ONLY, before any return
     existed. They are not adjusted here to produce a nicer group.
  3. The FULL ranked pool is logged, not just the winners. The rejected members
     are the control group; hiding them would make the selection unfalsifiable.

Run:  python src/step4_formation.py
Out:  data/clean/trackA_selected.csv, data/reports/step5_formation.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from step3_holding import member_monthly_returns

REPORT = C.REPORTS / "step5_formation.txt"


def main():
    panel = pd.read_parquet(C.CLEAN / "held_positions.parquet")
    eligible = pd.read_csv(C.CLEAN / "eligible_pool.csv")["filer_id"].tolist()
    trades = pd.read_parquet(C.CLEAN / "congress_trades.parquet")

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out("STEP 5 -- TRACK A SELECTION (formation-window performance ranking)")
    out("=" * 78)
    out(f"Formation window : {C.FORMATION_START[:7]} .. {C.FORMATION_END[:7]}")
    out(f"Evaluation window: {C.EVAL_START[:7]} .. {C.EVAL_END[:7]}  "
        f"(QUARANTINED from this step)")
    out(f"Eligible pool    : {len(eligible)} members")
    out()

    # *** QUARANTINE *** -- slice the panel to formation months BEFORE any return
    # is computed. Nothing after this line can see an evaluation month.
    form_panel = panel[(panel["month"] >= C.FORMATION_START) &
                       (panel["month"] <= C.FORMATION_END)]
    assert form_panel["month"].max() <= pd.Timestamp(C.FORMATION_END), \
        "Formation panel leaked past the formation window"

    mr = member_monthly_returns(form_panel, construction="long_only", weight="equal")
    mr = mr[[c for c in eligible if c in mr.columns]]

    stats = pd.DataFrame({
        "mean_monthly_ret": mr.mean(skipna=True),
        "months": mr.notna().sum(),
        "sd_monthly": mr.std(skipna=True),
    })
    stats["ann_ret"] = stats["mean_monthly_ret"] * C.ANNUALISE
    n_form = trades[(trades["filing_date"] >= C.FORMATION_START) &
                    (trades["filing_date"] <= C.FORMATION_END)].groupby("filer_id").size()
    stats["form_trades"] = n_form.reindex(stats.index)
    meta = trades.drop_duplicates("filer_id").set_index("filer_id")
    stats["party"] = meta["party"].reindex(stats.index)
    stats["chamber"] = meta["chamber"].reindex(stats.index)
    stats["name"] = meta["full_name"].reindex(stats.index)

    stats = stats.sort_values("mean_monthly_ret", ascending=False)

    # Top quartile of the pool.
    n_sel = max(1, int(round(len(stats) * C.TOP_QUANTILE)))
    selected = stats.index[:n_sel].tolist()

    out("-" * 78)
    out("FULL RANKED POOL (rejected members shown too -- they are the control group)")
    out("-" * 78)
    out(f"{'rank':>4} {'sel':>4} {'filer_id':<40} {'ann%':>8} {'mo':>4} "
        f"{'trades':>7} {'party':>6}")
    for i, (fid, r) in enumerate(stats.iterrows(), 1):
        mark = "***" if fid in selected else ""
        out(f"{i:>4} {mark:>4} {fid:<40} {r['ann_ret']*100:>7.1f}% "
            f"{int(r['months']):>4} {int(r['form_trades']):>7} {str(r['party']):>6}")

    out()
    out("-" * 78)
    out(f"SELECTED: top {C.TOP_QUANTILE:.0%} = {n_sel} of {len(stats)} members")
    out("-" * 78)
    sel_df = stats.loc[selected]
    for fid, r in sel_df.iterrows():
        out(f"  {fid:<40} {r['name']:<28} {str(r['party']):>2} "
            f"{r['ann_ret']*100:>7.1f}%/yr formation")
    out()
    out(f"  Party split: {sel_df['party'].value_counts().to_dict()}")
    out(f"  Chamber    : {sel_df['chamber'].value_counts().to_dict()}")
    out(f"  Formation mean return of selected group: "
        f"{sel_df['mean_monthly_ret'].mean()*12:.2%}/yr vs pool "
        f"{stats['mean_monthly_ret'].mean()*12:.2%}/yr")
    out()
    out("  The selected group outperformed IN FORMATION BY CONSTRUCTION. That number")
    out("  is not evidence of anything -- it is the selection itself. The test is")
    out("  whether it persists into the evaluation window (Steps 7-8).")

    # ---- Phase-1 comparison -------------------------------------------------
    phase1 = ["house_bradleys_schneider", "house_lamar_smith", "house_bob_gibbs",
              "house_katherinem_clark", "house_dwight_evans", "house_pete_sessions",
              "senate_william_cassidy"]
    out()
    out("-" * 78)
    out("CHANGE VS PHASE-1 SELECTED LIST (required log -- the holding rule recomputed")
    out("invested months, so the pool and therefore the selection may differ)")
    out("-" * 78)
    out(f"  Phase 1 ({len(phase1)} members, implicit-exit long-only construction):")
    for f in phase1:
        out(f"    {f}")
    added = [f for f in selected if f not in phase1]
    dropped = [f for f in phase1 if f not in selected]
    out(f"  Now ({len(selected)} members, explicit 12-month rule):")
    for f in selected:
        out(f"    {f}")
    out()
    out(f"  ADDED vs Phase 1   ({len(added)}): {added if added else 'none'}")
    out(f"  DROPPED vs Phase 1 ({len(dropped)}): {dropped if dropped else 'none'}")
    out(f"  Overlap: {len(set(selected) & set(phase1))}/{len(phase1)}")
    out("  Downstream deck/summary MUST quote the 'Now' list, not the Phase-1 list.")

    # ---- hard quarantine assertion -----------------------------------------
    assert form_panel["month"].max() <= pd.Timestamp(C.FORMATION_END)
    assert form_panel["month"].min() >= pd.Timestamp(C.FORMATION_START)
    leaked = mr.index[(mr.index > pd.Timestamp(C.FORMATION_END))]
    assert len(leaked) == 0, f"Evaluation months leaked into selection: {list(leaked)}"
    out()
    out("  ASSERTION PASSED: no evaluation-window month entered the selection.")
    out(f"  Ranking used months {mr.index.min():%Y-%m} .. {mr.index.max():%Y-%m} only.")

    stats.to_csv(C.CLEAN / "trackA_ranked_pool.csv")
    pd.Series(selected, name="filer_id").to_csv(
        C.CLEAN / "trackA_selected.csv", index=False)
    out()
    out(f"Saved -> {C.CLEAN / 'trackA_selected.csv'} and trackA_ranked_pool.csv")
    REPORT.write_text("\n".join(lines))
    print(f"Report -> {REPORT}")


if __name__ == "__main__":
    main()
