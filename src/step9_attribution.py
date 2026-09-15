"""
Step 10 -- Holdings and P&L attribution (DESCRIPTIVE ONLY).

Answers the questions the regressions cannot: what was actually held, what drove
the return, and where the portfolio was fragile.

*** READ THIS BEFORE USING THE OUTPUT ***
This module is DESCRIPTIVE. It runs AFTER selection and the regressions are fixed,
and nothing it produces may feed back into the split, the holding rule, the
weighting, the sector map, or the member list. Attribution is computed with full
knowledge of evaluation-window returns -- that is the whole point of it -- which
means any decision taken on the basis of these tables would be selection on the
outcome, and would silently void the out-of-sample design the rest of the
pipeline exists to protect.

Legitimate uses:  understanding the portfolio, reporting concentration risk,
                  explaining what the alpha estimate is actually made of.
Illegitimate use: "NVDA drove the result, let's exclude it and re-run."

ATTRIBUTION MATHS
The portfolio return in month t is, by construction,

    R_t = (1/A_t) * sum_m [ (1/L_mt) * sum_legs (side * ret) ]

where A_t is the number of active members and L_mt member m's open legs. So every
leg's contribution to month t is exactly

    c = (1 / A_t) * (1 / L_mt) * side * ret

and these sum to R_t with no residual. Summing c over months gives an ADDITIVE
attribution: contributions total the SUM of monthly returns, not the compounded
return. Arithmetic attribution is the standard choice here because it is exact and
decomposable; the compounded figure is reported separately for reference.

Run:  python src/step9_attribution.py
Out:  data/reports/step10_attribution.txt, data/clean/attribution_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

REPORT = C.REPORTS / "step10_attribution.txt"


def leg_contributions(panel, members, construction="long_only"):
    """Per-leg contribution to each month's portfolio return.

    Replicates step6_portfolio's two-stage equal weighting exactly:
    equal across ACTIVE MEMBERS, then equal across that member's OPEN LEGS.
    """
    p = panel[panel["filer_id"].isin(members)].copy()
    p = p[(p["month"] >= C.EVAL_START) & (p["month"] <= C.EVAL_END)]
    if construction == "long_only":
        p = p[p["side"] == 1]
    if not len(p):
        return pd.DataFrame()

    # Stage 1: 1 / (member's open legs that month)
    p["legs_m"] = p.groupby(["filer_id", "month"])["ticker"].transform("size")
    # Stage 2: 1 / (active members that month)
    active = p.groupby("month")["filer_id"].transform("nunique")
    p["weight"] = (1.0 / active) * (1.0 / p["legs_m"])
    p["contrib"] = p["weight"] * p["side"] * p["ret"]
    return p


def concentration(p, out, label):
    """Concentration diagnostics -- the portfolio's main structural weakness."""
    out(f"\n  CONCENTRATION -- {label}")
    # Average monthly Herfindahl over tickers, and its inverse (effective names).
    hhi, eff, topw, n_names = [], [], [], []
    for m, g in p.groupby("month"):
        w = g.groupby("ticker")["weight"].sum()
        w = w / w.sum()
        h = (w ** 2).sum()
        hhi.append(h)
        eff.append(1 / h)
        topw.append(w.max())
        n_names.append(len(w))
    out(f"    distinct names held : median {np.median(n_names):.0f}, "
        f"min {min(n_names)}, max {max(n_names)}")
    out(f"    EFFECTIVE names (1/HHI): median {np.median(eff):.1f}  "
        f"-- the portfolio behaves like this many equally-weighted positions")
    out(f"    largest single weight : median {np.median(topw):.1%}, "
        f"worst month {max(topw):.1%}")
    out( "    -> a large gap between distinct and EFFECTIVE names means the")
    out( "       portfolio is far less diversified than a position count suggests.")


def attribute(panel, members, label, out, construction="long_only", top=18):
    p = leg_contributions(panel, members, construction)
    if not len(p):
        out(f"\n{label}: no positions")
        return None

    monthly = p.groupby("month")["contrib"].sum()
    total_sum = monthly.sum()
    compounded = (1 + monthly).prod() - 1

    out("\n" + "=" * 92)
    out(f"{label}  ({construction})")
    out("=" * 92)
    out(f"  months: {len(monthly)}   sum of monthly returns: {total_sum:.1%}   "
        f"compounded: {compounded:.1%}")
    out(f"  distinct tickers ever held: {p['ticker'].nunique():,}   "
        f"leg-months: {len(p):,}")

    # ---- per-ticker attribution -------------------------------------------
    g = p.groupby("ticker").agg(
        contrib=("contrib", "sum"),
        avg_w=("weight", "mean"),
        months=("month", "nunique"),
        members=("filer_id", "nunique"),
        mean_ret=("ret", "mean"),
    )
    g["share_of_total"] = g["contrib"] / total_sum if total_sum else np.nan
    g = g.sort_values("contrib", ascending=False)

    out(f"\n  TOP {top} CONTRIBUTORS  (sum of monthly weight x return)")
    out(f"    {'ticker':<8}{'contrib':>9}{'% of tot':>10}{'avg wt':>9}"
        f"{'months':>8}{'members':>9}")
    for t, r in g.head(top).iterrows():
        out(f"    {t:<8}{r['contrib']*100:>8.2f}%{r['share_of_total']*100:>9.1f}%"
            f"{r['avg_w']*100:>8.2f}%{int(r['months']):>8}{int(r['members']):>9}")

    out(f"\n  BOTTOM {top} CONTRIBUTORS (the drag)")
    out(f"    {'ticker':<8}{'contrib':>9}{'% of tot':>10}{'avg wt':>9}"
        f"{'months':>8}{'members':>9}")
    for t, r in g.tail(top).iloc[::-1].iterrows():
        out(f"    {t:<8}{r['contrib']*100:>8.2f}%{r['share_of_total']*100:>9.1f}%"
            f"{r['avg_w']*100:>8.2f}%{int(r['months']):>8}{int(r['members']):>9}")

    # ---- how concentrated is the P&L itself? ------------------------------
    pos = g[g["contrib"] > 0]["contrib"].sum()
    neg = g[g["contrib"] < 0]["contrib"].sum()
    out(f"\n  P&L STRUCTURE")
    out(f"    gross winners {pos*100:+.1f}%   gross losers {neg*100:+.1f}%   "
        f"net {total_sum*100:+.1f}%")
    out(f"    names with positive contribution: {(g['contrib']>0).sum():,} of "
        f"{len(g):,} ({(g['contrib']>0).mean():.0%})")
    for k in (1, 3, 5, 10):
        if len(g) >= k:
            out(f"    top {k:>2} names = {g['contrib'].head(k).sum()/total_sum*100:>5.1f}% "
                f"of total return" if total_sum else "")
    out( "    -> if a handful of names account for most of the return, the alpha")
    out( "       estimate rests on those names, not on a repeatable process.")

    concentration(p, out, label)

    # ---- member attribution ------------------------------------------------
    mg = p.groupby("filer_id").agg(contrib=("contrib", "sum"),
                                   months=("month", "nunique"))
    mg = mg.sort_values("contrib", ascending=False)
    out(f"\n  BY MEMBER")
    out(f"    {'member':<34}{'contrib':>9}{'% of tot':>10}{'months':>8}")
    for f, r in mg.iterrows():
        sh = r["contrib"] / total_sum * 100 if total_sum else np.nan
        out(f"    {f:<34}{r['contrib']*100:>8.2f}%{sh:>9.1f}%{int(r['months']):>8}")

    # ---- positions over time ----------------------------------------------
    per_month = p.groupby("month").agg(names=("ticker", "nunique"),
                                       legs=("ticker", "size"),
                                       members=("filer_id", "nunique"))
    out(f"\n  POSITION COUNT OVER TIME (year-end snapshots)")
    out(f"    {'month':<10}{'members':>9}{'legs':>7}{'names':>7}")
    for m, r in per_month.iterrows():
        if m.month == 12 or m == per_month.index[0]:
            out(f"    {m:%Y-%m}   {int(r['members']):>7}{int(r['legs']):>7}"
                f"{int(r['names']):>7}")

    g.to_csv(C.CLEAN / f"attribution_{label.split()[0].lower()}_{construction}.csv")
    return g


def sector_mix(panel, members, out, label):
    """Sector composition, using the cached ticker->sector table."""
    if not C.SECTOR_CACHE.exists():
        return
    from step5_committee import to_gics
    sec = pd.read_csv(C.SECTOR_CACHE)
    smap = dict(zip(sec["ticker"], sec["sector"]))
    p = leg_contributions(panel, members, "long_only")
    if not len(p):
        return
    p["sector"] = p["ticker"].map(smap).map(to_gics)
    tot = p["contrib"].sum()
    g = p.groupby("sector").agg(weight=("weight", "sum"),
                                contrib=("contrib", "sum"))
    g["avg_weight"] = g["weight"] / p["month"].nunique()
    g = g.sort_values("contrib", ascending=False)
    out(f"\n  SECTOR MIX -- {label}")
    out(f"    {'sector':<26}{'avg wt':>9}{'contrib':>10}{'% of tot':>10}")
    for s, r in g.iterrows():
        sh = r["contrib"] / tot * 100 if tot else np.nan
        out(f"    {str(s):<26}{r['avg_weight']*100:>8.1f}%{r['contrib']*100:>9.2f}%"
            f"{sh:>9.1f}%")
    out( "    NOTE: sector labels are only used to DESCRIBE the portfolio here.")
    out( "    Track B's sector restriction is applied upstream in step5.")


def main():
    panel = pd.read_parquet(C.CLEAN / "held_positions.parquet")
    selected = pd.read_csv(C.CLEAN / "trackA_selected.csv")["filer_id"].tolist()

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 92)
    out("STEP 10 -- HOLDINGS AND P&L ATTRIBUTION")
    out("=" * 92)
    out("DESCRIPTIVE ONLY. Computed with full knowledge of evaluation-window")
    out("returns, so nothing here may feed back into selection, the split, the")
    out("holding rule or the weighting. Doing so would be selection on the")
    out("outcome and would void the out-of-sample design.")
    out(f"Window {C.EVAL_START[:7]} .. {C.EVAL_END[:7]}, equal-weight, long-only.")

    attribute(panel, selected, "TrackA_selected", out)
    sector_mix(panel, selected, out, "Track A selected")

    # Track B combined, built from the sector-restricted panel.
    bpath = C.CLEAN / "trackB_groups.parquet"
    if bpath.exists():
        from step6_portfolio import trackB_panel
        rets = pd.read_parquet(C.CLEAN / "returns_monthly.parquet")
        insec = pd.read_parquet(bpath).drop_duplicates("trade_id")
        bp = trackB_panel(insec, rets)
        if len(bp):
            attribute(bp, sorted(bp["filer_id"].unique()), "TrackB_combined", out)
            sector_mix(bp, sorted(bp["filer_id"].unique()), out, "Track B combined")

    REPORT.write_text("\n".join(lines))
    print(f"\nReport -> {REPORT}")


if __name__ == "__main__":
    main()
