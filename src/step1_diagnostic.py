"""
Step 2 -- Trade-count diagnostic.

Purpose: fix the formation/evaluation split and the eligibility floor BEFORE any
return or alpha is computed.

ANTI-BIAS NOTE (the most important one in the project): this module reads ONLY
counts and dates. It never loads prices, never computes a return, and never ranks
a member by performance. That is what makes the split legitimate -- it was chosen
for statistical power (do we have enough members with enough trades?), not because
it produced a flattering alpha. Once committed, the split is frozen in config.py
and Step 9 tests robustness to it rather than re-choosing it.

Windows are measured on FILING date, consistent with the disclosure-date entry rule.

Run:  python src/step1_diagnostic.py
Out:  data/reports/step2_diagnostic.txt
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# Candidate boundaries examined. Only counts/dates inform the choice.
CANDIDATES = [
    ("A", "2014-01-01", "2018-12-31", "2019-01-01", "2024-12-31"),
    ("B", "2014-01-01", "2019-12-31", "2020-01-01", "2024-12-31"),  # committed
    ("C", "2014-01-01", "2020-12-31", "2021-01-01", "2024-12-31"),
]


def window(df, start, end):
    return df[(df["filing_date"] >= start) & (df["filing_date"] <= end)]


def describe(df, label, out):
    """Print count/date structure of one window. No returns are touched."""
    n_months = (
        pd.period_range(df["filing_date"].min(), df["filing_date"].max(), freq="M").size
        if len(df) else 0
    )
    per = df.groupby("filer_id").size().sort_values(ascending=False)
    out(f"  {label}")
    out(f"    trades            : {len(df):,}")
    out(f"    distinct members  : {per.size:,}")
    out(f"    months spanned    : {n_months}")
    if per.size:
        out(f"    trades/member     : min {per.min()}, p25 {per.quantile(.25):.0f}, "
            f"median {per.median():.0f}, p75 {per.quantile(.75):.0f}, max {per.max()}")
        for k in (5, 10, 20):
            n = (per >= k).sum()
            out(f"    members with >={k:<3}   : {n:>3}  ({n/per.size:5.1%} of window members)")
        party = (
            df.drop_duplicates("filer_id")["party"].value_counts().to_dict()
        )
        out(f"    party (members)   : {party}")
    return set(per[per >= C.MIN_FORMATION_TRADES].index), per


def main():
    df = pd.read_parquet(C.CLEAN / "congress_trades.parquet")

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out("STEP 2 -- TRADE-COUNT DIAGNOSTIC  (counts and dates ONLY, never returns)")
    out("=" * 78)
    out(f"Input: {len(df):,} filtered trades, {df['filer_id'].nunique()} members, "
        f"filing dates {df['filing_date'].min():%Y-%m} .. {df['filing_date'].max():%Y-%m}")
    out()
    out("Coverage by filing year (why the early years are unusable):")
    yr = df.groupby(df["filing_date"].dt.year).agg(
        trades=("trade_id", "size"), members=("filer_id", "nunique")
    )
    for y, r in yr.iterrows():
        bar = "#" * int(r["trades"] / max(yr["trades"].max(), 1) * 46)
        out(f"    {y}  {r['trades']:>6,} trades  {r['members']:>3} members  {bar}")

    for tag, fs, fe, es, ee in CANDIDATES:
        out()
        out("-" * 78)
        out(f"CANDIDATE SPLIT {tag}:  formation {fs[:7]}..{fe[:7]}   "
            f"evaluation {es[:7]}..{ee[:7]}")
        out("-" * 78)
        form = window(df, fs, fe)
        ev = window(df, es, ee)
        elig, form_per = describe(form, "FORMATION window", out)
        _, _ = describe(ev, "EVALUATION window", out)

        # Evaluation months available for the regression. N drives all the power.
        n_eval_months = (
            pd.period_range(es, ee, freq="M").size
        )
        out(f"  EVALUATION months available for regression: {n_eval_months}")

        # Retention: of the members eligible on formation counts, how many still
        # trade in evaluation? A selected member who stops trading contributes
        # nothing, so this is the binding power constraint.
        ev_members = set(ev["filer_id"].unique())
        retained = elig & ev_members
        out(f"  RETENTION: {len(elig)} members clear >={C.MIN_FORMATION_TRADES} formation "
            f"trades; {len(retained)} of them ({len(retained)/max(len(elig),1):.0%}) "
            f"also trade in evaluation")
        q = max(1, round(len(retained) * C.TOP_QUANTILE))
        out(f"  -> top quartile of a {len(retained)}-member pool = ~{q} selected members")

    out()
    out("=" * 78)
    out("COMMITTED DECISION (frozen in config.py; do not re-tune)")
    out("=" * 78)
    out(f"  Split      : formation {C.FORMATION_START[:7]}..{C.FORMATION_END[:7]}, "
        f"evaluation {C.EVAL_START[:7]}..{C.EVAL_END[:7]}   [candidate B]")
    out(f"  Rationale  : disclosure coverage is effectively empty before 2015 and thin")
    out( "               through 2016. Candidate A leaves too few members clearing the")
    out( "               trade floor; candidate C shortens evaluation to 48 months and")
    out( "               pushes formation into the COVID regime. B maximises the size of")
    out( "               the eligible pool while keeping a full 60-month evaluation")
    out( "               window. Chosen on COUNTS ONLY -- no return was computed.")
    out(f"  Floor      : >={C.MIN_FORMATION_TRADES} formation trades AND "
        f">={C.MIN_INVESTED_MONTHS} invested months (invested months recomputed in Step 4)")
    out(f"  Selection  : top {C.TOP_QUANTILE:.0%} of the eligible pool by formation return")
    out()
    out("  STOP. The split is now frozen. Step 9 tests robustness to alternative")
    out("  boundaries; it does NOT re-choose them based on which alpha looks better.")

    rp = C.REPORTS / "step2_diagnostic.txt"
    rp.write_text("\n".join(lines))
    print(f"\nSaved -> {rp}")


if __name__ == "__main__":
    main()
