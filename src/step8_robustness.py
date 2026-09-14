"""
Step 9 -- Robustness.

EVERY variant below is reported beside the baseline. None is presented alone, and
none is used to replace the baseline. The reason is the multiple-comparisons
problem: run enough specifications and one will cross t = 2 by chance. The defence
is not to run fewer specifications -- it is to show all of them, so a reader can
see whether a "significant" cell is the exception among many nulls or a consistent
pattern.

Variants:
  1. Subperiod stability   -- 2020-21 vs 2022-24, SELECTION UNCHANGED.
  2. Rolling 24-month alpha -- every window shown, so no window can be cherry-picked.
  3. Era-dummy interaction  -- one model formally testing whether alpha differs by era.
  4. Weighting              -- equal-weight vs midpoint-size.
  5. Entry lag              -- 1 month (baseline) vs 0 months.
  6. Leg de-duplication     -- one leg per ticker vs conviction-weighted legs.
  7. Split boundaries       -- two alternative formation/evaluation splits.
  8. Committee map variants -- Track B under debatable sector assignments.

ANTI-BIAS NOTE on the subperiod and era analyses: the SPLIT IS NEVER MOVED to
answer a regime question. Selection stays exactly as fixed in Step 5; only the
evaluation window is sliced afterwards. Re-selecting members inside a chosen era
would be selecting on the outcome, which is the precise error this design exists
to avoid.

FRAMING CAVEAT for era comparisons: political regime is confounded with market
conditions (the COVID crash and recovery, the 2022 bear market). Most cross-era
difference in these portfolios is BETA, not politics. This is kept as a
market-regime-stability question and must not be reported as a partisan claim.

Run:  python src/step8_robustness.py
Out:  data/reports/step9_robustness.txt, data/clean/rolling_alpha.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from step3_holding import (build_panel, attach_returns, member_monthly_returns,
                           group_portfolio, invested_months)
from step6_portfolio import build_all, series_for, trackB_panel
from step7_regression import run_regression, load_factors
from step5_committee import load_sector_map, committee_sectors

REPORT = C.REPORTS / "step9_robustness.txt"
HEAD = C.HEADLINE_CONSTRUCTION


def line(r, label):
    if r is None:
        return f"  {label:<44} insufficient data"
    sig = "***" if r["alpha_p"] < .01 else "**" if r["alpha_p"] < .05 \
        else "*" if r["alpha_p"] < .10 else ""
    return (f"  {label:<44} {r['alpha_ann']*100:>+7.2f}%  t={r['alpha_t']:>5.2f}{sig:<3} "
            f"CI[{r['alpha_ci_lo_ann']*100:>+6.1f}%,{r['alpha_ci_hi_ann']*100:>+6.1f}%] "
            f"b={r['beta']:>5.2f} N={r['N']:>3}")


# ------------------------------------------------------------------ 1. subperiods
def subperiods(px, factors, out):
    out("-" * 100)
    out("1. SUBPERIOD STABILITY  (selection UNCHANGED; only the evaluation window is sliced)")
    out("-" * 100)
    eras = [("full 2020-2024", C.EVAL_START, C.EVAL_END),
            ("2020-2021 (COVID crash + recovery)", "2020-01-01", "2021-12-31"),
            ("2022-2024 (bear then recovery)", "2022-01-01", "2024-12-31")]
    for g in ("trackA_selected", "trackB_combined"):
        col = f"{g}|{HEAD}"
        if col not in px.columns:
            continue
        out(f"  {g} ({HEAD}, CAPM):")
        for label, s, e in eras:
            ser = px[col].dropna()
            ser = ser[(ser.index >= s) & (ser.index <= e)]
            out(line(run_regression(ser, factors), "    " + label))
        out("")


# ------------------------------------------------------------------ 2. rolling
def rolling_alpha(px, factors, out, window=24):
    out("-" * 100)
    out(f"2. ROLLING {window}-MONTH CAPM ALPHA  (every window shown -- no cherry-picking)")
    out("-" * 100)
    recs = []
    for g in ("trackA_selected", "trackB_combined"):
        col = f"{g}|{HEAD}"
        if col not in px.columns:
            continue
        s = px[col].dropna()
        out(f"  {g} ({HEAD}):")
        for i in range(len(s) - window + 1):
            w = s.iloc[i:i + window]
            r = run_regression(w, factors, lags=3)
            if r is None:
                continue
            recs.append({"group": g, "end": w.index[-1],
                         "alpha_ann": r["alpha_ann"], "t": r["alpha_t"],
                         "beta": r["beta"]})
            if i % 6 == 0 or i == len(s) - window:
                bar = "+" * int(max(0, min(28, r["alpha_ann"] * 100 / 2))) or \
                      "-" * int(max(0, min(28, -r["alpha_ann"] * 100 / 2)))
                out(f"    {w.index[0]:%Y-%m}..{w.index[-1]:%Y-%m}  "
                    f"a={r['alpha_ann']*100:>+7.2f}%  t={r['alpha_t']:>5.2f}  "
                    f"b={r['beta']:>5.2f}  {bar}")
        out("")
    df = pd.DataFrame(recs)
    if len(df):
        df.to_csv(C.CLEAN / "rolling_alpha.csv", index=False)
        for g, sub in df.groupby("group"):
            n_sig = (sub["t"].abs() > 2).sum()
            out(f"  {g}: {len(sub)} windows, alpha range "
                f"[{sub['alpha_ann'].min()*100:+.1f}%, {sub['alpha_ann'].max()*100:+.1f}%], "
                f"{n_sig} with |t|>2")
        out("  Windows overlap heavily, so these t-stats are NOT independent tests --")
        out("  the chart is for shape and stability, not for counting significances.")
    out("")


# ------------------------------------------------------------------ 3. era dummy
def era_dummy(px, factors, out, cut="2022-01-01"):
    out("-" * 100)
    out(f"3. ERA-DUMMY INTERACTION  (one model: does alpha differ before/after {cut[:7]}?)")
    out("-" * 100)
    out("   R_p - R_f = a0 + a1*D + b0*MKT + b1*(D x MKT) + e,  D = 1 after the cut")
    out("   a1 tests the ALPHA difference; b1 tests the BETA difference.")
    for g in ("trackA_selected", "trackB_combined"):
        col = f"{g}|{HEAD}"
        if col not in px.columns:
            continue
        s = px[col].dropna()
        d = pd.concat([s.rename("rp"), factors[["Mkt_RF", "RF"]]], axis=1).dropna()
        if len(d) < 24:
            continue
        d["D"] = (d.index >= pd.Timestamp(cut)).astype(float)
        d["DxM"] = d["D"] * d["Mkt_RF"]
        y = d["rp"] - d["RF"]
        X = sm.add_constant(d[["D", "Mkt_RF", "DxM"]])
        f = sm.OLS(y, X).fit(cov_type="HAC",
                             cov_kwds={"maxlags": C.NEWEY_WEST_LAGS})
        out(f"  {g} ({HEAD}):  N={int(f.nobs)}")
        out(f"    a0 (pre-cut alpha)   {f.params['const']*12*100:>+7.2f}%/yr  "
            f"t={f.tvalues['const']:>5.2f}")
        out(f"    a1 (alpha SHIFT)     {f.params['D']*12*100:>+7.2f}%/yr  "
            f"t={f.tvalues['D']:>5.2f}   p={f.pvalues['D']:.3f}")
        out(f"    b0 (pre-cut beta)    {f.params['Mkt_RF']:>7.2f}     "
            f"t={f.tvalues['Mkt_RF']:>5.2f}")
        out(f"    b1 (beta SHIFT)      {f.params['DxM']:>7.2f}     "
            f"t={f.tvalues['DxM']:>5.2f}   p={f.pvalues['DxM']:.3f}")
        verdict = ("alpha DIFFERS across eras" if f.pvalues["D"] < .05
                   else "no detectable alpha difference across eras")
        out(f"    -> {verdict}")
    out("  CAVEAT: regime is confounded with market conditions. Read this as market-")
    out("  regime stability, NOT as a claim about politics.")
    out("")


# ------------------------------------------------------------------ 4-6. variants
def construction_variants(factors, out):
    out("-" * 100)
    out("4-6. WEIGHTING / ENTRY-LAG / LEG-DEDUPE VARIANTS")
    out("-" * 100)
    specs = [
        ("BASELINE equal-weight, lag 1, legs kept", dict()),
        ("midpoint-SIZE weighted", dict(weight="amount")),
        ("entry lag 0 (earns filing month itself)", dict(entry_lag=0)),
        ("de-duplicated legs (1 per ticker)", dict(dedupe=True)),
    ]
    for g in ("trackA_selected", "trackB_combined"):
        out(f"  {g} ({HEAD}, CAPM):")
        for label, kw in specs:
            try:
                ser = build_all(**kw).get((g, HEAD))
            except Exception as e:
                out(f"    {label:<42} FAILED ({type(e).__name__}: {e})")
                continue
            if ser is None or not len(ser):
                out(f"    {label:<42} no series")
                continue
            out(line(run_regression(ser, factors), "  " + label))
        out("")
    out("  Entry lag 0 is shown because it is the LESS conservative choice: it lets a")
    out("  position earn the part of the filing month that preceded disclosure. If")
    out("  alpha appears only at lag 0, that alpha is a look-ahead artefact.")
    out("")


# ------------------------------------------------------------------ 7. splits
def split_variants(factors, out):
    out("-" * 100)
    out("7. SPLIT-BOUNDARY VARIANTS  (does the conclusion hinge on the exact date?)")
    out("-" * 100)
    out("   Selection is RE-RUN INSIDE each alternative split -- formation ranking on")
    out("   that split's formation window only. The baseline split is NOT replaced.")
    trades = pd.read_parquet(C.CLEAN / "congress_trades.parquet")
    rets = pd.read_parquet(C.CLEAN / "returns_monthly.parquet")
    panel_all = pd.read_parquet(C.CLEAN / "held_positions.parquet")

    alts = [("BASELINE  form 2014-2019 / eval 2020-2024",
             C.FORMATION_START, C.FORMATION_END, C.EVAL_START, C.EVAL_END),
            ("ALT A     form 2014-2018 / eval 2019-2024",
             "2014-01-01", "2018-12-31", "2019-01-01", "2024-12-31"),
            ("ALT B     form 2014-2020 / eval 2021-2024",
             "2014-01-01", "2020-12-31", "2021-01-01", "2024-12-31")]

    for label, fs, fe, es, ee in alts:
        nf = trades[(trades["filing_date"] >= fs) &
                    (trades["filing_date"] <= fe)].groupby("filer_id").size()
        inv = invested_months(panel_all, fs, fe, long_only=True)
        pool = sorted(set(nf[nf >= C.MIN_FORMATION_TRADES].index) &
                      set(inv[inv >= C.MIN_INVESTED_MONTHS].index))
        fp = panel_all[(panel_all["month"] >= fs) & (panel_all["month"] <= fe)]
        mr = member_monthly_returns(fp, construction=HEAD)
        mr = mr[[c for c in pool if c in mr.columns]]
        rank = mr.mean().sort_values(ascending=False)
        n_sel = max(1, int(round(len(rank) * C.TOP_QUANTILE)))
        sel = rank.index[:n_sel].tolist()

        ep = panel_all[(panel_all["month"] >= es) & (panel_all["month"] <= ee)]
        emr = member_monthly_returns(ep, construction=HEAD)
        ser = group_portfolio(emr, sel, es, ee)
        r = run_regression(ser, factors)
        out(line(r, label))
        out(f"      pool {len(pool)} -> selected {n_sel}: {sel}")
    out("")


# ------------------------------------------------------------------ 8. map variants
def map_variants(factors, out):
    out("-" * 100)
    out("8. COMMITTEE->SECTOR MAP ROBUSTNESS  (Track B)")
    out("-" * 100)
    path = C.CLEAN / "trackB_oncommittee_all.parquet"
    if not path.exists():
        out("  Track B tables absent -- run step5_committee.py first.")
        return
    oncom = pd.read_parquet(path)
    rets = pd.read_parquet(C.CLEAN / "returns_monthly.parquet")
    smap = load_sector_map()

    variants = [("BASELINE map", None)] + [
        (f"{k}: {v['description']}", k) for k, v in smap["variants"].items()]

    for label, vk in variants:
        csec = committee_sectors(smap, variant=vk)
        keep = oncom[oncom.apply(
            lambda r: r["sector"] in csec.get(r["committee_key"], []), axis=1)]
        if not len(keep):
            out(f"  {label:<44} no trades survive")
            continue
        # Dedupe on trade_id first: a trade overseen by two of the member's
        # committees appears twice in `keep`, which would double-count it.
        bp = trackB_panel(keep.drop_duplicates("trade_id"), rets)
        if not len(bp):
            out(f"  {label:<44} no priced legs")
            continue
        mem = sorted(bp["filer_id"].unique())
        ser = series_for(bp, mem, HEAD)
        r = run_regression(ser, factors)
        out(line(r, label))
        out(f"      {len(keep):,} trades, {keep['filer_id'].nunique()} members")
    out("")
    out("  If the baseline and the variants agree, the objection 'you chose the map")
    out("  to get the answer' is answered. If they disagree, that is reported here")
    out("  as a genuine limitation rather than resolved by picking a favourite.")
    out("")


def main():
    px = pd.read_parquet(C.CLEAN / "portfolio_returns.parquet")
    factors = load_factors()

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out("STEP 9 -- ROBUSTNESS.  ALL VARIANTS REPORTED BESIDE THE BASELINE.")
    out("=" * 100)
    out(f"Headline construction: {HEAD}. Significance: * .10  ** .05  *** .01")
    out("")

    subperiods(px, factors, out)
    rolling_alpha(px, factors, out)
    era_dummy(px, factors, out)
    construction_variants(factors, out)
    split_variants(factors, out)
    map_variants(factors, out)

    out("=" * 100)
    out("MULTIPLE-COMPARISONS WARNING")
    out("=" * 100)
    out("  This report contains many specifications. Under the null of zero alpha,")
    out("  roughly 1 in 20 would cross the 5% threshold by chance alone. A single")
    out("  starred cell among many nulls is therefore NOT evidence of skill. Read")
    out("  the table for CONSISTENCY of sign and magnitude across variants, not for")
    out("  the presence of a star somewhere in it.")

    REPORT.write_text("\n".join(lines))
    print(f"\nReport -> {REPORT}")


if __name__ == "__main__":
    main()
