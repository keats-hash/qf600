"""
Step 8 -- CAPM and 4-factor regressions.

For each group x construction:
    CAPM     :  R_p - R_f = a + b*(R_M - R_f) + e
    4-factor :  R_p - R_f = a + b*MKT + s*SMB + h*HML + m*MOM + e

Estimated on monthly data with NEWEY-WEST (HAC) standard errors. HAC matters here
because overlapping 12-month holding windows induce serial correlation in the
portfolio return: consecutive months share most of their open legs. Ordinary OLS
standard errors would be too small and would manufacture significance.

REPORTING RULE: alpha is reported with its t-statistic AND its 95% confidence
interval, always. With samples this small the CI is the honest statistic -- a wide
interval straddling zero means UNDERPOWERED, which is a different claim from
"no alpha", and the two must not be conflated.

If alpha survives the market factor but dies against the 4-factor model, the
correct reading is "alpha decays into beta" -- the return was compensation for
known risk premia, not skill. Both models are always reported together so that
reading is available.

Run:  python src/step7_regression.py
Out:  data/clean/regression_results.csv, data/reports/step8_regressions.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

REPORT = C.REPORTS / "step8_regressions.txt"


def load_factors():
    return pd.read_parquet(C.CLEAN / "factors_monthly.parquet")


def run_regression(ret, factors, model="CAPM", lags=None):
    """One HAC regression. Returns a dict of the reported statistics."""
    lags = C.NEWEY_WEST_LAGS if lags is None else lags
    cols = ["Mkt_RF"] if model == "CAPM" else ["Mkt_RF", "SMB", "HML", "MOM"]

    d = pd.concat([ret.rename("rp"), factors[cols + ["RF"]]], axis=1).dropna()
    if len(d) < 12:
        return None

    y = d["rp"] - d["RF"]          # excess return of the portfolio
    X = sm.add_constant(d[cols])
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    ci = fit.conf_int(alpha=0.05).loc["const"]
    res = {
        "model": model,
        "N": int(len(d)),
        "alpha_m": fit.params["const"],
        "alpha_ann": fit.params["const"] * C.ANNUALISE,
        "alpha_t": fit.tvalues["const"],
        "alpha_p": fit.pvalues["const"],
        "alpha_ci_lo_ann": ci[0] * C.ANNUALISE,
        "alpha_ci_hi_ann": ci[1] * C.ANNUALISE,
        "beta": fit.params["Mkt_RF"],
        "beta_t": fit.tvalues["Mkt_RF"],
        "r2": fit.rsquared,
        "r2_adj": fit.rsquared_adj,
    }
    for f in ("SMB", "HML", "MOM"):
        res[f.lower()] = fit.params.get(f, np.nan)
    return res


def fmt_row(g, c, r):
    sig = "***" if r["alpha_p"] < .01 else "**" if r["alpha_p"] < .05 \
        else "*" if r["alpha_p"] < .10 else ""
    return (f"{g:<26} {c:<11} {r['model']:<9} "
            f"{r['alpha_ann']*100:>7.2f}% {r['alpha_t']:>6.2f}{sig:<3} "
            f"[{r['alpha_ci_lo_ann']*100:>6.1f}%,{r['alpha_ci_hi_ann']*100:>6.1f}%] "
            f"{r['beta']:>6.2f} {r['r2']:>5.2f} {r['N']:>4}")


def main():
    px = pd.read_parquet(C.CLEAN / "portfolio_returns.parquet")
    factors = load_factors()

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 110)
    out("STEP 8 -- CAPM AND 4-FACTOR REGRESSIONS  (Newey-West HAC, "
        f"{C.NEWEY_WEST_LAGS} lags)")
    out("=" * 110)
    out(f"Evaluation window {C.EVAL_START[:7]} .. {C.EVAL_END[:7]}. "
        f"alpha annualised (x12). Significance: * .10  ** .05  *** .01")
    out("HAC lags chosen for the 12-month overlapping holding windows, which make")
    out("consecutive monthly returns mechanically autocorrelated.")
    out()
    out(f"{'group':<26} {'constr':<11} {'model':<9} "
        f"{'alpha/yr':>8} {'t':>6}    {'95% CI (ann)':<17} "
        f"{'beta':>6} {'R2':>5} {'N':>4}")
    out("-" * 110)

    rows = []
    for col in sorted(px.columns):
        g, c = col.split("|")
        s = px[col].dropna()
        for model in ("CAPM", "FF4"):
            r = run_regression(s, factors, model=model)
            if r is None:
                out(f"{g:<26} {c:<11} {model:<9} insufficient data (N<12)")
                continue
            r["group"], r["construction"] = g, c
            rows.append(r)
            out(fmt_row(g, c, r))
        out("")

    res = pd.DataFrame(rows)
    res.to_csv(C.CLEAN / "regression_results.csv", index=False)

    # ---------------------------------------------------------------- reading
    out("=" * 110)
    out("READING THE TABLE")
    out("=" * 110)
    head = res[(res["group"] == "trackA_selected") &
               (res["construction"] == C.HEADLINE_CONSTRUCTION) &
               (res["model"] == "CAPM")]
    if len(head):
        r = head.iloc[0]
        out(f"HEADLINE (Track A, {C.HEADLINE_CONSTRUCTION}, CAPM):")
        out(f"  alpha = {r['alpha_ann']*100:+.2f}%/yr, t = {r['alpha_t']:.2f}, "
            f"95% CI [{r['alpha_ci_lo_ann']*100:+.1f}%, {r['alpha_ci_hi_ann']*100:+.1f}%], "
            f"beta = {r['beta']:.2f}, R2 = {r['r2']:.2f}, N = {int(r['N'])}")
        verdict = ("DETECTABLE alpha" if r["alpha_p"] < 0.05
                   else "NO detectable risk-adjusted alpha")
        out(f"  -> {verdict} at the 5% level.")
        width = (r["alpha_ci_hi_ann"] - r["alpha_ci_lo_ann"]) * 100
        out(f"  -> CI width {width:.1f} percentage points: with N={int(r['N'])} months this")
        out( "     test could only detect a very large effect, so a null here is better")
        out( "     described as UNDERPOWERED than as proof of no skill.")
    out()

    # Beta comparison across constructions -- the directional-timing test.
    out("BETA: long-only vs long-short (the directional-timing check)")
    for g in sorted(res["group"].unique()):
        lo = res[(res["group"] == g) & (res["construction"] == "long_only") &
                 (res["model"] == "CAPM")]
        ls = res[(res["group"] == g) & (res["construction"] == "long_short") &
                 (res["model"] == "CAPM")]
        if len(lo) and len(ls):
            out(f"  {g:<26} long-only beta {lo.iloc[0]['beta']:>5.2f}  ->  "
                f"long-short beta {ls.iloc[0]['beta']:>5.2f}")
    out("  A long-short beta near zero means the sale disclosures genuinely offset")
    out("  market exposure. A long-short beta that stays high means the 'short' legs")
    out("  are not hedging anything and the portfolio is still just long the market.")
    out()

    out("ALPHA DECAY: CAPM vs 4-factor")
    for g in sorted(res["group"].unique()):
        for c in ("long_only", "long_short"):
            a = res[(res["group"] == g) & (res["construction"] == c) &
                    (res["model"] == "CAPM")]
            b = res[(res["group"] == g) & (res["construction"] == c) &
                    (res["model"] == "FF4")]
            if len(a) and len(b):
                out(f"  {g:<26} {c:<11} CAPM {a.iloc[0]['alpha_ann']*100:>+7.2f}%  "
                    f"-> FF4 {b.iloc[0]['alpha_ann']*100:>+7.2f}%  "
                    f"(t {a.iloc[0]['alpha_t']:>5.2f} -> {b.iloc[0]['alpha_t']:>5.2f})")

    out()
    out(f"Saved -> {C.CLEAN / 'regression_results.csv'}")
    REPORT.write_text("\n".join(lines))
    print(f"Report -> {REPORT}")


if __name__ == "__main__":
    main()
