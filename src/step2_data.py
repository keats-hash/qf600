"""
Step 3 -- Prices and factors.

Fetches monthly adjusted-close prices (yfinance, auto_adjust=True) for every ticker
the eligible universe traded, plus the Ken French factors, and builds the monthly
simple-return panel that every downstream step consumes.

ANTI-BIAS NOTES:
  * auto_adjust=True gives split- and dividend-adjusted closes, so a return is a
    real total return and not a corporate-action artefact.
  * Prices are fetched for the FULL span once. No step is allowed to re-fetch a
    narrower span, because a narrower span could be chosen to flatter a result.
  * Tickers with no price coverage are DROPPED AND LOGGED per member. Silent drops
    would be survivorship bias: a delisted name that went to zero would simply
    vanish from the portfolio instead of registering its loss.

Caching: prices are cached to data/clean/prices_monthly.parquet and the fetch is
resumable -- re-running only pulls tickers not already cached.

Run:  python src/step2_data.py [--refresh]
Out:  data/clean/prices_monthly.parquet, data/clean/returns_monthly.parquet,
      data/clean/factors_monthly.parquet
"""
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

import warnings
warnings.filterwarnings("ignore")

PRICES = C.CLEAN / "prices_monthly.parquet"
RETURNS = C.CLEAN / "returns_monthly.parquet"
FACTORS = C.CLEAN / "factors_monthly.parquet"
COVERAGE = C.REPORTS / "step3_price_coverage.txt"

BATCH = 80
MAX_RETRIES = 3


def universe_tickers():
    """Union of tickers traded by congress members over the full study span.

    Deliberately wide: it covers both tracks' candidate universes, so no later
    step can shrink the price panel in a way that changes which trades survive.
    """
    df = pd.read_parquet(C.CLEAN / "congress_trades.parquet")
    m = (df["filing_date"] >= C.FORMATION_START) & (df["filing_date"] <= C.EVAL_END)
    tk = sorted(df.loc[m, "ticker"].dropna().unique())
    print(f"Universe: {len(tk):,} distinct tickers from {m.sum():,} trades "
          f"({df.loc[m, 'filer_id'].nunique()} members)")
    return tk, df.loc[m]


def fetch_prices(tickers, refresh=False):
    """Batched monthly adjusted-close download, resumable via the parquet cache."""
    import yfinance as yf

    cached = pd.DataFrame()
    if PRICES.exists() and not refresh:
        cached = pd.read_parquet(PRICES)
        have = set(cached.columns)
        todo = [t for t in tickers if t not in have]
        print(f"Cache: {len(have):,} tickers present, {len(todo):,} to fetch")
    else:
        todo = list(tickers)
        print(f"No cache -- fetching all {len(todo):,} tickers")

    if not todo:
        return cached

    frames = []
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        for attempt in range(MAX_RETRIES):
            try:
                d = yf.download(
                    chunk, start=C.PRICE_START, end=C.PRICE_END, interval="1mo",
                    auto_adjust=True, progress=False, threads=True,
                )
                break
            except Exception as e:                      # transient network/rate limit
                if attempt == MAX_RETRIES - 1:
                    print(f"  batch {i//BATCH}: FAILED after {MAX_RETRIES} tries ({e})")
                    d = pd.DataFrame()
                else:
                    time.sleep(3 * (attempt + 1))
        if len(d):
            close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]]
            if not isinstance(d.columns, pd.MultiIndex):
                close.columns = chunk[:1]
            frames.append(close)
        got = len(frames[-1].columns) if frames else 0
        print(f"  batch {i//BATCH + 1}/{(len(todo)-1)//BATCH + 1}: "
              f"requested {len(chunk)}, got {got} cols")
        time.sleep(1)

    new = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    px = pd.concat([cached, new], axis=1) if len(cached) else new
    px = px.loc[:, ~px.columns.duplicated()]

    # yfinance labels a monthly bar at the START of the month; the close belongs to
    # the END of that month. Normalise so every series aligns on month-end.
    px.index = pd.to_datetime(px.index).to_period("M").to_timestamp("M")
    px = px.sort_index()
    px = px.dropna(axis=1, how="all")

    px.to_parquet(PRICES)
    print(f"Prices: {px.shape[0]} months x {px.shape[1]:,} tickers -> {PRICES}")
    return px


def fill_gaps(tickers, rounds=6):
    """Retry tickers still missing from the cache, with shrinking batches and backoff.

    Yahoo rate-limits aggressively on a universe this size, and a rate-limited
    batch looks identical to a delisted ticker in the log. Treating a rate-limit
    as "no price" would silently delete large, very-much-alive names (PEP, PG,
    PFE were all lost to rate limits on the first pass) -- which biases the
    portfolio toward whatever happened to download. So we retry until the missing
    set stops shrinking; only then is a ticker genuinely unavailable.
    """
    import yfinance as yf

    px = pd.read_parquet(PRICES)
    for rnd in range(rounds):
        missing = [t for t in tickers if t not in px.columns]
        if not missing:
            print("  No gaps remain.")
            break
        size = max(5, 40 // (rnd + 1))
        wait = 5 * (rnd + 1)
        print(f"  Round {rnd+1}: {len(missing):,} missing, batch={size}, sleep={wait}s")
        got_any = 0
        frames = []
        for i in range(0, len(missing), size):
            chunk = missing[i:i + size]
            try:
                d = yf.download(chunk, start=C.PRICE_START, end=C.PRICE_END,
                                interval="1mo", auto_adjust=True, progress=False,
                                threads=False)
            except Exception:
                d = pd.DataFrame()
            if len(d):
                close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]]
                if not isinstance(d.columns, pd.MultiIndex):
                    close.columns = chunk[:1]
                close = close.dropna(axis=1, how="all")
                if close.shape[1]:
                    frames.append(close)
                    got_any += close.shape[1]
            time.sleep(wait)
        if frames:
            new = pd.concat(frames, axis=1)
            new.index = pd.to_datetime(new.index).to_period("M").to_timestamp("M")
            px = pd.concat([px, new], axis=1)
            px = px.loc[:, ~px.columns.duplicated()].sort_index()
            px.to_parquet(PRICES)
        print(f"    recovered {got_any} tickers; cache now {px.shape[1]:,}")
        if got_any == 0:
            print("    no progress -- remaining tickers are genuinely unavailable")
            break
    return px


# Explicit, AUTHOR-DEFINED continuation map for companies that merely changed
# ticker while remaining the same listed entity. Using the successor symbol
# recovers the SAME company's continuous adjusted price history, so it is not
# look-ahead -- it is the same series under a different name. Companies that were
# genuinely ACQUIRED or taken private are deliberately NOT listed here: for those
# there is no continuing series, and pretending otherwise would fabricate returns.
TICKER_RENAMES = {
    "SQ": "XYZ",        # Block renamed its ticker 2025
    "FB": "META",       # Meta 2022
    "UTX": "RTX",       # United Technologies -> Raytheon Technologies 2020
    "ANTM": "ELV",      # Anthem -> Elevance 2022
    "BLL": "BALL",      # Ball Corp ticker change
    "WLTW": "WTW",      # Willis Towers Watson 2022
    "FBHS": "FBIN",     # Fortune Brands 2022
    "RDS.A": "SHEL", "RDS.B": "SHEL", "RDSA": "SHEL", "RDSB": "SHEL",
    "BRK.A": "BRK-A",
}


def alias_candidates(t):
    """Yahoo-compatible spellings to try for a ticker we could not fetch."""
    out = []
    if t in TICKER_RENAMES:
        out.append(TICKER_RENAMES[t])
    if "." in t:
        out.append(t.replace(".", "-"))   # Yahoo writes share classes BRK-B, not BRK.B
        out.append(t.replace(".", ""))
    return [a for a in dict.fromkeys(out) if a and a != t]


def recover_missing(tickers, sleep=1.5):
    """Last pass: try ticker aliases, then slow serial retries.

    Splits the missing set into (a) recoverable spelling/rename cases and (b)
    genuine delistings. Only (b) should end up dropped, and the write-up reports
    it as delisting bias rather than hiding it.
    """
    import yfinance as yf

    px = pd.read_parquet(PRICES)
    missing = [t for t in tickers if t not in px.columns]
    print(f"Recovery pass over {len(missing):,} missing tickers")

    recovered, still = {}, []
    for i, t in enumerate(missing, 1):
        series = None
        for cand in alias_candidates(t) + [t]:
            try:
                d = yf.download(cand, start=C.PRICE_START, end=C.PRICE_END,
                                interval="1mo", auto_adjust=True, progress=False,
                                threads=False)
            except Exception:
                d = pd.DataFrame()
            if len(d):
                s = d["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s = s.dropna()
                if len(s) > 6:
                    series = s.rename(t)     # store under the ORIGINAL symbol
                    if cand != t:
                        print(f"    {t} recovered via alias {cand} ({len(s)} months)")
                    break
            time.sleep(sleep)
        if series is not None:
            recovered[t] = series
        else:
            still.append(t)
        if i % 50 == 0:
            print(f"  {i:,}/{len(missing):,}: recovered {len(recovered)}, "
                  f"unavailable {len(still)}")
            if recovered:
                _merge_save(px, recovered)
        time.sleep(sleep)

    px = _merge_save(px, recovered)
    print(f"Recovery complete: +{len(recovered)} tickers, "
          f"{len(still)} genuinely unavailable")
    (C.REPORTS / "step3_unavailable_tickers.txt").write_text(
        "Tickers with no price series after alias + retry recovery.\n"
        "These are overwhelmingly companies ACQUIRED or taken private during the\n"
        "study window, for which no continuing series exists. Trades in them are\n"
        "dropped; this is DELISTING BIAS and is reported as a limitation.\n\n"
        + "\n".join(still))
    return px


def _merge_save(px, recovered):
    if not recovered:
        return px
    new = pd.DataFrame(recovered)
    new.index = pd.to_datetime(new.index).to_period("M").to_timestamp("M")
    px = pd.concat([px, new], axis=1)
    px = px.loc[:, ~px.columns.duplicated()].sort_index()
    px.to_parquet(PRICES)
    return px


def build_returns(px):
    """Monthly simple return per ticker. Month t return uses close(t)/close(t-1)."""
    ret = px.sort_index().pct_change()
    # A single-month gap must not silently become a two-month compounded return,
    # so a return is only valid where BOTH endpoints are observed.
    valid = px.notna() & px.shift(1).notna()
    ret = ret.where(valid)
    # Guard against yfinance artefacts (bad splits) producing absurd monthly moves.
    extreme = (ret.abs() > 5).sum().sum()
    if extreme:
        print(f"  WARNING: {extreme} monthly returns with |r| > 500% -- masked as NaN "
              f"(likely unadjusted split artefacts)")
        ret = ret.where(ret.abs() <= 5)
    ret.to_parquet(RETURNS)
    print(f"Returns: {ret.shape[0]} months x {ret.shape[1]:,} tickers -> {RETURNS}")
    return ret


def log_coverage(trades, px):
    """Report which tickers have no price coverage, and which members that hits."""
    have = set(px.columns)
    traded = set(trades["ticker"].unique())
    missing = sorted(traded - have)

    lines = ["STEP 3 -- PRICE COVERAGE", "=" * 60,
             f"Tickers traded : {len(traded):,}",
             f"Tickers priced : {len(traded & have):,} ({len(traded & have)/len(traded):.1%})",
             f"Tickers missing: {len(missing):,}", ""]

    bad = trades[trades["ticker"].isin(missing)]
    lines.append(f"Trades dropped for want of a price: {len(bad):,} "
                 f"({len(bad)/len(trades):.1%} of in-span trades)")
    lines.append("")
    lines.append("Unpriced tickers (delisted, acquired, or bad symbols):")
    lines.append("  " + ", ".join(missing[:120]) + (" ..." if len(missing) > 120 else ""))
    lines.append("")
    lines.append("Trades dropped per member (top 25) -- survivorship exposure:")
    per = bad.groupby("filer_id").size().sort_values(ascending=False)
    tot = trades.groupby("filer_id").size()
    for fid, n in per.head(25).items():
        lines.append(f"  {fid:<40} {n:>4} dropped / {tot[fid]:>5} traded "
                     f"({n/tot[fid]:5.1%})")

    COVERAGE.write_text("\n".join(lines))
    print("\n".join(lines[:8]))
    print(f"... full report -> {COVERAGE}")
    return missing


def _parse_ff(url, value_names):
    """Parse a Ken French monthly CSV: skip the preamble, stop at the annual block."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=90).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    txt = z.read(z.namelist()[0]).decode("latin-1")

    rows = []
    for line in txt.splitlines():
        parts = [p.strip() for p in line.split(",")]
        # Monthly rows are keyed YYYYMM (6 digits). Annual rows are YYYY (4) --
        # hitting those means the monthly block has ended.
        if len(parts) >= 2 and parts[0].isdigit():
            if len(parts[0]) == 6:
                rows.append(parts)
            elif len(parts[0]) == 4 and rows:
                break
    df = pd.DataFrame(rows).set_index(0)
    df = df.iloc[:, :len(value_names)]
    df.columns = value_names
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index, format="%Y%m").to_period("M").to_timestamp("M")
    return df / 100.0          # Ken French ships percent; we work in decimals


def fetch_factors():
    print("\nFetching Ken French factors ...")
    ff = _parse_ff(C.FF_FACTORS_URL, ["Mkt_RF", "SMB", "HML", "RF"])
    mom = _parse_ff(C.FF_MOM_URL, ["MOM"])
    f = ff.join(mom, how="left")
    f = f.loc[(f.index >= C.PRICE_START) & (f.index <= C.PRICE_END)]
    f.to_parquet(FACTORS)
    print(f"Factors: {f.index.min():%Y-%m} .. {f.index.max():%Y-%m}, "
          f"{len(f)} months, cols {list(f.columns)} -> {FACTORS}")
    ev = f.loc[C.EVAL_START:C.EVAL_END]
    print(f"  Evaluation window has {len(ev)} factor months "
          f"(need 60); Mkt-RF mean {ev['Mkt_RF'].mean()*12:.2%}/yr")
    assert f["MOM"].notna().sum() > 0, "Momentum factor failed to join"
    return f


def main():
    refresh = "--refresh" in sys.argv
    tickers, trades = universe_tickers()
    if "--recover" in sys.argv:
        px = recover_missing(tickers)
    elif "--gapfill-only" in sys.argv:
        px = fill_gaps(tickers)
    else:
        px = fetch_prices(tickers, refresh=refresh)
        print("\nGap-filling rate-limited tickers ...")
        px = fill_gaps(tickers)
    build_returns(px)
    log_coverage(trades, px)
    fetch_factors()
    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
