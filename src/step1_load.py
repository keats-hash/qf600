"""
Step 1 -- Load, filter, clean congressional disclosures.

Reads the per-filer JSON archive from the kadoa tarball and produces one tidy row
per trade. Every filter prints the row count it dropped so the funnel is auditable.

ANTI-BIAS NOTE: the shipped return fields (ret_since, excess_since, ret_30d,
ret_1y) are deliberately IGNORED. They are computed from the trade date and with
an unspecified methodology; we compute every return ourselves from disclosure-date
entry in later steps. Reading them here would import look-ahead bias.

Run:  python src/step1_load.py
Out:  data/clean/congress_trades.parquet
"""
import json
import sys
import tarfile
import urllib.request

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C


def ensure_raw_data():
    """Download + extract the kadoa tarball once. No-op if already present."""
    if C.KADOA_DIR.exists() and len(list(C.KADOA_DIR.glob("*.json"))) > 100:
        print(f"Raw data present: {len(list(C.KADOA_DIR.glob('*.json')))} filer files")
        return
    tarball = C.RAW / "kadoa.tar.gz"
    if not tarball.exists():
        print(f"Downloading {C.KADOA_TARBALL_URL} ...")
        urllib.request.urlretrieve(C.KADOA_TARBALL_URL, tarball)
    print("Extracting ...")
    with tarfile.open(tarball) as tf:
        tf.extractall(C.RAW)
    print(f"Extracted: {len(list(C.KADOA_DIR.glob('*.json')))} filer files")


def flatten():
    """One row per trade, with filer-level attributes attached."""
    rows = []
    files = sorted(C.KADOA_DIR.glob("*.json"))
    for fp in files:
        with open(fp) as fh:
            doc = json.load(fh)
        filer = doc.get("filer", {})
        for t in doc.get("trades", []):
            rows.append(
                {
                    "trade_id": t.get("id"),
                    "filer_id": filer.get("id") or t.get("filer_id"),
                    "full_name": filer.get("full_name"),
                    "branch": filer.get("branch"),
                    "chamber": filer.get("chamber"),
                    "party": filer.get("party"),
                    "state": filer.get("state"),
                    "transaction_date": t.get("transaction_date"),
                    "filing_date": t.get("filing_date"),
                    "ticker": t.get("ticker"),
                    "asset_name": t.get("asset_name"),
                    "asset_type": t.get("asset_type"),
                    "transaction_type": t.get("transaction_type"),
                    "amount_range_low": t.get("amount_range_low"),
                    "amount_range_high": t.get("amount_range_high"),
                    "owner": t.get("owner"),
                }
            )
    df = pd.DataFrame(rows)
    print(f"Flattened {len(df):,} trades from {len(files)} filer files "
          f"({df['filer_id'].nunique():,} distinct filers)")
    return df


def filter_trades(df):
    """Apply the six committed filters, logging the funnel at every stage."""
    print("\n=== FILTER FUNNEL (auditable) ===")
    n0 = len(df)

    # (1) Congress only -- drops executive-branch filers, who are out of scope.
    d = df[df["branch"] == C.KEEP_BRANCH]
    C.log_funnel("1. branch == congress", n0, len(d))

    # (2) Common-stock-like assets only. Options, bonds, municipal securities and
    #     funds have different return dynamics and cannot be priced the same way.
    n = len(d)
    d = d[d["asset_type"].isin(C.KEEP_ASSET_TYPES)]
    C.log_funnel("2. asset_type in {ST,Stock,CS}", n, len(d))

    # (3) Directional transactions only -- drops Exchange, which has no clean side.
    n = len(d)
    d = d[d["transaction_type"].isin(C.KEEP_TXN_TYPES)]
    C.log_funnel("3. transaction_type in {Buy,Sell}", n, len(d))

    # (4) Must have a ticker -- no ticker means we cannot price it.
    n = len(d)
    d = d[d["ticker"].notna() & (d["ticker"].astype(str).str.strip() != "")]
    C.log_funnel("4. ticker present", n, len(d))

    # (5) Both dates present. Asserted near-100%: the disclosure regime requires them.
    n = len(d)
    d = d.copy()
    d["transaction_date"] = pd.to_datetime(d["transaction_date"], errors="coerce")
    d["filing_date"] = pd.to_datetime(d["filing_date"], errors="coerce")
    d = d[d["transaction_date"].notna() & d["filing_date"].notna()]
    C.log_funnel("5. both dates parse", n, len(d))
    retained = len(d) / n if n else 0
    assert retained > 0.98, f"Date coverage {retained:.1%} below 98% -- investigate"
    print(f"     -> date coverage {retained:.2%} (assertion >98% passed)")

    # (6) filing_date < transaction_date is impossible (you cannot disclose a trade
    #     before making it). These are data-entry errors; dropping them protects the
    #     disclosure-date entry rule.
    n = len(d)
    d = d[d["filing_date"] >= d["transaction_date"]]
    C.log_funnel("6. filing_date >= transaction_date", n, len(d))

    print(f"  {'TOTAL':<38} {n0:>7,} -> {len(d):>7,}  (kept {len(d)/n0:5.1%})")
    return d


def clean(df):
    """Normalise tickers, add amount_mid and side."""
    print("\n=== CLEANING ===")
    d = df.copy()

    raw_tickers = d["ticker"].astype(str)
    d["ticker"] = raw_tickers.str.strip().str.upper()
    changed = (d["ticker"] != raw_tickers).sum()
    print(f"  Tickers uppercased/stripped: {changed:,} changed")

    # Log oddities rather than silently dropping them: a ticker that is not a plain
    # alphanumeric symbol may still be valid (e.g. BRK.B, class shares).
    odd = d[~d["ticker"].str.fullmatch(r"[A-Z0-9.\-]{1,8}")]
    if len(odd):
        vals = sorted(odd["ticker"].unique())
        print(f"  ODDITY: {len(odd):,} rows with {len(vals)} non-standard tickers "
              f"(kept, will fail price lookup if invalid): {vals[:15]}")
    else:
        print("  No non-standard tickers found")

    # Amount is disclosed as a RANGE only. Midpoint is the standard convention; it
    # is noisy, which is why equal-weight (not size-weight) is the default.
    d["amount_mid"] = (
        pd.to_numeric(d["amount_range_low"], errors="coerce")
        + pd.to_numeric(d["amount_range_high"], errors="coerce")
    ) / 2.0
    print(f"  amount_mid computed; {d['amount_mid'].isna().sum():,} missing")

    # side: +1 opens a long leg, -1 opens a short leg (used by long-short only).
    d["side"] = d["transaction_type"].map(
        lambda t: 1 if t in C.BUY_TYPES else (-1 if t in C.SELL_TYPES else 0)
    )
    assert (d["side"] != 0).all(), "Unmapped transaction_type survived filtering"
    print(f"  side: +1 buys = {(d['side'] == 1).sum():,}, "
          f"-1 sells = {(d['side'] == -1).sum():,}")

    # Disclosure lag -- the reason disclosure-date entry matters at all.
    lag = (d["filing_date"] - d["transaction_date"]).dt.days
    print(f"  Disclosure lag (days): median {lag.median():.0f}, "
          f"p75 {lag.quantile(.75):.0f}, p95 {lag.quantile(.95):.0f}, max {lag.max():.0f}")

    # Entry month = month of FILING date. This is the anti-look-ahead anchor.
    d["entry_month"] = C.month_end(d["filing_date"])

    drop_cols = [c for c in ("branch",) if c in d.columns]
    return d.drop(columns=drop_cols)


def main():
    ensure_raw_data()
    df = flatten()
    df = filter_trades(df)
    df = clean(df)

    out = C.CLEAN / "congress_trades.parquet"
    df.to_parquet(out, index=False)

    print("\n=== OUTPUT ===")
    print(f"  Saved {len(df):,} trades -> {out}")
    print(f"  Members: {df['filer_id'].nunique():,}   "
          f"Tickers: {df['ticker'].nunique():,}")
    print(f"  Date span (filing): {df['filing_date'].min():%Y-%m-%d} .. "
          f"{df['filing_date'].max():%Y-%m-%d}")
    print(f"  Chamber: {df['chamber'].value_counts().to_dict()}")
    print(f"  Party:   {df['party'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    main()
