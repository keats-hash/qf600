"""
QF600 — shared configuration.

Every committed methodological choice lives here so that no downstream module can
quietly re-tune it. Anything in this file was decided on STRUCTURAL grounds BEFORE
any alpha was computed (see CLAUDE.md "Decisions log"). Changing a value here
changes every step at once, which is the point: it makes tuning visible.

DO NOT edit these to improve alpha. A change is legitimate only with a logged,
non-results-based reason (e.g. a trade-count diagnostic).
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "data" / "reports"
CONFIG = ROOT / "config"
for _d in (RAW, CLEAN, REPORTS, CONFIG):
    _d.mkdir(parents=True, exist_ok=True)

KADOA_DIR = RAW / "congress-trading-monitor-main" / "public" / "data" / "filer"
KADOA_TARBALL_URL = (
    "https://codeload.github.com/kadoa-org/congress-trading-monitor/tar.gz/refs/heads/main"
)

# ---------------------------------------------------------------- the split
# COMMITTED from step1_diagnostic.py (counts and dates only, never returns).
# Formation window is used for SELECTION ONLY. Evaluation window is used for
# TESTING ONLY. They do not overlap -- this is the whole anti-data-snooping design.
FORMATION_START = "2014-01-01"
FORMATION_END = "2019-12-31"
EVAL_START = "2020-01-01"
EVAL_END = "2024-12-31"

# ---------------------------------------------------------------- eligibility
# Floor set by the trade-count diagnostic, not by looking at returns.
MIN_FORMATION_TRADES = 10   # >=10 disclosed trades inside the formation window
MIN_INVESTED_MONTHS = 12    # >=12 formation months with at least one open leg
TOP_QUANTILE = 0.25         # Track A selects the top quartile of the eligible pool

# ---------------------------------------------------------------- holding rule
# PRIMARY RULE: fixed 12-month window from the FILING (disclosure) date.
# Exit is a pure function of entry, so it injects no forward-looking information.
HOLDING_MONTHS = 12

# Entry lag, in months, between the filing month and the first month a leg earns.
# =1 means a trade disclosed in month t earns returns in t+1..t+12. Rationale: a
# filing lands mid-month, so crediting month t's FULL return would capture the
# pre-disclosure part of that month -- a partial look-ahead. Lag 1 is the strictly
# conservative reading of CLAUDE.md's "open from its entry month onward"; it can
# only reduce measured alpha. The lag-0 variant is reported in Step 9.
ENTRY_LAG_MONTHS = 1

# ---------------------------------------------------------------- data filters
KEEP_BRANCH = "congress"
KEEP_ASSET_TYPES = {"ST", "Stock", "CS"}
KEEP_TXN_TYPES = {"Purchase", "Sale (Full)", "Sale (Partial)"}
BUY_TYPES = {"Purchase"}
SELL_TYPES = {"Sale (Full)", "Sale (Partial)"}

# ---------------------------------------------------------------- prices
PRICE_START = "2013-01-01"   # pad before formation so 12m windows have coverage
PRICE_END = "2025-12-31"
FF_FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)
FF_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)

# ---------------------------------------------------------------- committee track
LEGISLATORS_BASE = (
    "https://raw.githubusercontent.com/unitedstates/congress-legislators/main/"
)
# Ticker -> GICS sector reference source. DECIDED 2026-09-14 with the user:
# yfinance `.info` sector field, pulled once and cached to disk; every later run
# reads only the cache, so results are reproducible.
SECTOR_SOURCE = "yfinance_cached"
SECTOR_CACHE = CLEAN / "ticker_sectors.csv"

# Headline construction for the write-up. DECIDED 2026-09-14 with the user.
# Both constructions are always computed and always reported; this only sets
# which one leads the narrative.
HEADLINE_CONSTRUCTION = "long_only"

# ---------------------------------------------------------------- regression
NEWEY_WEST_LAGS = 6   # HAC lags for monthly data (~6 months of autocorrelation)
ANNUALISE = 12        # monthly alpha -> annual


def month_end(s):
    """Normalise any datetime-like series to month-end timestamps."""
    import pandas as pd
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("M")


def log_funnel(name, before, after, note=""):
    """Print one auditable line of the filter funnel."""
    dropped = before - after
    pct = (dropped / before * 100) if before else 0.0
    print(f"  {name:<38} {before:>7,} -> {after:>7,}  (dropped {dropped:>6,}, {pct:5.1f}%) {note}")

# test alpha feature 1 changes
# test alpha feature 1 changes 2
