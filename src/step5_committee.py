"""
Step 6 -- Track B selection (committee oversight).

Selects members by an EX-ANTE CHARACTERISTIC -- which committee they sat on --
restricts their trades to the sectors that committee oversees, and tests for alpha
there. The hypothesis is that any information advantage should concentrate where
a member has oversight.

ANTI-BIAS NOTE: selection here uses COMMITTEE MEMBERSHIP ONLY. No return, no
ranking, and no formation-window performance enters this step at any point. That
is what makes Track B a genuinely independent test rather than a second look at
the same data: it cannot be "lucky", because nothing about the outcome informed
the choice of who is in the group.

POINT-IN-TIME CORRECTNESS -- and a deviation from CLAUDE.md worth reading:
  CLAUDE.md Step 6 item 1 assumes @unitedstates/congress-legislators ships
  committee membership "by Congress". It does not. The repository contains only
  `committee-membership-current.yaml`, a single CURRENT snapshot (119th Congress).
  Using it directly would be a point-in-time violation: it would test a member's
  2020 trades against a committee they joined in 2025, and would silently drop
  every member who has since left Congress -- a survivorship filter on the
  selection variable itself.

  Resolution: reconstruct the historical snapshots from the file's GIT HISTORY.
  For each Congress in the evaluation window we pin the commit of
  committee-membership-current.yaml as it stood DURING that Congress, and read
  membership from that commit. This delivers exactly the by-Congress data
  CLAUDE.md specifies, from the source CLAUDE.md specifies.

Run:  python src/step5_committee.py [--sectors-only]
Out:  data/clean/committee_membership.parquet, data/clean/ticker_sectors.csv,
      data/clean/trackB_groups.parquet, data/reports/step6_committee.txt
"""
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

import warnings
warnings.filterwarnings("ignore")

CACHE = C.RAW / "legislators"
CACHE.mkdir(parents=True, exist_ok=True)
REPORT = C.REPORTS / "step6_committee.txt"

# Congress -> (start, end) and the commit of committee-membership-current.yaml
# pinned to a date INSIDE that Congress. Pinning by date, not by sha, keeps the
# choice mechanical and re-derivable.
CONGRESSES = {
    116: ("2019-01-03", "2021-01-02", "2020-05-01"),
    117: ("2021-01-03", "2023-01-02", "2021-10-01"),
    118: ("2023-01-03", "2025-01-02", "2023-10-01"),
}


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def _cached(name, url):
    fp = CACHE / name
    if not fp.exists():
        print(f"  fetching {name} ...")
        fp.write_bytes(_get(url))
    return yaml.safe_load(fp.read_bytes())


def commit_at(path, until):
    """SHA of the last commit to `path` on or before `until`."""
    u = (f"https://api.github.com/repos/unitedstates/congress-legislators/commits"
         f"?path={path}&until={until}T00:00:00Z&per_page=1")
    data = json.loads(_get(u))
    if not data:
        raise RuntimeError(f"No commit for {path} before {until}")
    return data[0]["sha"], data[0]["commit"]["committer"]["date"][:10]


def fetch_committee_snapshots():
    """Point-in-time committee membership for each Congress in the eval window."""
    snaps = {}
    for cong, (_s, _e, pin) in CONGRESSES.items():
        mem_f = CACHE / f"membership_{cong}.yaml"
        com_f = CACHE / f"committees_{cong}.yaml"
        if not (mem_f.exists() and com_f.exists()):
            sha, date = commit_at("committee-membership-current.yaml", pin)
            sha2, _ = commit_at("committees-current.yaml", pin)
            print(f"  Congress {cong}: pinned to {date} (sha {sha[:8]})")
            base = "https://raw.githubusercontent.com/unitedstates/congress-legislators/"
            mem_f.write_bytes(_get(base + sha + "/committee-membership-current.yaml"))
            com_f.write_bytes(_get(base + sha2 + "/committees-current.yaml"))
            time.sleep(1)
        snaps[cong] = {
            "membership": yaml.safe_load(mem_f.read_bytes()),
            "committees": yaml.safe_load(com_f.read_bytes()),
        }
        n = sum(len(v) for v in snaps[cong]["membership"].values())
        print(f"  Congress {cong}: {len(snaps[cong]['membership'])} committee units, "
              f"{n:,} member-slots")
    return snaps


# ----------------------------------------------------------------- name matching
def norm(s):
    """Lowercase, strip accents/punctuation/suffixes -- for name comparison only."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|md|phd|dds)\b\.?", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Common legal-name <-> familiar-name pairs. kadoa uses the legal name from the
# filing ("Daniel Crenshaw", "Cynthia Axne"); the legislator file uses the name
# the member goes by ("Dan", "Cindy"). Without this, ~25 sitting members fail to
# match purely on first name, which would silently shrink the committee sample.
NICKNAMES = [
    {"dan", "daniel", "danny"}, {"dave", "david"}, {"jim", "james", "jimmy"},
    {"ken", "kenneth"}, {"mike", "michael"}, {"bill", "william", "billy"},
    {"bob", "rob", "robert", "bobby", "robbie"}, {"rick", "dick", "richard", "richie"},
    {"tom", "thomas", "tommy"}, {"tony", "anthony"}, {"chris", "christopher"},
    {"greg", "gregory"}, {"joe", "joseph", "joey"}, {"steve", "steven", "stephen"},
    {"matt", "matthew"}, {"nick", "nicholas"}, {"pete", "peter"},
    {"ed", "ted", "edward", "eddie"}, {"tim", "timothy"}, {"ron", "ronald"},
    {"don", "donald", "donnie"}, {"jack", "john", "johnny", "jonathan"},
    {"cindy", "cynthia"}, {"debbie", "deborah", "debra"},
    {"liz", "beth", "betsy", "elizabeth"}, {"kathy", "kate", "katie", "katherine",
    "kathleen", "catherine"}, {"sue", "susan", "suzy"}, {"jen", "jenny", "jennifer"},
    {"pat", "patrick", "patricia", "patty"}, {"andy", "andrew"},
    {"ben", "benjamin", "benji"}, {"sam", "samuel", "sammy"},
    {"alex", "alexander", "alexandria"}, {"charlie", "chuck", "charles"},
    {"frank", "francis", "frankie"}, {"hank", "henry"}, {"larry", "lawrence"},
    {"lou", "louis"}, {"marty", "martin"}, {"phil", "philip", "phillip"},
    {"randy", "randall"}, {"russ", "russell"}, {"sandy", "sandra"},
    {"terry", "terrence", "terence"}, {"vern", "vernon"}, {"walt", "walter"},
    {"will", "willie", "william"}, {"gus", "august"}, {"abby", "abigail"},
    {"maggie", "margaret", "meg", "peggy"}, {"nancy", "ann", "anne"},
    {"barb", "barbara"}, {"carol", "caroline", "carolyn"}, {"jeff", "jeffrey"},
    {"doug", "douglas"}, {"brad", "bradley"}, {"rod", "rodney"}, {"gil", "gilbert"},
]
_NICK = {}
for _grp in NICKNAMES:
    for _n in _grp:
        _NICK.setdefault(_n, set()).update(_grp)


def first_equiv(a, b):
    """True if two given names plausibly denote the same person."""
    if not a or not b:
        return False
    if a == b:
        return True
    if b in _NICK.get(a, set()):
        return True
    # Initial matching is allowed ONLY when the filing side is itself an initial
    # (e.g. kadoa's "A. Mitchell McConnell"). Allowing it in the other direction
    # would let a legislator's MIDDLE initial match any given name with the same
    # letter -- which made John D. Dingell tie with Debbie Dingell.
    if len(a) == 1:
        return b.startswith(a)
    if len(b) == 1:
        return False
    return a.startswith(b[:3]) or b.startswith(a[:3])


def surname_candidates(parts):
    """Possible surnames for a full name, longest first.

    Congressional names include compound surnames that carry no hyphen
    ("Wasserman Schultz", "Leger Fernandez", "Van Epps", "O'Halleran"), so the
    last WORD is often not the surname.
    """
    cands = []
    for k in (3, 2, 1):
        if len(parts) >= k:
            cands.append(" ".join(parts[-k:]))
    # Fallback for married/maiden orderings such as "Ashley Hinson Arenholz",
    # where the surname sits in the middle.
    for i in range(1, len(parts) - 1):
        cands.append(parts[i])
    return cands


def legislator_index():
    """Every legislator (current + historical) keyed for matching."""
    cur = _cached("legislators-current.yaml", C.LEGISLATORS_BASE + "legislators-current.yaml")
    hist = _cached("legislators-historical.yaml",
                   C.LEGISLATORS_BASE + "legislators-historical.yaml")
    recs = []
    for L in list(cur) + list(hist):
        bio = L.get("id", {}).get("bioguide")
        if not bio:
            continue
        nm = L.get("name", {})
        terms = L.get("terms", [])
        if not terms:
            continue
        # Only legislators serving at any point in our study span can match.
        last_end = max(t.get("end", "1900") for t in terms)
        if last_end < C.FORMATION_START:
            continue
        states = {t.get("state") for t in terms}
        chambers = {"house" if t.get("type") == "rep" else "senate" for t in terms}
        firsts = {nm.get("first"), nm.get("nickname"), nm.get("middle")}
        official = nm.get("official_full") or ""
        recs.append({
            "bioguide": bio,
            "last": norm(nm.get("last")),
            "firsts": {norm(f) for f in firsts if f},
            "official": norm(official),
            "states": states,
            "chambers": chambers,
        })
    print(f"  Legislator index: {len(recs):,} candidates active since {C.FORMATION_START[:4]}")
    return recs


def match_members(trades, legs):
    """Join kadoa filer_id -> bioguide by name + chamber + state.

    Reported as a match RATE with every unmatched member listed. The join is the
    crux of Track B: an unmatched member is silently excluded from the committee
    test, so the exclusion has to be visible.
    """
    members = trades.drop_duplicates("filer_id")[
        ["filer_id", "full_name", "chamber", "state", "party"]]

    rows, unmatched, review = [], [], []
    for _, m in members.iterrows():
        fn = norm(m["full_name"])
        parts = fn.split()
        if not parts:
            unmatched.append((m["filer_id"], m["full_name"], "unparseable"))
            continue
        first = parts[0]
        surnames = surname_candidates(parts)
        ch, st = m["chamber"], m["state"]

        def score(r):
            if r["last"] not in surnames:
                return -1
            s = 0
            # First name, allowing nicknames and initials.
            if any(first_equiv(first, f) for f in r["firsts"]) or \
               (r["official"] and first_equiv(first, r["official"].split()[0])):
                s += 3
            # State is a strong discriminator: two members rarely share a surname
            # AND a state AND a chamber.
            if st in r["states"]:
                s += 3
            if ch in r["chambers"]:
                s += 1
            return s

        scored = sorted(((score(r), r) for r in legs), key=lambda x: -x[0])
        scored = [(s, r) for s, r in scored if s >= 0]

        # Accept at >=4, which requires surname PLUS either (state AND chamber) or
        # (first name AND chamber). Never accept a tie -- an ambiguous member is
        # reported as unmatched rather than guessed at.
        if scored and scored[0][0] >= 4:
            best_s = scored[0][0]
            tied = [r for s, r in scored if s == best_s]
            if len(tied) > 1:
                unmatched.append((m["filer_id"], m["full_name"],
                                  f"AMBIGUOUS: {[t['bioguide'] for t in tied]}"))
                continue
            rows.append({"filer_id": m["filer_id"], "bioguide": tied[0]["bioguide"],
                         "full_name": m["full_name"], "chamber": ch, "state": st,
                         "party": m["party"], "match_score": best_s})
            # Anything short of a perfect 7 (name + state + chamber) is flagged for
            # hand-verification, as CLAUDE.md requires.
            if best_s < 7:
                review.append((m["filer_id"], m["full_name"],
                               tied[0]["bioguide"], best_s))
        else:
            why = f"best score {scored[0][0]}" if scored else "no surname match"
            unmatched.append((m["filer_id"], m["full_name"], why))

    mm = pd.DataFrame(rows)
    return mm, unmatched, len(members), review


# ----------------------------------------------------------------- sector map
def load_sector_map():
    with open(C.CONFIG / "sector_map.yaml") as fh:
        return yaml.safe_load(fh)


def committee_sectors(smap, variant=None):
    """key -> sectors, applying a named robustness variant if given."""
    out = {r["key"]: list(r["sectors"]) for r in smap["baseline"]}
    if variant:
        ov = smap["variants"][variant]["override"]
        out.update({k: list(v) for k, v in ov.items()})
    return out


def resolve_committees(snaps, smap):
    """(congress, bioguide) -> set of mapped committee keys, point-in-time.

    Most rows match at FULL-COMMITTEE level. The health row matches at
    SUBCOMMITTEE level on the House side, because the House has no single health
    committee -- health jurisdiction sits in the health subcommittees of Ways &
    Means and of Energy & Commerce. Matching those full committees instead would
    sweep in every tax-writing member and dilute the oversight claim.
    """
    rows = []
    for cong, snap in snaps.items():
        # Resolve each membership unit to (chamber, parent name, subcommittee name).
        # Subcommittee units are keyed parent_code + subcommittee_code, e.g. HSWM02.
        units = {}
        for c in snap["committees"]:
            tid = c.get("thomas_id")
            if not tid:
                continue
            units[tid] = (c.get("type"), c.get("name", ""), None)
            for sc in (c.get("subcommittees") or []):
                stid = sc.get("thomas_id")
                if stid:
                    units[tid + stid] = (c.get("type"), c.get("name", ""),
                                         sc.get("name", ""))

        for unit, members in snap["membership"].items():
            if unit not in units:
                continue
            chamber, name, subname = units[unit]

            key = None
            for row in smap["baseline"]:
                if subname is None:
                    # Full-committee unit. Subcommittee members also appear in
                    # their full-committee unit, so nothing is lost by not
                    # folding subcommittees upward.
                    pats = row.get("match", {}).get(chamber) or []
                    if any(p.lower() in name.lower() for p in pats):
                        key = row["key"]
                        break
                else:
                    # Subcommittee unit: matches only rows that explicitly ask for
                    # subcommittee granularity.
                    for sm in (row.get("subcommittee_match") or {}).get(chamber, []):
                        if (sm["committee"].lower() in name.lower()
                                and sm["subcommittee"].lower() in (subname or "").lower()):
                            key = row["key"]
                            break
                    if key:
                        break
            if key is None:
                continue

            label = name if subname is None else f"{name} -- {subname} subcmte"
            for mem in members:
                bio = mem.get("bioguide")
                if bio:
                    rows.append({"congress": cong, "bioguide": bio,
                                 "committee_key": key, "committee_name": label,
                                 "chamber": chamber})
    df = pd.DataFrame(rows).drop_duplicates(
        ["congress", "bioguide", "committee_key"])
    return df


# ----------------------------------------------------------------- ticker sectors
def fetch_ticker_sectors(tickers, batch_sleep=0.35):
    """Ticker -> GICS-style sector via yfinance .info, CACHED to disk.

    DECIDED 2026-09-14 with the user: yfinance `.info`, pulled once and cached, so
    every later run reads only the cache and results are reproducible. Limitations
    to state in the write-up: yfinance sectors are a GICS-LIKE approximation, not
    licensed GICS, and they are a CURRENT snapshot -- a company reclassified since
    2020 carries today's label. Both are logged rather than papered over.
    """
    import yfinance as yf

    have = {}
    if C.SECTOR_CACHE.exists():
        d = pd.read_csv(C.SECTOR_CACHE)
        have = dict(zip(d["ticker"], d["sector"]))
    todo = [t for t in tickers if t not in have]
    print(f"  Sector cache: {len(have):,} known, {len(todo):,} to fetch")

    # Rate limiting is the real hazard here: if Yahoo throttles us, every ticker
    # comes back sector-less and Track B would quietly collapse to nothing while
    # still "running". So we distinguish a genuine empty sector from a throttled
    # request, retry the latter, and back off when failures cluster.
    consec_fail = 0
    for i, t in enumerate(todo, 1):
        # Try the ticker's rename aliases too, so a symbol that changed (ANTM ->
        # ELV) still resolves a sector instead of being dropped as UNKNOWN.
        from step2_data import alias_candidates
        sec, threw = None, False
        for sym in [t] + alias_candidates(t):
            for attempt in range(3):
                try:
                    info = yf.Ticker(sym).info or {}
                    sec = info.get("sector")
                    threw = False
                    break
                except Exception:
                    threw = True
                    time.sleep(2 ** attempt)
            if sec:
                break
        if threw or not sec:
            consec_fail += 1
        else:
            consec_fail = 0
        if consec_fail and consec_fail % 25 == 0:
            # Sustained failure almost certainly means throttling, not 25
            # consecutive sector-less companies. Pause rather than burn the list.
            print(f"    {consec_fail} consecutive failures -- backing off 60s")
            time.sleep(60)
        have[t] = sec if sec else "UNKNOWN"
        if i % 100 == 0:
            known = sum(1 for v in have.values() if v != "UNKNOWN")
            print(f"    {i:,}/{len(todo):,} fetched; {known:,} with a sector")
            pd.DataFrame({"ticker": list(have), "sector": list(have.values())}).to_csv(
                C.SECTOR_CACHE, index=False)
        time.sleep(batch_sleep)

    df = pd.DataFrame({"ticker": list(have), "sector": list(have.values())})
    df.to_csv(C.SECTOR_CACHE, index=False)
    print(f"  Saved {len(df):,} ticker sectors -> {C.SECTOR_CACHE}")
    return dict(zip(df["ticker"], df["sector"]))


# yfinance reports its OWN sector taxonomy, which is NOT the GICS vocabulary the
# committed sector map is written in. Five of the ten mapped sectors have
# different names ("Financial Services" vs "Financials", "Healthcare" vs
# "Health Care", "Technology" vs "Information Technology", ...). Without this
# translation the map silently matches nothing for those sectors, and whole
# committee groups -- health and agriculture -- vanish from Track B while the
# pipeline still appears to run correctly. `assert_map_vocabulary` below exists
# so that this class of silent failure cannot recur unnoticed.
YF_TO_GICS = {
    "Financial Services": "Financials",
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    # Already identical to their GICS names:
    "Industrials": "Industrials",
    "Real Estate": "Real Estate",
    "Energy": "Energy",
    "Communication Services": "Communication Services",
    "Utilities": "Utilities",
}


def to_gics(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "UNKNOWN"
    return YF_TO_GICS.get(s, s)


def assert_map_vocabulary(smap, observed, out):
    """Fail loudly if the sector map names a sector the data can never produce.

    A typo or taxonomy drift in either the map or the price source would
    otherwise just quietly shrink Track B to nothing.
    """
    wanted = set()
    for row in smap["baseline"]:
        wanted.update(row["sectors"])
    for v in smap.get("variants", {}).values():
        for sl in v.get("override", {}).values():
            wanted.update(sl)
    missing = sorted(wanted - observed - {"UNKNOWN"})
    out(f"   Sector vocabulary check: map names {len(wanted)} sectors; "
        f"{len(wanted) - len(missing)} are present in the data")
    if missing:
        out(f"   *** WARNING: these mapped sectors NEVER appear after translation: "
            f"{missing}")
        out( "   *** Any committee row relying on them will test zero trades.")
    return missing


def congress_of(ts):
    for cong, (s, e, _p) in CONGRESSES.items():
        if pd.Timestamp(s) <= ts <= pd.Timestamp(e):
            return cong
    return None


def main():
    trades = pd.read_parquet(C.CLEAN / "congress_trades.parquet")
    smap = load_sector_map()

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out("STEP 6 -- TRACK B SELECTION (committee oversight)")
    out("=" * 78)
    out("Selection variable: COMMITTEE MEMBERSHIP ONLY. No return is consulted.")
    out()

    out("1. Point-in-time committee snapshots (reconstructed from git history --")
    out("   upstream ships only a current snapshot; see module docstring)")
    snaps = fetch_committee_snapshots()
    out()

    out("2. Name-matching kadoa filer_id -> bioguide")
    legs = legislator_index()
    mm, unmatched, n_members, review = match_members(trades, legs)
    rate = len(mm) / n_members
    out(f"   MATCH RATE: {len(mm)}/{n_members} = {rate:.1%}")
    out(f"   Unmatched ({len(unmatched)}) -- EXCLUDED from Track B:")
    for fid, nm, why in unmatched:
        out(f"     {fid:<42} {str(nm):<26} {why}")
    out()
    out(f"   Matched but below a perfect name+state+chamber agreement "
        f"({len(review)}) -- flagged for hand-verification:")
    for fid, nm, bio, sc in review:
        out(f"     {fid:<42} {str(nm):<26} -> {bio} (score {sc}/7)")
    mm.to_parquet(C.CLEAN / "member_bioguide_match.parquet", index=False)
    out()

    out("3. Resolving committee membership through the COMMITTED sector map")
    cs = resolve_committees(snaps, smap)
    joined = cs.merge(mm[["filer_id", "bioguide", "party", "full_name"]],
                      on="bioguide", how="inner")
    out(f"   Mapped committee-member rows: {len(cs):,}")
    out(f"   ... of which match a trading member: {len(joined):,} "
        f"({joined['filer_id'].nunique()} distinct members)")
    out()
    out("   Members per committee group, by Congress:")
    piv = joined.pivot_table(index="committee_key", columns="congress",
                             values="filer_id", aggfunc="nunique", fill_value=0)
    out(piv.to_string())
    joined.to_parquet(C.CLEAN / "committee_membership.parquet", index=False)
    out()

    # ---- restrict to in-sector trades, point-in-time --------------------------
    out("4. Ticker -> GICS sector (yfinance .info, cached)")
    ev = trades[(trades["filing_date"] >= C.EVAL_START) &
                (trades["filing_date"] <= C.EVAL_END)].copy()
    ev["congress"] = ev["filing_date"].apply(congress_of)
    oncom = ev.merge(joined[["filer_id", "congress", "committee_key"]],
                     on=["filer_id", "congress"], how="inner")
    # Only priced tickers can contribute a return, so only they need a sector.
    # This is an efficiency filter, not a methodological one: an unpriced ticker
    # is dropped by the holding panel regardless of what sector it belongs to.
    priced = set(pd.read_parquet(C.CLEAN / "returns_monthly.parquet").columns)
    all_tk = sorted(oncom["ticker"].unique())
    need = [t for t in all_tk if t in priced]
    out(f"   On-committee eval-window trades: {len(oncom):,} "
        f"({len(all_tk):,} distinct tickers, of which {len(need):,} are priced "
        f"and therefore need a sector)")

    sectors = fetch_ticker_sectors(need)
    # Translate the price source's taxonomy into the GICS vocabulary the committed
    # map is written in (see YF_TO_GICS).
    oncom["sector_raw"] = oncom["ticker"].map(sectors)
    oncom["sector"] = oncom["sector_raw"].map(to_gics)
    assert_map_vocabulary(smap, set(oncom["sector"].dropna().unique()), out)
    n_unknown = (oncom["sector"].isin(["UNKNOWN"]) | oncom["sector"].isna()).sum()
    out(f"   Sector mapped: {len(oncom)-n_unknown:,}/{len(oncom):,} trade-rows "
        f"({1-n_unknown/max(len(oncom),1):.1%}); {n_unknown:,} UNKNOWN (dropped)")
    unk = sorted(oncom.loc[oncom["sector"].isin(["UNKNOWN"]) |
                           oncom["sector"].isna(), "ticker"].unique())
    out(f"   Unmapped tickers ({len(unk)}): {unk[:40]}")
    out()

    out("5. Keeping only trades where committee sector == ticker sector")
    csec = committee_sectors(smap)
    oncom["overseen"] = oncom.apply(
        lambda r: r["sector"] in csec.get(r["committee_key"], []), axis=1)
    insec = oncom[oncom["overseen"]].copy()
    out(f"   On-committee, IN-SECTOR trades: {len(insec):,} "
        f"({insec['filer_id'].nunique()} members)")
    out()
    out("   Per-committee group sizes (evaluation window):")
    g = insec.groupby("committee_key").agg(
        trades=("trade_id", "size"), members=("filer_id", "nunique"),
        tickers=("ticker", "nunique"))
    out(g.to_string())
    out()
    out("   NOTE: samples are small by construction. Step 8 therefore reports 95%")
    out("   confidence intervals, not just t-statistics -- a wide CI around zero is")
    out("   honest 'underpowered', not a null result to oversell.")

    insec.to_parquet(C.CLEAN / "trackB_groups.parquet", index=False)
    # Save the PRE-FILTER table (on-committee trades with sectors attached, before
    # the committee->sector map is applied) so Step 9 can re-apply the robustness
    # variants of the map without re-fetching anything.
    oncom.to_parquet(C.CLEAN / "trackB_oncommittee_all.parquet", index=False)
    out()
    out(f"Saved -> {C.CLEAN / 'trackB_groups.parquet'}")
    REPORT.write_text("\n".join(lines))
    print(f"Report -> {REPORT}")


if __name__ == "__main__":
    if "--sectors-only" in sys.argv:
        tr = pd.read_parquet(C.CLEAN / "congress_trades.parquet")
        ev = tr[(tr["filing_date"] >= C.EVAL_START) & (tr["filing_date"] <= C.EVAL_END)]
        fetch_ticker_sectors(sorted(ev["ticker"].unique()))
    else:
        main()
