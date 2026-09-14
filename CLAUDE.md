# CLAUDE.md — QF600 Alpha Homework

## Project

Backtest a portfolio strategy and test whether it produces **risk-adjusted alpha**, per the
QF600 (Asset Pricing) assignment: separate skill (α) from market exposure (β) via a CAPM
regression.

**Strategy under test:** Do a subset of U.S. Congress members generate risk-adjusted alpha in
their disclosed personal trades — either members who traded well in the past (formation
ranking), or members positioned to have an information edge (committee oversight)?

**Language / env:** Python, run in VS Code, local virtual environment. Core libs: pandas,
numpy, statsmodels, yfinance, pyyaml.

---

## The core method — READ THIS BEFORE WRITING ANY CODE

Two selection hypotheses, both tested the same honest way — an **out-of-sample split** where
selection is quarantined from evaluation:

- **Track A — performance selection.** Rank members by trading performance in a **formation
  window**; test the selected group in a later, non-overlapping **evaluation window**. Measures
  whether past winners *persist* (skill) rather than were *lucky* (noise).
- **Track B — committee selection.** Select members by an **ex-ante characteristic** (committee
  membership), restrict to the sectors that committee oversees, and test for alpha there. Tests
  whether information advantage concentrates where members have oversight. Selection uses
  committee membership only — never returns.

Both tracks share one construction (below) and one regression (below). Report both, in full,
next to each other.

### Hard rules (do not violate, even if asked to "just get a result")

- **Selection never touches evaluation-window returns.** Track A ranks on formation data only;
  Track B selects on committee membership only. Neither may peek at the outcome being tested.
- **Never tune the split, the holding rule, the weighting, or the sector map to improve alpha.**
  Every such choice is decided on structural grounds and committed *before* seeing the alpha.
  A choice may change only for a logged, non-results-based reason (e.g. a trade-count diagnostic).
- **Enter on the DISCLOSURE (filing) date, not the trade date.** Filings lag the trade by up to
  ~45 days; trade-date entry is look-ahead bias.
- **Exit is mechanical** (a fixed function of entry + holding window) so it adds no forward info.
- **No look-ahead anywhere.** A position contributes to month *t* only if it was disclosed and
  open by *t*.
- **Regime/era questions never move the split.** They go through the fixed-selection subperiod /
  rolling-alpha analysis (Step 8), not by re-selecting on a chosen era.
- **Report all variants, not the flattering one.** With multiple specifications the risk of a
  chance "significant" result rises; the defence is showing every run beside the baseline.
- If a requested change would break a rule, flag it and explain rather than silently doing it.

---

## Pipeline overview

| Step | Module | Purpose |
|---|---|---|
| 1 | `step1_load.py` | Parse + filter + clean disclosures → tidy trade table. |
| 2 | `step1_diagnostic.py` | Trade-count diagnostic → fixes split boundary + eligibility floor (dates/counts only). |
| 3 | `step2_data.py` | Fetch monthly prices (yfinance) + Ken French factors; build the return-construction layer. |
| 4 | `step3_holding.py` | Turn (ticker, filing_date, side) into a **held-position** monthly panel under the holding rule. Foundational — everything downstream uses its output. |
| 5 | `step4_formation.py` | Track A selection: rank eligible members on formation-window return → top quartile. |
| 6 | `step5_committee.py` | Track B selection: fetch committee data, join to members, apply sector map → on-committee/in-sector groups. |
| 7 | `step6_portfolio.py` | Build evaluation-window monthly return series for each group, **long-only and long–short**. |
| 8 | `step7_regression.py` | CAPM (and 4-factor) regression w/ Newey-West; α, t, β, R², CI. |
| 9 | `step8_robustness.py` | Subperiod + rolling-alpha, weighting variants, split-boundary variants. All reported. |

Keep modules separate and independently runnable; cache intermediate tables to `data/clean/`.
Comment every anti-bias point inline so it survives future edits.

---

## Step 1 — Load, filter, clean  (`step1_load.py`)

Data source is LOCKED (see "Data source" below). Produce a tidy per-trade table.

- Download the kadoa repo tarball once to `data/raw/`; use the **per-filer files**
  `public/data/filer/*.json` (447 files, full 2015–2026 archive) — NOT top-level `trades.json`
  (truncated live feed).
- Flatten one row per trade; attach filer-level `filer_id`, `party`, `chamber`, `branch`.
- Parse `transaction_date`, `filing_date` as datetimes.
- **Filter (log the row count dropped at each step — the funnel must be auditable):**
  1. `branch == "congress"`
  2. `asset_type` in {`ST`, `Stock`, `CS`}
  3. `transaction_type` in {`Purchase`, `Sale (Full)`, `Sale (Partial)`}
  4. `ticker` present
  5. both dates present (assert ~100%)
  6. drop `filing_date < transaction_date` (data errors)
- **Clean:** uppercase/strip tickers (log oddities, don't silently drop); add
  `amount_mid = (low+high)/2`; add `side` = +1 Purchase / −1 Sale.
- **IGNORE the shipped return fields** (`ret_since`, `excess_since`, `ret_1y`, …) — we compute
  returns ourselves off disclosure-date entry.
- Save `data/clean/congress_trades.parquet`.

## Step 2 — Trade-count diagnostic  (`step1_diagnostic.py`)

Counts and dates ONLY — never returns. Fixes the split and the eligibility floor before any
alpha is computed. Print per candidate window: distinct members, trades-per-member distribution
(min/p25/median/p75/max), members clearing ≥5/≥10/≥20 trades, party breakdown, and
formation→evaluation retention. Commit the split + floor from this, then stop and log it.

## Step 3 — Prices & factors  (`step2_data.py`)

- Fetch monthly **adjusted-close** prices (yfinance, `auto_adjust=True`) for the union of
  tickers traded by the eligible universe (both tracks), over the full 2015–2024 span.
- Fetch Ken French **Mkt-RF, SMB, HML, MOM, RF** (monthly). Align all to month-end.
- Build a helper: monthly simple return per ticker. Log tickers with no price coverage; a trade
  in an unpriced ticker is dropped (log per member).

## Step 4 — Holding-period construction  (`step3_holding.py`)  ← FOUNDATIONAL

Convert discrete disclosed trades into a **monthly held-position panel** with an explicit exit.

- **Primary rule: fixed 12-month holding window from `filing_date`.** A purchase opens a +1
  (long) position in that ticker for the 12 months following disclosure; a sale opens a −1
  (short) leg for 12 months (used by the long–short construction; long-only ignores the short
  legs). Positions roll off after the window.
- Entry month = month of `filing_date`. No look-ahead: a position is "open" only from its entry
  month onward.
- Output, per member per month: the set of open legs (ticker, sign, months-held), used
  downstream to compute that member's monthly return.
- **Recompute eligibility under this rule.** "Invested months" = months a member has ≥1 open
  leg. The ≥12-invested-month floor is re-applied here, so the eligible pool (and therefore the
  Track-A selected group) may change vs the Phase-1 long-only run. **Log any change to the
  selected member list** — downstream deck/summary must match.
- Alternatives considered and rejected (note in write-up, don't implement as default):
  hold-until-disclosed-closing-trade (closing trades sparsely/late disclosed);
  hold-to-next-rebalance.

## Step 5 — Track A selection  (`step4_formation.py`)

- Eligible pool = members with ≥10 formation-window trades AND ≥12 invested months (post
  Step-4 recompute).
- Rank by mean monthly (long-only) return over the **formation window only**.
- Select **top quartile**. Log the ranked pool and the selected set. Selection uses formation
  data only — assert no evaluation month enters this step.

## Step 6 — Track B selection  (`step5_committee.py`)

**Data acquisition (this is the real work of the committee track):**

1. Fetch the **@unitedstates/congress-legislators** GitHub data (tarball or raw files;
   github.com / raw.githubusercontent.com are network-whitelisted). Use
   `legislators-current.yaml` + `legislators-historical.yaml` (names, bioguide IDs, terms) and
   the committee-membership files (member → committee, **by Congress** — assignments change per
   session).
2. **Join to kadoa members.** kadoa `filer_id` (e.g. `house_bradleys_schneider`) has no bioguide
   ID → build a name-match (normalise names; match on last name + first name + chamber; hand-
   verify ambiguous cases; log unmatched members). This join is the crux — report its match rate.
3. **Sector map — COMMITTED (author-defined; there is no dataset for this). Do not change it to
   improve alpha.** Granularity = **GICS sector level** (not industry level) — deliberate, so
   groups stay large enough for any statistical power. Each row is included only because it has a
   defensible oversight→sector rationale (the one-line justification is the test for inclusion;
   no row without one).

   ```yaml
   # committee (House / Senate equivalent)            -> GICS sector(s)
   Financial Services / Banking, Housing & Urban Affairs:
       sectors: [Financials, Real Estate]
       why: writes and oversees banking, securities and housing-finance rules
   Armed Services:
       sectors: [Industrials]          # defense primes & aerospace sit in Industrials under GICS
       why: sets the defense budget and weapons-procurement policy
   Energy & Commerce / Energy & Natural Resources:
       sectors: [Energy, Utilities]
       why: oversees energy production, grid and utility regulation
       note: BROAD committee — also touches Health Care and Communication Services;
             breadth logged as a limitation (see below)
   Health, Education, Labor & Pensions (HELP) / Ways & Means health, E&C health subcmte:
       sectors: [Health Care]          # incl. pharma & biotech (GICS puts them in Health Care)
       why: oversees drug approval pathway, Medicare/Medicaid, health regulation
   Agriculture:
       sectors: [Consumer Staples]     # food & agribusiness
       why: sets farm policy, subsidies and food regulation
   Science, Space & Technology / Commerce, Science & Transportation:
       sectors: [Information Technology, Communication Services]
       why: oversees tech, telecom and communications policy
   Transportation & Infrastructure:
       sectors: [Industrials, Materials]
       why: sets transport, infrastructure and public-works policy
   ```

   Map to SECTORS, never to hand-picked tickers — picking names would smuggle outcome selection
   back in. Keep it rules-based: committee→sector here, ticker→sector in item 4, test only where
   the two match.

4. **Ticker → GICS sector mapping.** Attach a sector to each traded ticker via a reference
   source (e.g. a yfinance `.info` sector field, or a static GICS/sector lookup table). Log the
   mapping method, the share of tickers successfully mapped, and unmapped tickers (dropped, with
   count per group). Prefer a cached static lookup over per-ticker live calls for reproducibility.

5. **Point-in-time correctness.** A member's committee can change across Congresses, and so does
   the sector they should be tested against. Use the by-Congress committee data: test each
   member's trades against the sector(s) of the committee they sat on **at the time of that
   trade's disclosure**, not a single current snapshot.

**Test construction:**

- For each committee, restrict that committee's members' trades to tickers in the overseen
  sector(s); build the group's held-position panel (Step 4 rule) over the evaluation window.
- Produce per-committee groups AND a combined "on-committee, in-sector" portfolio.
- Selection uses committee membership only. Same disclosure-date entry + holding rule.
- Samples will be small → the regression step must report confidence intervals, not just t.
- **Map-robustness check (marks-earning):** re-run with one or two debatable mappings varied
  (e.g. assign Energy & Commerce to Health Care instead of Energy; drop Real Estate from
  Financials) and show the conclusion holds. This converts "you chose the map to get the answer"
  into "the answer survives reasonable map choices." Report all variants.
- **Limitation to log:** broad committees (esp. Energy & Commerce) span multiple sectors, so the
  committee→sector link is coarser for them; note this rather than forcing a tidy one-to-one.

## Step 7 — Portfolio return series  (`step6_portfolio.py`)

For every group (Track-A selected, each Track-B committee group, combined), over the
**evaluation window**, build monthly return series in two constructions:

- **Long-only:** equal-weight across each member's open long legs, then equal-weight across
  active members. (This is the Phase-1-comparable baseline.)
- **Long–short:** long open purchase legs, short open sale legs; aggregate as the net weighted
  position return, equal-weight across open legs, then across active members. Handle a member
  simultaneously long some names / short others in the same month. Expect β to fall toward 0 if
  genuine directional timing exists; if β stays high the "buys" were just market exposure.
- Weighting default = equal-weight; **midpoint-size weighting** (using `amount_mid`) is a
  reported robustness variant, not the default.
- Save each series to `data/clean/`.

## Step 8 — Regression  (`step7_regression.py`)

For each group × construction:

- **CAPM:** `R_p − R_f = α + β(R_M − R_f) + ε`, monthly, **Newey-West (HAC)** standard errors.
- **4-factor control:** add SMB, HML, MOM. If α survives the market but dies against factors →
  "alpha decays into beta" — report both.
- Report α (annualised = ×12), its t-stat **and 95% CI**, β, R², N. With small samples the CI is
  essential: a wide CI around zero is honest "underpowered," not a null to oversell.

## Step 9 — Robustness  (`step8_robustness.py`)

Report ALL of the following beside the baseline — never cherry-pick:

- **Subperiod stability:** evaluation-window α for sub-eras (e.g. 2020–2021 vs 2022–2024),
  **selection unchanged**. Legitimate because the split is not moved.
- **Rolling alpha:** group CAPM α over a rolling 24-month window across the whole evaluation
  period — the key figure for "did skill shift across regimes?" Every window shown → no cherry-
  picking.
- **Optional formal era test:** one full-sample regression interacting α and the market factor
  with an era dummy; report the interaction's significance. Answers "is α different across eras?"
  properly, in one model.
- **Weighting variant:** equal-weight vs midpoint-size.
- **Split-boundary variants:** 1–2 alternative formation/evaluation boundaries, to show the
  conclusion doesn't hinge on the exact date.
- Framing caveat for era comparisons: political regime is confounded with market conditions
  (COVID crash/recovery, 2022 bear). Attribute cautiously; most cross-era difference is beta,
  not politics. Keep it a market-regime-stability question, not a partisan claim.

---

## Data sources

**Disclosures — LOCKED: `kadoa-org/congress-trading-monitor` (GitHub, MIT).**
- Tarball: `https://codeload.github.com/kadoa-org/congress-trading-monitor/tar.gz/refs/heads/main`
  (contents API is rate-limited; don't use it).
- Use `public/data/filer/*.json` (447 files, full archive). NOT top-level `trades.json`.
- Fields used: `transaction_date`, `filing_date`, `ticker`, `asset_type`, `transaction_type`,
  `amount_range_low/high`, `owner`, `filer_id`, + filer `party`, `chamber`, `branch`.

**Committee data — `@unitedstates/congress-legislators` (GitHub, public domain).**
- legislators + committee-membership YAML, by Congress. Fetch from github.com /
  raw.githubusercontent.com (both whitelisted). Requires the filer_id↔bioguide name-match.
- Alternative: official `api.congress.gov` (committee endpoints, needs a key; live domain may
  not be network-whitelisted in the run environment — prefer the GitHub data).

**Prices/factors — yfinance (adjusted close) + Ken French data library.**

### Data caveats to log in the write-up
- Disclosure coverage effectively starts 2015; early years thin.
- Senate pre-2015 paper filings missing → partial early Senate coverage.
- Amounts are ranges → size-weighting uses midpoints only (noise); equal-weight is the default.
- Committee join is name-matched → report match rate and unmatched members.
- Sector map is author-defined → state it and its limitations explicitly.

---

## Decisions log

**Resolved (Phase 1, 2026-09-12) — carry forward unless the holding-rule recompute changes them:**
- **Split: formation 2014–2019, evaluation 2020–2024.** Coverage starts 2015; 2014–2018 is thin
  (median 9 trades/member, 27/55 clear ≥10); 2014–2019 better (median 15, 45/81 clear ≥10, 60
  eval months). From `step1_diagnostic.py`.
- **Selection: top quartile of members with ≥10 formation trades AND ≥12 invested months.**
  Under the Phase-1 long-only, implicit-exit construction this gave **7 members**
  (`house_bradleys_schneider`, `house_lamar_smith`, `house_bob_gibbs`, `house_katherinem_clark`,
  `house_dwight_evans`, `house_pete_sessions`, `senate_william_cassidy`; 4 R / 3 D).
  **NOTE:** the explicit 12-month holding rule (Step 4) recomputes invested months and may
  change this list — re-run and log the new selected set.
- **Price source: yfinance monthly adjusted close.**

**Agreed 2026-09-14 (this rewrite):**
- Holding-period rule = fixed 12-month window (primary). Long-only + long–short both reported;
  long–short handles simultaneous long/short legs. Committee track added (Track B). Era question
  handled by fixed-selection subperiod + rolling-alpha, NOT by moving the split.

**Resolved 2026-09-14 (with the user, at build time) — formerly open:**
- **Ticker→sector source: yfinance `.info`, pulled once and cached** to
  `data/clean/ticker_sectors.csv`; every later run reads only the cache. NOTE: yfinance uses
  its OWN taxonomy, not GICS names ("Financial Services" ≠ "Financials", "Healthcare" ≠
  "Health Care"). `YF_TO_GICS` in `step5_committee.py` translates it, and
  `assert_map_vocabulary()` fails loudly if any mapped sector never appears — without this,
  5 of 10 sectors silently matched nothing and the health + agriculture groups vanished.
- **Headline construction: long-only** (comparable to the Phase-1 baseline). Long–short is
  reported immediately beside it as the β-neutrality test. Both always computed.

**Additional decisions taken at build time (logged, non-results-based):**
- **Entry lag = 1 month.** A filing lands mid-month, so crediting month *t*'s full return would
  capture the pre-disclosure part of it. Legs earn from *t+1*. Strictly conservative — can only
  reduce alpha. Lag-0 reported in Step 9.
- **Point-in-time committee data reconstructed from git history.** `congress-legislators` ships
  only `committee-membership-current.yaml` (a current snapshot); there is NO by-Congress
  historical file as Step 6 item 1 assumes. Membership is read from the commit of that file as
  it stood *during* each Congress (116th/117th/118th).
- **House health jurisdiction matched at SUBCOMMITTEE level** (Ways & Means Health, E&C Health),
  per the committed map. Matching the full committees swept in every tax-writing member
  (115 → 73).

**Resolved 2026-09-14:**
- **Committee→sector map: COMMITTED** at GICS sector level with per-row oversight justifications
  (see Step 6). Map-robustness check (vary 1–2 debatable mappings) to be reported.

---

## Phase 1 results (baseline, for comparison — 2026-09-12)

Track-A long-only, equal-weight, 7 members, eval 2020–2024:
CAPM α = **−1.77%/yr**, t = **−0.52** (n.s.), β = 0.94, R² = 0.87, N = 60.
Robustness (all n.s.): 4-factor α = −1.35% (t −0.51); size-weighted α = +0.39% (t +0.07); alt
split A (14–18/19–24) α = −2.69% (t −0.55); alt split B (14–20/21–24) α = +0.94% (t +0.37).
Conclusion: no detectable risk-adjusted alpha out-of-sample, robust across variants. The
rewrite re-tests this under the explicit holding rule and long–short, and adds the committee
track — results to be regenerated.

---

## Phase 2 results (2026-09-14) — REGENERATED under the explicit 12-month holding rule

**Eligible pool 30 members → Track A selected 8** (5 D / 3 R, all House):
`house_bradleys_schneider`, `house_suzank_delbene`, `house_lamar_smith`,
`house_katherinem_clark`, `house_alans_lowenthal`, `house_bob_gibbs`, `house_pete_sessions`,
`house_dwight_evans`. **Overlap with Phase 1: 6/7** — added DelBene + Lowenthal, dropped
Cassidy (Senate). Quote THIS list downstream, not the Phase-1 one.

| Group (long-only, CAPM) | α %/yr | t | 95% CI | β | R² | N |
|---|---|---|---|---|---|---|
| **Track A selected** | **+3.49** | 1.34 | [−1.6, +8.6] | 1.03 | 0.90 | 60 |
| Track A eligible pool (30) | −1.30 | −0.60 | [−5.5, +2.9] | 1.00 | 0.95 | 60 |
| Track A rejected (control) | −3.38 | −1.26 | [−8.6, +1.9] | 0.99 | 0.92 | 60 |
| **Track B combined** | **−5.34** | −1.11 | [−14.8, +4.1] | 1.10 | 0.87 | 59 |

Long–short: Track A α = −0.40% (t −0.09), β falls 1.03 → −0.18. 4-factor: Track A +3.42%
(t 1.56). Robustness (all n.s.): size-weighted +1.79%; lag-0 +4.01%; deduped legs +3.08%;
**split A −0.77%, split B −2.13% (SIGN FLIPS)**. Track B map variants −3.86% / −3.64%.
Subperiods: 2020–21 +0.12%, 2022–24 +6.19% (t 1.91*); era-dummy alpha shift p = 0.19 (n.s.),
Track B beta shift p = 0.004.

**44 specifications, 3 significant at 5%, ~2.2 expected by chance.** Two of the three are
negative (control group). The only positive one is Track B science_commerce FF4 (+6.78%,
t 2.16) which does not survive its own CAPM spec (t 1.80).

**Conclusion: no detectable risk-adjusted alpha in either track.** Track A's positive point
estimate is insignificant, reverses sign under alternative splits, and vanishes when market
exposure is hedged out. Data caveat: 617 tickers unpriced (10.5% of in-span trades), mostly
acquisitions — delisting bias most likely *understates* returns.

Artifact write-up: https://claude.ai/code/artifact/3407511a-6ab6-45b5-9b6a-f7a893effda6

---

## Style

- Explain each step before/as you build it; don't dump the whole pipeline at once.
- Keep data-fetching, construction, and regression in separate, runnable modules.
- Comment the anti-bias points inline so they survive future edits.
- Report every variant beside the baseline; never present the flattering run alone.
