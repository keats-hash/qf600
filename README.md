# QF600 — Do Congressional Traders Generate Risk-Adjusted Alpha?

Backtests two independent selection hypotheses on U.S. Congress members' disclosed
personal stock trades, and tests each for **risk-adjusted alpha** by separating skill (α)
from market exposure (β) via CAPM and 4-factor regressions.

- **Track A — performance selection.** Rank members by trading performance in a *formation*
  window; test the selected group in a later, non-overlapping *evaluation* window. Asks
  whether past winners **persist** (skill) or were **lucky** (noise).
- **Track B — committee selection.** Select members by an *ex-ante* characteristic —
  committee membership — restrict to the sectors that committee oversees, and test for alpha
  there. Asks whether information advantage concentrates where members have oversight.

Both tracks share one holding-period construction and one regression specification, and
both are reported in full beside each other.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy statsmodels yfinance pyyaml pyarrow requests matplotlib
.venv/bin/python run_all.py            # full pipeline, ~1h on a cold cache
.venv/bin/python run_all.py --from 4   # resume (steps 1–3 are cached)
```

Every step is also independently runnable: `.venv/bin/python src/step3_holding.py`.
All human-readable output lands in `data/reports/`.

---

## Pipeline

| Step | Module | Purpose |
|---|---|---|
| 1 | `step1_load.py` | Parse + filter + clean disclosures → tidy trade table |
| 2 | `step1_diagnostic.py` | Trade-count diagnostic → fixes split + eligibility floor (**dates/counts only**) |
| 3 | `step2_data.py` | Monthly prices (yfinance) + Ken French factors |
| 4 | `step3_holding.py` | **Foundational.** Trades → monthly held-position panel |
| 5 | `step4_formation.py` | Track A selection: formation ranking → top quartile |
| 6 | `step5_committee.py` | Track B selection: committee data → sector-restricted groups |
| 7 | `step6_portfolio.py` | Evaluation-window return series, long-only and long–short |
| 8 | `step7_regression.py` | CAPM + 4-factor, Newey–West HAC; α, t, β, R², CI |
| 9 | `step8_robustness.py` | Subperiods, rolling α, weighting, splits, map variants |

Committed methodological constants live in [`src/config.py`](src/config.py); the
committee→sector map lives in [`config/sector_map.yaml`](config/sector_map.yaml).

---

## The design, and why it is built this way

The whole point is that **selection is quarantined from evaluation**. Four rules are
enforced in code rather than by good intentions:

**1. Entry is on the DISCLOSURE date, never the trade date.**
Filings lag the trade by a median of 29 days and 438 days at the 95th percentile in this
dataset. Entering on the trade date credits the portfolio with a price move that was not
public and not investable. This is the single largest look-ahead trap in congressional-trading
research and the main reason headline "Congress beats the market" claims don't replicate.

**2. Selection never touches evaluation-window returns.**
Track A ranks on formation months only, with a hard assertion in
[`step4_formation.py`](src/step4_formation.py) that raises if an evaluation month leaks in.
Track B selects on committee membership only — no return enters the choice at all.

**3. The exit carries no information.**
Exit is a fixed function of entry (entry + 12 months), so it cannot encode what the price
did afterwards. Hold-until-disclosed-closing-trade was rejected because closing trades are
sparsely and very late disclosed, which would make the exit date itself an information-bearing
selected quantity.

**4. Every variant is reported beside the baseline.**
Run enough specifications and one crosses t = 2 by chance. The defence isn't fewer
specifications — it's showing all of them, so a reader can see whether a starred cell is the
exception among many nulls or a consistent pattern.

---

## Decisions worth knowing about

**Entry lag = 1 month.** A filing lands mid-month, so crediting that month's *full* return
would capture the part that preceded disclosure — a partial look-ahead. Legs therefore earn
from month *t+1*. This is the strictly conservative direction: it can only reduce measured
alpha. The lag-0 variant is reported in Step 9, because if alpha appears only at lag 0, that
alpha is an artefact.

**Point-in-time committee data is reconstructed from git history.**
`@unitedstates/congress-legislators` ships only `committee-membership-current.yaml`, a single
*current* snapshot — there is no by-Congress historical file. Using the current snapshot would
test a member's 2020 trades against a committee they joined in 2025, and would silently drop
every member who has since left Congress — a survivorship filter on the selection variable
itself. Instead, `step5_committee.py` pins the commit of that file as it stood *during* each
Congress (116th/117th/118th) and reads membership from that commit.

**Equal-weight is the default.** Disclosed amounts are ranges, so midpoint sizes are badly
measured. Midpoint-size weighting is a reported robustness variant, never the headline.

**Delisted tickers are logged, not silently dropped.** Silently dropping them is survivorship
bias — a name that went to zero would vanish instead of registering its loss. `step2_data.py`
recovers ticker renames and share-class spellings (BRK.B → BRK-B, ANTM → ELV) via an explicit
logged map, and writes genuinely unavailable tickers to
`data/reports/step3_unavailable_tickers.txt`. Companies that were *acquired* have no continuing
series and are reported as delisting bias.

---

## Known limitations (stated, not hidden)

- Disclosure coverage effectively starts 2015; early years are thin, and Senate pre-2015 paper
  filings are missing.
- **Delisting bias is real and vintage-dependent.** 617 traded tickers have no retrievable price
  series (10.5% of in-span trades). Most were acquired or taken private *during or after* the
  study window — including names that were liquid and fully tradeable throughout 2020–2024 but
  whose history the price source no longer serves because they were acquired in 2025–26
  (EA, DFS, DISH, CYBR among them). The direction of this bias is not neutral: acquisition
  targets typically jump on announcement, so dropping them most likely **understates** the
  portfolios' returns. Every dropped ticker is listed in
  `data/reports/step3_unavailable_tickers.txt`, and per-member drop counts are in
  `data/reports/step3_price_coverage.txt`.
- Amounts are ranges, so size-weighting uses noisy midpoints.
- The committee join is name-matched (**99.6%**, 258/259 members); unmatched members are listed
  in the Step 6 report.
- The committee→sector map is **author-defined** — there is no dataset for it. Every row carries
  a one-line oversight rationale, and Step 9 re-runs the analysis under varied mappings.
- Broad committees (especially Energy & Commerce) span multiple sectors, so the
  committee→sector link is coarser for them.
- yfinance sectors are a GICS-*like* approximation, not licensed GICS, and are a current
  snapshot — a company reclassified since 2020 carries today's label.
- Samples in Track B are small by construction, so the regressions report **confidence
  intervals**, not just t-statistics. A wide CI around zero is honest *underpowered*, which is
  a different claim from "no alpha".

---

## Data sources

| What | Source | Licence |
|---|---|---|
| Disclosures | [`kadoa-org/congress-trading-monitor`](https://github.com/kadoa-org/congress-trading-monitor) — per-filer archive (447 files) | MIT |
| Committees | [`unitedstates/congress-legislators`](https://github.com/unitedstates/congress-legislators) | Public domain |
| Prices | yfinance, monthly adjusted close (`auto_adjust=True`) | — |
| Factors | Ken French Data Library (Mkt-RF, SMB, HML, MOM, RF) | — |
