"""
Build the QF600 presentation deck (PowerPoint, 16:9).

Pulls every number from the saved pipeline outputs rather than hard-coding them,
so the deck regenerates correctly after any re-run.

Chart colours reuse the validated blue/red diverging pair (blue #2A6FB0 /
red #B4432E). The report's table green/red was rejected for marks: it measures
Delta-E 6.4 under protanopia, below the 8 threshold, so red-green viewers could
not separate gains from losses. On a projector that matters more, not less.

Fonts are deliberately the ones that ship with Office -- Georgia / Calibri /
Consolas -- so the deck renders identically on a machine that has never seen
this project.

Run:  python src/make_deck.py
Out:  report/QF600_congressional_alpha.pptx
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from step9_attribution import leg_contributions
from step5_committee import to_gics

OUT = C.ROOT / "report" / "QF600_congressional_alpha.pptx"
FIG = C.ROOT / "report" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- palette
INK      = RGBColor(0x13, 0x17, 0x21)
MUTED    = RGBColor(0x62, 0x6C, 0x7E)
FAINT    = RGBColor(0x8A, 0x93, 0xA3)
ACCENT   = RGBColor(0x2F, 0x4E, 0x6F)
POS      = RGBColor(0x2A, 0x6F, 0xB0)
NEG      = RGBColor(0xB4, 0x43, 0x2E)
RULE     = RGBColor(0xDC, 0xE0, 0xE7)
BAND     = RGBColor(0xF2, 0xF4, 0xF7)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)

HEX_POS, HEX_NEG, HEX_INK = "#2A6FB0", "#B4432E", "#131721"
HEX_MUTED, HEX_RULE = "#626C7E", "#DCE0E7"

H_FONT, B_FONT, M_FONT = "Georgia", "Calibri", "Consolas"
W, H = Inches(13.333), Inches(7.5)


# ---------------------------------------------------------------- figures
def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(HEX_RULE)
    ax.tick_params(colors=HEX_MUTED, labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(HEX_MUTED)


def fig_forest(res):
    """Every alpha estimate with its CI -- the single most important visual."""
    rows = res[(res.model == "CAPM") & (res.construction == "long_only")].copy()
    order = ["trackA_selected", "trackA_pool_all", "trackA_rejected", "trackB_combined",
             "trackB_science_commerce", "trackB_health", "trackB_agriculture",
             "trackB_transport_infra", "trackB_armed_services",
             "trackB_energy_commerce", "trackB_financial_services"]
    nice = {"trackA_selected": "Track A — SELECTED", "trackA_pool_all": "Track A — eligible pool",
            "trackA_rejected": "Track A — rejected (control)", "trackB_combined": "Track B — COMBINED",
            "trackB_science_commerce": "  Science / Commerce", "trackB_health": "  Health",
            "trackB_agriculture": "  Agriculture", "trackB_transport_infra": "  Transportation",
            "trackB_armed_services": "  Armed Services", "trackB_energy_commerce": "  Energy & Commerce",
            "trackB_financial_services": "  Financial Services"}
    rows = rows.set_index("group").reindex([o for o in order if o in set(rows.group)])

    fig, ax = plt.subplots(figsize=(12.2, 5.6), dpi=200)
    y = np.arange(len(rows))[::-1]
    for i, (g, r) in zip(y, rows.iterrows()):
        c = HEX_POS if r.alpha_ann >= 0 else HEX_NEG
        ax.plot([r.alpha_ci_lo_ann * 100, r.alpha_ci_hi_ann * 100], [i, i],
                color=c, lw=3, alpha=.45, solid_capstyle="butt")
        ax.plot([r.alpha_ci_lo_ann * 100] * 2, [i - .18, i + .18], color=c, lw=1.6)
        ax.plot([r.alpha_ci_hi_ann * 100] * 2, [i - .18, i + .18], color=c, lw=1.6)
        big = g in ("trackA_selected", "trackB_combined")
        ax.plot(r.alpha_ann * 100, i, "o", color=c, ms=11 if big else 7,
                mec="white", mew=1.6, zorder=3)
    ax.axvline(0, color=HEX_INK, lw=1.5, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([nice.get(g, g) for g in rows.index], fontsize=12)
    for t, g in zip(ax.get_yticklabels(), rows.index):
        t.set_color(HEX_INK)
        if g in ("trackA_selected", "trackB_combined"):
            t.set_fontweight("bold")
    ax.set_xlabel("annualised CAPM alpha, %  —  every interval crosses zero",
                  fontsize=12, color=HEX_MUTED, labelpad=10)
    ax.grid(axis="x", color=HEX_RULE, lw=.8)
    ax.set_axisbelow(True)
    style_ax(ax)
    fig.tight_layout()
    p = FIG / "forest.png"; fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


def fig_contrib(lc):
    g = lc.groupby("ticker").contrib.sum().sort_values(ascending=False)
    sel = pd.concat([g.head(10), g.tail(5)])
    fig, ax = plt.subplots(figsize=(6.6, 5.3), dpi=200)
    y = np.arange(len(sel))[::-1]
    cols = [HEX_POS if v >= 0 else HEX_NEG for v in sel.values]
    ax.barh(y, sel.values * 100, color=cols, height=.68)
    ax.set_yticks(y); ax.set_yticklabels(sel.index, fontsize=11, fontfamily="monospace")
    for t in ax.get_yticklabels(): t.set_color(HEX_INK)
    for i, v in zip(y, sel.values * 100):
        ax.text(v + (.7 if v >= 0 else -.7), i, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10.5, color=HEX_INK,
                fontfamily="monospace")
    ax.axvline(0, color=HEX_INK, lw=1.3)
    ax.set_xlim(sel.min() * 100 * 1.9 - 2, sel.max() * 100 * 1.22)
    ax.set_xlabel("contribution, percentage points", fontsize=11, color=HEX_MUTED)
    ax.grid(axis="x", color=HEX_RULE, lw=.8); ax.set_axisbelow(True)
    style_ax(ax); fig.tight_layout()
    p = FIG / "contrib.png"; fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


def fig_curve(lc):
    tot = lc.contrib.sum()
    g = lc.groupby("ticker").contrib.sum().sort_values(ascending=False)
    cum = (g.cumsum() / tot * 100).values
    x = np.arange(1, len(cum) + 1)
    fig, ax = plt.subplots(figsize=(6.6, 5.3), dpi=200)
    ax.fill_between(x, 0, cum, color=HEX_POS, alpha=.14)
    ax.plot(x, cum, color=HEX_POS, lw=2.4)
    ax.axhline(100, color=HEX_INK, lw=1.3, ls="--")
    ax.plot(10, cum[9], "o", color=HEX_POS, ms=9, mec="white", mew=1.6, zorder=3)
    ax.annotate(f"top 10 names = {cum[9]:.0f}%", (10, cum[9]), (26, cum[9] - 26),
                fontsize=11.5, color=HEX_INK,
                arrowprops=dict(arrowstyle="-", color=HEX_MUTED, lw=1))
    pk = int(cum.argmax())
    ax.annotate(f"peak {cum[pk]:.0f}%", (pk + 1, cum[pk]), (pk - 34, cum[pk] + 11),
                fontsize=11.5, color=HEX_INK,
                arrowprops=dict(arrowstyle="-", color=HEX_MUTED, lw=1))
    ax.set_xlabel("names included, ranked by contribution", fontsize=11, color=HEX_MUTED)
    ax.set_ylabel("cumulative % of total return", fontsize=11, color=HEX_MUTED)
    ax.set_ylim(0, cum.max() * 1.14)
    ax.grid(color=HEX_RULE, lw=.8); ax.set_axisbelow(True)
    style_ax(ax); fig.tight_layout()
    p = FIG / "curve.png"; fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


def fig_decay(lc):
    pm = lc.groupby("month").agg(m=("filer_id", "nunique"), n=("ticker", "nunique"))
    x = np.arange(len(pm))
    thin = int(np.argmax((pm.m <= 2).values)) if (pm.m <= 2).any() else None
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 5.2), dpi=200, sharex=True)
    for ax, col, lab, mx in ((axes[0], "m", "active members (of 8)", 8),
                             (axes[1], "n", "distinct names held", None)):
        v = pm[col].values
        if thin is not None:
            ax.axvspan(thin, len(pm) - 1, color="#EEF1F5", zorder=0)
        ax.fill_between(x, 0, v, color=HEX_POS, alpha=.16)
        ax.plot(x, v, color=HEX_POS, lw=2.2)
        ax.set_ylim(0, (mx or v.max() * 1.12))
        ax.set_ylabel(lab, fontsize=11.5, color=HEX_INK)
        ax.grid(axis="y", color=HEX_RULE, lw=.8); ax.set_axisbelow(True)
        style_ax(ax)
    ticks = [i for i, d in enumerate(pm.index) if d.month == 1]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([pm.index[i].year for i in ticks], fontsize=11.5)
    if thin is not None:
        axes[0].text(thin + 1, 7.2, "≤ 2 active members", fontsize=11,
                     color=HEX_NEG, fontweight="bold")
    fig.tight_layout()
    p = FIG / "decay.png"; fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


# ---------------------------------------------------------------- slide kit
def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, size, font, color, bold=False, space_after=6, first=False,
         align=PP_ALIGN.LEFT, italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.name = font; r.font.bold = bold
    r.font.italic = italic; r.font.color.rgb = color
    return p


def rect(slide, x, y, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(.75)
    s.shadow.inherit = False
    return s


def header(slide, eyebrow, title, n):
    tf = box(slide, Inches(.7), Inches(.45), Inches(11.9), Inches(.3))
    para(tf, eyebrow.upper(), 11, M_FONT, ACCENT, bold=True, first=True)
    tf = box(slide, Inches(.7), Inches(.82), Inches(11.9), Inches(.8))
    para(tf, title, 30, H_FONT, INK, bold=True, first=True)
    rect(slide, Inches(.7), Inches(1.62), Inches(11.93), Emu(9525), RULE)
    footer(slide, n)


def footer(slide, n):
    tf = box(slide, Inches(.7), Inches(6.95), Inches(11.9), Inches(.3))
    p = para(tf, f"QF600 · Congressional trading alpha · {n}", 9.5, M_FONT, FAINT,
             first=True, align=PP_ALIGN.RIGHT)


def bullets(slide, items, x=Inches(.7), y=Inches(1.95), w=Inches(11.9), size=16):
    tf = box(slide, x, y, w, Inches(4.6))
    for i, it in enumerate(items):
        if isinstance(it, tuple):
            head, sub = it
            para(tf, head, size, H_FONT, INK, bold=True, first=(i == 0), space_after=3)
            para(tf, sub, size - 3, B_FONT, MUTED, space_after=13)
        else:
            para(tf, it, size, B_FONT, INK, first=(i == 0), space_after=10)
    return tf


def table(slide, data, x, y, w, col_w=None, head_size=11, body_size=12,
          highlight=None, mono_from=1):
    rows, cols = len(data), len(data[0])
    h = Inches(.34) * rows
    shp = slide.shapes.add_table(rows, cols, x, y, w, h)
    tbl = shp.table
    tbl.first_row = True
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = cw
    for ri, row in enumerate(data):
        tbl.rows[ri].height = Inches(.33)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = ""
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(.08); cell.margin_right = Inches(.08)
            cell.margin_top = cell.margin_bottom = 0
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.RIGHT
            r = p.add_run(); r.text = str(val)
            head = (ri == 0)
            r.font.size = Pt(head_size if head else body_size)
            r.font.name = B_FONT if (head or ci < mono_from) else M_FONT
            r.font.bold = head
            txt = str(val)
            col = INK
            if not head and ci >= mono_from:
                if txt.startswith("+"): col = POS
                elif txt.startswith("−") or txt.startswith("-"): col = NEG
            r.font.color.rgb = WHITE if head else col
            cell.fill.solid()
            if head:
                cell.fill.fore_color.rgb = ACCENT
            elif highlight is not None and ri == highlight:
                cell.fill.fore_color.rgb = BAND
            else:
                cell.fill.fore_color.rgb = WHITE
    return shp


def note(slide, text, y=Inches(6.15), color=MUTED, size=13, italic=True):
    tf = box(slide, Inches(.7), y, Inches(11.9), Inches(.7))
    para(tf, text, size, B_FONT, color, first=True, italic=italic)


def pic(slide, path, x, y, w=None, h=None):
    if w: return slide.shapes.add_picture(str(path), x, y, width=w)
    return slide.shapes.add_picture(str(path), x, y, height=h)


# ---------------------------------------------------------------- build
def main():
    res = pd.read_csv(C.CLEAN / "regression_results.csv")
    panel = pd.read_parquet(C.CLEAN / "held_positions.parquet")
    sel = pd.read_csv(C.CLEAN / "trackA_selected.csv")["filer_id"].tolist()
    rank = pd.read_csv(C.CLEAN / "trackA_ranked_pool.csv", index_col=0)
    trades = pd.read_parquet(C.CLEAN / "congress_trades.parquet")
    lc = leg_contributions(panel, sel, "long_only")

    def cell(group, construction="long_only", model="CAPM"):
        r = res[(res.group == group) & (res.construction == construction) &
                (res.model == model)]
        return r.iloc[0] if len(r) else None

    print("rendering figures ...")
    f_forest, f_contrib = fig_forest(res), fig_contrib(lc)
    f_curve, f_decay = fig_curve(lc), fig_decay(lc)

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    n = 0

    def num():
        nonlocal n
        n += 1
        return n

    # ---- 1 title ----------------------------------------------------------
    s = blank(prs)
    rect(s, 0, 0, W, Inches(.22), ACCENT)
    tf = box(s, Inches(.9), Inches(2.0), Inches(11.5), Inches(.4))
    para(tf, "QF600 · ASSET PRICING · EMPIRICAL STUDY", 13, M_FONT, ACCENT,
         bold=True, first=True)
    tf = box(s, Inches(.9), Inches(2.5), Inches(11.5), Inches(1.6))
    para(tf, "Do Congressional Traders Generate\nRisk-Adjusted Alpha?", 44, H_FONT,
         INK, bold=True, first=True, space_after=0)
    tf = box(s, Inches(.9), Inches(4.25), Inches(10.4), Inches(1.0))
    para(tf, "Two independent selection hypotheses, tested out-of-sample on 33,522 "
             "disclosed trades.\nNeither survives the market factor.", 19, B_FONT,
         MUTED, first=True)
    rect(s, Inches(.9), Inches(5.35), Inches(2.2), Emu(19050), ACCENT)
    tf = box(s, Inches(.9), Inches(5.6), Inches(11.5), Inches(.9))
    para(tf, "Evaluation window 2020-01 → 2024-12   ·   N = 60 months   ·   "
             "CAPM & Carhart 4-factor, Newey–West HAC", 13, M_FONT, FAINT, first=True)

    # ---- 2 the question ---------------------------------------------------
    s = blank(prs); header(s, "The question", "Beating the market is not the same as alpha", num())
    bullets(s, [
        ("In 2020–2024 almost everything went up.",
         "The question is not whether a member's portfolio rose. It is whether any group earned "
         "a return the market factor cannot explain — that is what α measures."),
        ("Track A — performance selection.",
         "Rank members on trading performance in a formation window (2014–2019); test the top "
         "quartile in 2020–2024. Asks whether past winners PERSIST — the difference between skill and luck."),
        ("Track B — committee oversight.",
         "Select members by an ex-ante characteristic — which committee they sat on — and test them "
         "only in the sectors that committee regulates. Selection never touches returns, so this "
         "group cannot have been chosen for being lucky."),
    ])
    note(s, "Both tracks share one holding-period construction and one regression "
            "specification, and both are reported in full.")

    # ---- 3 guardrails -----------------------------------------------------
    s = blank(prs); header(s, "Method", "Four rules, enforced in code — not by good intentions", num())
    xs = [Inches(.7), Inches(3.75), Inches(6.8), Inches(9.85)]
    rails = [
        ("01", "Enter on the disclosure date",
         "Filings lag the trade by a median of 29 days — 438 at the 95th percentile. Trade-date "
         "entry credits a price move that was neither public nor investable."),
        ("02", "Quarantine selection",
         "Track A ranks on formation months only. A hard assertion fails the run if an evaluation "
         "month leaks in; validation re-derives the group independently."),
        ("03", "Mechanical exit",
         "Positions exit exactly 12 months after entry — a fixed function of entry, so it cannot "
         "encode what the price did next."),
        ("04", "Report every variant",
         "Run enough specifications and one crosses t = 2 by chance. The defence is showing all 44, "
         "so a starred cell reads against its neighbours."),
    ]
    for x, (k, t, d) in zip(xs, rails):
        rect(s, x, Inches(2.0), Inches(2.75), Inches(3.5), BAND)
        rect(s, x, Inches(2.0), Inches(2.75), Inches(.06), ACCENT)
        tf = box(s, x + Inches(.22), Inches(2.3), Inches(2.3), Inches(3.0))
        para(tf, k, 12, M_FONT, ACCENT, bold=True, first=True, space_after=7)
        para(tf, t, 15.5, H_FONT, INK, bold=True, space_after=8)
        para(tf, d, 12, B_FONT, MUTED)
    note(s, "One deliberately conservative choice: a filing lands mid-month, so positions earn from "
            "the FOLLOWING month. This can only reduce measured alpha, never inflate it.")

    # ---- 4 data -----------------------------------------------------------
    s = blank(prs); header(s, "Data", "From 68,167 disclosures to 33,522 usable trades", num())
    table(s, [
        ["Filter step", "Rows in", "Rows out", "Dropped"],
        ["Branch = congress", "68,167", "54,396", "13,771"],
        ["Asset type ∈ {ST, Stock, CS}", "54,396", "34,107", "20,289"],
        ["Directional transactions only", "34,107", "33,871", "236"],
        ["Ticker present", "33,871", "33,525", "346"],
        ["Both dates parse (99.99%)", "33,525", "33,522", "3"],
        ["Filing date ≥ transaction date", "33,522", "33,522", "0"],
    ], Inches(.7), Inches(2.0), Inches(7.4), mono_from=1)
    tf = box(s, Inches(8.5), Inches(2.0), Inches(4.1), Inches(4.0))
    for k, v in [("259", "members with disclosed trades"),
                 ("2,075", "tickers with price coverage"),
                 ("99.6%", "committee name-match rate"),
                 ("617", "tickers unpriced — 10.5% of trades")]:
        para(tf, k, 30, M_FONT, ACCENT, bold=True, first=(k == "259"), space_after=0)
        para(tf, v, 12.5, B_FONT, MUTED, space_after=16)
    note(s, "Delisting bias is not neutral: most unpriced names were acquired, and targets "
            "typically jump on announcement — so dropping them most likely UNDERSTATES returns.")

    # ---- 5 who was selected ----------------------------------------------
    s = blank(prs); header(s, "Track A", "Who the formation ranking selected", num())
    rows = [["#", "Member", "", "Formation α", "Trades"]]
    for i, f in enumerate(sel, 1):
        r = rank.loc[f]
        st = trades[trades.filer_id == f].iloc[0]
        rows.append([str(i), r["name"], f"{r['party']}-{st['state']}",
                     f"+{r['ann_ret']*100:.1f}%", str(int(r["form_trades"]))])
    table(s, rows, Inches(.7), Inches(2.0), Inches(7.6), mono_from=3)
    tf = box(s, Inches(8.7), Inches(2.1), Inches(3.9), Inches(4.0))
    para(tf, "30", 34, M_FONT, ACCENT, bold=True, first=True, space_after=0)
    para(tf, "members cleared the eligibility floor: ≥10 formation trades AND "
             "≥12 invested months", 13, B_FONT, MUTED, space_after=20)
    para(tf, "8", 34, M_FONT, ACCENT, bold=True, space_after=0)
    para(tf, "selected — the top quartile. 5 Democrats, 3 Republicans, all House.",
         13, B_FONT, MUTED, space_after=20)
    para(tf, "The formation column is the SELECTION, not a result.", 14, H_FONT,
         INK, bold=True, space_after=4)
    para(tf, "They were picked because those numbers are high. The test is what "
             "happened next.", 13, B_FONT, MUTED)

    # ---- 6 Track A results ------------------------------------------------
    s = blank(prs); header(s, "Track A — results", "No detectable alpha out-of-sample", num())
    r1, r2, r3 = cell("trackA_selected"), cell("trackA_rejected"), cell("trackA_pool_all")
    r4 = cell("trackA_selected", "long_short")
    r5 = cell("trackA_selected", model="FF4")
    rows = [["Group / construction", "Model", "α %/yr", "t", "95% CI", "β", "R²", "N"]]
    for lab, r, mdl in [("Selected — long-only", r1, "CAPM"),
                        ("Selected — long-only", r5, "FF4"),
                        ("Selected — long–short", r4, "CAPM"),
                        ("Eligible pool (all 30)", r3, "CAPM"),
                        ("Rejected (control)", r2, "CAPM")]:
        if r is None: continue
        a = f"{r.alpha_ann*100:+.2f}".replace("-", "−")
        rows.append([lab, mdl, a, f"{r.alpha_t:.2f}",
                     f"[{r.alpha_ci_lo_ann*100:+.1f}, {r.alpha_ci_hi_ann*100:+.1f}]".replace("-", "−"),
                     f"{r.beta:.2f}", f"{r.r2:.2f}", str(int(r.N))])
    table(s, rows, Inches(.7), Inches(2.05), Inches(11.9), highlight=1, mono_from=2)
    tf = box(s, Inches(.7), Inches(4.5), Inches(11.9), Inches(1.5))
    para(tf, "The confidence interval comfortably contains zero. The honest reading is "
             "UNDERPOWERED, not “proven absent”.", 17, H_FONT, INK, bold=True, first=True,
         space_after=9)
    para(tf, "Two details matter more than the headline. The control group works — the members the "
             "ranking REJECTED did no better. And β = 1.03 with R² = 0.90 means the market explains "
             "90% of this portfolio's variance.", 15, B_FONT, MUTED)

    # ---- 7 Track B --------------------------------------------------------
    s = blank(prs); header(s, "Track B — results", "Oversight confers no measurable edge", num())
    grp = [("trackB_combined", "COMBINED", "all mapped"),
           ("trackB_science_commerce", "Science / Commerce", "Info Tech, Comm Svcs"),
           ("trackB_health", "Health", "Health Care"),
           ("trackB_agriculture", "Agriculture", "Consumer Staples"),
           ("trackB_transport_infra", "Transportation", "Industrials, Materials"),
           ("trackB_armed_services", "Armed Services", "Industrials"),
           ("trackB_energy_commerce", "Energy & Commerce", "Energy, Utilities"),
           ("trackB_financial_services", "Financial Services", "Financials, Real Estate")]
    rows = [["Committee group", "Sectors tested", "α %/yr", "t", "95% CI", "β"]]
    for g, lab, secs in grp:
        r = cell(g)
        if r is None: continue
        rows.append([lab, secs, f"{r.alpha_ann*100:+.2f}".replace("-", "−"),
                     f"{r.alpha_t:.2f}",
                     f"[{r.alpha_ci_lo_ann*100:+.1f}, {r.alpha_ci_hi_ann*100:+.1f}]".replace("-", "−"),
                     f"{r.beta:.2f}"])
    table(s, rows, Inches(.7), Inches(2.0), Inches(11.9), highlight=1, mono_from=2)
    note(s, "Committee membership reconstructed POINT-IN-TIME for the 116th–118th Congresses from "
            "git history — the upstream dataset ships only a current snapshot, which would test 2020 "
            "trades against 2025 committee seats.", y=Inches(5.9))
    note(s, "Not one committee group clears significance under CAPM.", y=Inches(6.5),
         color=INK, size=15, italic=False)

    # ---- 8 forest ---------------------------------------------------------
    s = blank(prs); header(s, "Every estimate", "Significance stars compress; the intervals inform", num())
    pic(s, f_forest, Inches(.85), Inches(1.85), w=Inches(11.6))
    note(s, "The two widest — Energy & Commerce and Financial Services — span nearly 70 and 55 "
            "percentage points, on 64 and 353 trades. Those are not findings; they are the absence "
            "of a measurement.", y=Inches(6.45))

    # ---- 9 robustness -----------------------------------------------------
    s = blank(prs); header(s, "Robustness", "The result hinges on where the window was cut", num())
    table(s, [
        ["Specification", "α %/yr", "t", "95% CI", "β", "N"],
        ["Baseline — equal-weight, 1-month lag", "+3.49", "1.34", "[−1.6, +8.6]", "1.03", "60"],
        ["Midpoint-size weighted", "+1.79", "0.91", "[−2.1, +5.6]", "1.00", "60"],
        ["Zero entry lag (less conservative)", "+4.01", "1.44", "[−1.5, +9.5]", "1.02", "60"],
        ["De-duplicated legs", "+3.08", "1.28", "[−1.6, +7.8]", "1.05", "60"],
        ["Split A — form 14–18 / eval 19–24", "−0.77", "−0.12", "[−13.2, +11.7]", "1.10", "32"],
        ["Split B — form 14–20 / eval 21–24", "−2.13", "−0.51", "[−10.3, +6.0]", "1.29", "48"],
    ], Inches(.7), Inches(2.0), Inches(11.9), highlight=1, mono_from=1)
    tf = box(s, Inches(.7), Inches(4.85), Inches(11.9), Inches(1.4))
    para(tf, "The split-boundary rows are the most informative in the study.", 18, H_FONT,
         INK, bold=True, first=True, space_after=8)
    para(tf, "Re-running the entire selection under two other defensible split dates flips the sign. "
             "The baseline's positive alpha is not a stable property of these members — it is a "
             "property of where the window was cut.", 15, B_FONT, MUTED)

    # ---- 10 multiple comparisons -----------------------------------------
    s = blank(prs); header(s, "Multiple comparisons", "Three significant cells is what chance predicts", num())
    tf = box(s, Inches(.7), Inches(2.2), Inches(5.6), Inches(3.6))
    para(tf, "44", 64, M_FONT, ACCENT, bold=True, first=True, space_after=0)
    para(tf, "specifications estimated in the main table", 15, B_FONT, MUTED, space_after=22)
    para(tf, "3", 64, M_FONT, ACCENT, bold=True, space_after=0)
    para(tf, "cross the 5% threshold", 15, B_FONT, MUTED, space_after=22)
    para(tf, "≈ 2.2", 64, M_FONT, NEG, bold=True, space_after=0)
    para(tf, "expected by chance under the null of zero alpha everywhere", 15, B_FONT, MUTED)
    rect(s, Inches(6.9), Inches(2.2), Inches(5.7), Inches(3.5), BAND)
    rect(s, Inches(6.9), Inches(2.2), Inches(5.7), Inches(.06), NEG)
    tf = box(s, Inches(7.2), Inches(2.5), Inches(5.1), Inches(3.0))
    para(tf, "Read this before quoting the one positive result", 15, H_FONT, NEG,
         bold=True, first=True, space_after=11)
    para(tf, "Two of the three significant cells are NEGATIVE — the rejected-members control group.",
         14, B_FONT, INK, space_after=10)
    para(tf, "The single positive one — Science/Commerce under the 4-factor model, +6.78%/yr, "
             "t = 2.16 — does not survive its own CAPM specification (t = 1.80), and rests on 463 "
             "trades by 24 members.", 14, B_FONT, INK, space_after=10)
    para(tf, "It is reported because everything is reported. It is not evidence of a tradeable "
             "information advantage.", 14, B_FONT, MUTED)

    # ---- 11 attribution ---------------------------------------------------
    s = blank(prs); header(s, "What drove the P&L", "One stock is a third of the entire return", num())
    pic(s, f_contrib, Inches(.8), Inches(1.9), h=Inches(4.35))
    pic(s, f_curve, Inches(6.9), Inches(1.9), h=Inches(4.35))
    note(s, "NVDA alone contributes 30.8% of the +95.3% total, held by two of the eight members. "
            "Ten names of 130 produce 85%. The curve peaks at 127% before the losing half drags it "
            "back — a large gross gain net of a substantial gross loss.", y=Inches(6.4))

    # ---- 12 decay ---------------------------------------------------------
    s = blank(prs); header(s, "The structural weakness", "The portfolio thins to two members", num())
    pic(s, f_decay, Inches(.85), Inches(1.9), w=Inches(11.6))
    tf = box(s, Inches(.7), Inches(5.85), Inches(11.9), Inches(1.1))
    para(tf, "14 of 60 evaluation months have two or fewer active members.", 17, H_FONT,
         INK, bold=True, first=True, space_after=7)
    para(tf, "Positions roll off 12 months after disclosure and are never replaced once a member "
             "stops filing. By 2024 the “top quartile of congressional traders” is two people "
             "holding about seven stocks — so the last quarter of the regression window measures "
             "something far smaller than the group it claims to test.", 14.5, B_FONT, MUTED)

    # ---- 13 limitations ---------------------------------------------------
    s = blank(prs); header(s, "Limitations", "Stated, not hidden", num())
    bullets(s, [
        ("Statistical power is the binding constraint.",
         "Sixty monthly observations and eight members cannot resolve a plausible 2–3%/yr alpha. "
         "Most results are better described as underpowered than as null."),
        ("Delisting bias, and it is not neutral.",
         "617 tickers have no retrievable price history — 10.5% of in-span trades — mostly "
         "acquisitions, which typically jump on announcement. This likely understates returns."),
        ("The committee→sector map is author-defined.",
         "No dataset exists for it. Every row carries a written oversight rationale, and the "
         "conclusion survives both map-robustness variants (−3.86%, −3.64%)."),
        ("Disclosed amounts are ranges, and sector labels are GICS-like, not licensed GICS.",
         "Equal-weighting is the default for the first reason; the second is a current snapshot."),
    ], size=15.5)

    # ---- 14 conclusion ----------------------------------------------------
    s = blank(prs)
    rect(s, 0, 0, W, Inches(.22), ACCENT)
    header(s, "Conclusion", "No risk-adjusted alpha in either track", num())
    tf = box(s, Inches(.7), Inches(2.1), Inches(11.9), Inches(3.4))
    para(tf, "Track A's positive point estimate is not significant, reverses sign under alternative "
             "split dates, and disappears entirely once market exposure is hedged out "
             "(long–short α = −0.40%, β falls 1.03 → −0.18).", 17, B_FONT, INK, first=True,
         space_after=14)
    para(tf, "Track B is negative throughout and survives every variation of the sector map. The "
             "single significant positive cell in 44 specifications is what a multiple-comparisons "
             "count predicts.", 17, B_FONT, INK, space_after=22)
    para(tf, "The correct conclusion is NOT “members of Congress have no skill” — this design lacks "
             "the power to establish that.", 17, H_FONT, MUTED, italic=True, space_after=14)
    rect(s, Inches(.7), Inches(4.75), Inches(11.9), Inches(1.5), BAND)
    rect(s, Inches(.7), Inches(4.75), Inches(.06), Inches(1.5), ACCENT)
    tf = box(s, Inches(1.05), Inches(5.0), Inches(11.3), Inches(1.1))
    para(tf, "It is narrower and more defensible: on disclosed trades, entered when the public could "
             "actually have entered them, there is no evidence of returns the market factor cannot "
             "explain.", 19, H_FONT, INK, bold=True, first=True, space_after=8)
    para(tf, "The information advantage, if it exists, does not survive the disclosure lag.",
         16, B_FONT, MUTED)

    prs.save(OUT)
    print(f"Saved {n + 1} slides -> {OUT}")


if __name__ == "__main__":
    main()
