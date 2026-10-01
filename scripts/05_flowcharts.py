"""
05_flowcharts.py - methodological flowchart (Fig. 1), sample-flow diagram (Fig. 2)
and operational screening workflow (Fig. 15), drawn with matplotlib so they
reproduce in Colab without extra software.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from common import *

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight"})
C = dict(data="#DCE9F5", proc="#FFFFFF", key="#FDE3C8", dec="#FFF4C2", out="#D9EFD9", edge="#333333")


def box(ax, x, y, w, h, txt, fc, fs=7.4, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.015,rounding_size=0.08",
                                fc=fc, ec=C["edge"], lw=1))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def diamond(ax, x, y, w, h, txt, fs=7):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], closed=True,
                         fc=C["dec"], ec=C["edge"], lw=1))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, txt=None, off=(0.08, 0)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=C["edge"], lw=1))
    if txt:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], txt, fontsize=6.5, style="italic")


# ------------------------------------------------------------------ Fig 1 methodology
fig, ax = plt.subplots(figsize=(8.4, 9.8)); ax.axis("off"); ax.set_xlim(-0.1, 10.1); ax.set_ylim(2.9, 22)
ax.text(5, 21.6, "Research methodology flowchart", ha="center", fontsize=10, weight="bold")
box(ax, 5, 20.8, 7.2, .9, "SKI 2023 under-five microdata (86,364 children, 171 variables)\n+ codebook + WHO LMS length/height-for-age table", C["data"], bold=True)
arrow(ax, 5, 20.35, 5, 19.85)
box(ax, 5, 19.35, 7.2, 1.0, "STAGE 1 - Ingestion & audit: date-based age (days), WHO position correction (+/-0.7 cm),\nHAZ by LMS, |HAZ|>6 flags, sentinel codes (88/888/8888), duplicates, design variables", C["proc"])
arrow(ax, 5, 18.85, 5, 18.35)
diamond(ax, 5, 17.7, 4.6, 1.3, "Variable audit:\ndoes it form, co-measure,\nor follow the outcome?")
arrow(ax, 7.3, 17.7, 7.95, 17.7, "yes", off=(-0.35, 0.12))
box(ax, 8.95, 17.7, 1.95, 1.45, "Excluded\n(T1 outcome,\nT2 concurrent\nanthropometry,\nT3 proxies)", "#F6D5D5", fs=6.5)
arrow(ax, 5, 17.05, 5, 16.55, "no")
box(ax, 5, 16.1, 7.2, .8, "Locked predictor set: 121 pre-measurement features in 8 domains", C["key"], bold=True)
arrow(ax, 5, 15.7, 5, 15.2)
box(ax, 5, 14.7, 8.6, 1.0, "Design-aware split: spatial holdout (West Sumatra, Bali, Maluku) | locked test (20% of PSUs)\n| development set with 5-fold PSU-grouped cross-validation", C["proc"])
# two branches
arrow(ax, 3.2, 14.2, 2.6, 13.75); arrow(ax, 6.8, 14.2, 7.4, 13.75)
ax.text(2.6, 13.45, "AXIS B - risk (ML)", ha="center", fontsize=8, weight="bold", color="#8A3B00")
ax.text(7.4, 13.45, "AXIS A - measurement certainty", ha="center", fontsize=8, weight="bold", color="#0B4F8A")
box(ax, 2.6, 12.6, 4.4, 1.2, "STAGE 2a - Candidate models:\nLR, RF, HGB (weighted/unweighted)\n+ leaky & random-CV comparators", C["proc"])
box(ax, 7.4, 12.6, 4.4, 1.2, "STAGE 3a - Error propagation:\nSD units per 1 cm by age & sex\n(WHO LMS: dz/dh = 1/(M*S))", C["proc"])
arrow(ax, 2.6, 12.0, 2.6, 11.5); arrow(ax, 7.4, 12.0, 7.4, 11.5)
box(ax, 2.6, 10.9, 4.4, 1.2, "STAGE 2b - Platt / isotonic calibration\nfitted on out-of-fold predictions only;\nthreshold t* fixed on OOF (Youden J)", C["proc"])
box(ax, 7.4, 10.9, 4.4, 1.2, "STAGE 3b - Stress test in cm (+/-0.5, +/-1.0,\n+/-0.7 position) & Monte Carlo errors\n(sigma = 0.5 / 1.0 / 1.5 cm, 100 reps)", C["proc"])
arrow(ax, 2.6, 10.3, 2.6, 9.8); arrow(ax, 7.4, 10.3, 7.4, 9.8)
box(ax, 2.6, 9.2, 4.4, 1.2, "Evaluation: weighted AUROC, AUPRC, Brier,\nECE, calibration slope/intercept, PSU\nbootstrap CI, DCA, subgroups, importance", C["out"])
box(ax, 7.4, 9.2, 4.4, 1.2, "Re-measurement policy comparison:\nP0 none | P1 draft +/-0.10 SD | P2 budget-\nmatched fixed band | P3 per-child band", C["out"])
arrow(ax, 2.6, 8.6, 4.2, 7.75); arrow(ax, 7.4, 8.6, 5.8, 7.75)
box(ax, 5, 7.2, 9.6, 1.1, "STAGE 4 - Decision framework: Axis A (3 levels, alpha) x Axis B (2 levels, t*) = 6 cells\n-> merge the two 'confirmed stunted' cells -> 5 mutually exclusive, exhaustive categories", C["key"], bold=True)
arrow(ax, 5, 6.65, 5, 6.15)
box(ax, 5, 5.6, 7.2, 1.0, "Category validation on locked test & spatial holdout: observed stunting, mean HAZ,\nlabel-error probability, re-measurement stability, sensitivity to sigma & alpha,\ncomparison with the draft matrix", C["proc"])
arrow(ax, 5, 5.1, 5, 4.6)
box(ax, 5, 4.05, 9.6, 1.0, "Output: 5 screening categories with graded actions\n(refer & manage | priority re-measure | routine re-measure | intensified prevention | routine monitoring)", C["out"], bold=True)
ax.text(5, 3.2, "Blue = data; white = processing; yellow = decision; orange = locked/derived artefact; green = evaluation/output",
        ha="center", fontsize=6.5, color="#555")
fig.savefig(os.path.join(FIG_DIR, "F01_methodology_flowchart.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig 2 sample flow
fl = pd.read_csv(os.path.join(RES_DIR, "T_flow_exclusion.csv"))
sp = pd.read_csv(os.path.join(RES_DIR, "T_split_summary.csv")).set_index("split")
n = dict(zip(fl.step, fl.n))
fig, ax = plt.subplots(figsize=(8, 4.9)); ax.axis("off"); ax.set_xlim(-0.1, 10.1); ax.set_ylim(3.9, 11.7)
box(ax, 4, 11.2, 5, .8, f"SKI 2023 under-five records\nn = {n['Raw SKI 2023 under-five records']:,}", C["data"], fs=8, bold=True)
arrow(ax, 4, 10.8, 4, 9.5)
box(ax, 8, 10.15, 3.4, .9, f"Excluded: length/height not measured\nn = {n['Raw SKI 2023 under-five records'] - n['Length/height measured']:,}", "#F6D5D5", fs=7)
arrow(ax, 4, 10.15, 6.3, 10.15)
box(ax, 4, 9.1, 5, .8, f"Length/height measured, age 0-59 months\nn = {n['Length/height measured']:,}", C["proc"], fs=8)
arrow(ax, 4, 8.7, 4, 7.4)
box(ax, 8, 8.05, 3.4, .9, f"Excluded: implausible HAZ (|HAZ| > 6)\nn = {n['Excluded biologically implausible HAZ (|HAZ|>6, WHO flag)']:,}", "#F6D5D5", fs=7)
arrow(ax, 4, 8.05, 6.3, 8.05)
box(ax, 4, 7.0, 6.2, .8, f"Final analytic sample (complete design variables)\nn = {n['Final analytic sample (unique child ID)']:,}", C["key"], fs=8, bold=True)
for x, key, lbl in [(1.6, "dev", "Development set"), (4.9, "test", "Locked test set"), (8.2, "holdout", "Spatial holdout")]:
    arrow(ax, 4, 6.6, x, 5.5)
    r = sp.loc[key]
    extra = "\n(W. Sumatra, Bali, Maluku)" if key == "holdout" else ("\n(20% of PSUs)" if key == "test" else "\n(5-fold PSU-grouped CV)")
    box(ax, x, 4.8, 3.0, 1.3, f"{lbl}{extra}\nn = {int(r.n):,}; PSU = {int(r.n_psu):,}\nweighted stunting {100 * r.prev_w:.1f}%", C["out"], fs=7)
fig.savefig(os.path.join(FIG_DIR, "F02_sample_flow.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig 15 operational workflow
fig, ax = plt.subplots(figsize=(9, 4.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
box(ax, 1.2, 4.6, 2.1, 1.1, "Household & maternal\ninformation\n(no anthropometry)", C["data"], fs=7)
arrow(ax, 2.25, 4.6, 2.95, 4.6)
box(ax, 4.05, 4.6, 2.1, 1.1, "Calibrated ML risk\nP_cal -> high / low\n(Axis B)", C["key"], fs=7)
arrow(ax, 5.1, 4.6, 5.8, 4.6)
ax.text(5.45, 5.35, "prioritise\nwho is measured first", ha="center", fontsize=6.3, style="italic")
box(ax, 6.9, 4.6, 2.1, 1.1, "Standardised length/\nheight measurement\n+ age in days", C["proc"], fs=7)
arrow(ax, 7.95, 4.6, 8.55, 4.6)
box(ax, 9.9, 4.6, 2.6, 1.1, "p = P(true HAZ < -2 |\nobserved, sigma_cm)\n(Axis A)", C["key"], fs=7)
arrow(ax, 9.9, 4.05, 9.9, 3.45)
diamond(ax, 9.9, 2.85, 2.6, 1.1, "p >= 1-alpha ?\nalpha < p < 1-alpha ?\np <= alpha ?", fs=6.3)
cols = {1: "#B2182B", 2: "#EF8A62", 3: "#F4C28F", 4: "#67A9CF", 5: "#2166AC"}
lab = {1: "C1 Refer &\nmanage", 2: "C2 Priority\nre-measure", 3: "C3 Routine\nre-measure", 4: "C4 Intensified\nprevention", 5: "C5 Routine\nmonitoring"}
for i, k in enumerate(range(1, 6)):
    x = 1.1 + i * 1.75
    ax.add_patch(FancyBboxPatch((x - .75, .35), 1.5, .95, boxstyle="round,pad=0.02", fc=cols[k], ec="white"))
    ax.text(x, .82, lab[k], ha="center", va="center", fontsize=6.8, color="white" if k in (1, 5) else "black", weight="bold")
    ax.annotate("", xy=(x, 1.32), xytext=(8.6, 2.85), arrowprops=dict(arrowstyle="-|>", color="#777", lw=.7))
ax.annotate("", xy=(6.9, 4.05), xytext=(3.5, 1.32), arrowprops=dict(arrowstyle="-|>", color="#B2182B", lw=1, ls="--", connectionstyle="arc3,rad=-0.3"))
ax.annotate("", xy=(6.9, 4.05), xytext=(5.0, 1.32), arrowprops=dict(arrowstyle="-|>", color="#B2182B", lw=1, ls="--", connectionstyle="arc3,rad=-0.2"))
ax.text(4.9, 2.9, "second measurement;\naverage and re-classify", fontsize=6.5, color="#B2182B", style="italic")
fig.savefig(os.path.join(FIG_DIR, "F15_operational_workflow.png")); plt.close(fig)
print("flowcharts done")
