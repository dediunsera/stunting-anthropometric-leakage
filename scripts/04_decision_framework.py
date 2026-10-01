"""
04_decision_framework.py
Stage 4 - Derivation and evaluation of the five screening categories.

Axis A (measurement certainty, per child):
    p_i = P(true HAZ < -2 | observed HAZ, sigma_cm)   = Phi((-2 - HAZ_obs) / (sigma_cm * z_per_cm_i))
    A1 confirmed stunted      p_i >= 1 - alpha
    A2 uncertain              alpha < p_i < 1 - alpha
    A3 confirmed not stunted  p_i <= alpha
Axis B (calibrated, leakage-free ML risk; threshold locked on development OOF):
    B1 high risk  P_cal >= t*      B0 low risk  P_cal < t*
Cells: 3 x 2 = 6. For A1 the action does not depend on B (the child is already stunted
and must be managed), so A1xB1 and A1xB0 merge -> 5 mutually exclusive, exhaustive categories.

Also applies the DRAFT matrix (fixed +/-0.10 SD, P_cal 0.30/0.60) to the same children
to show its coverage gaps.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm
from common import *
from plot_style import apply_style, COLORBLIND_SAFE_PALETTE as PAL

apply_style(10)
rng = np.random.default_rng(SEED)
P = pd.read_pickle(os.path.join(DATA_DIR, "predictions.pkl"))
CFG = json.load(open(os.path.join(RES_DIR, "config_locked.json")))
T_RISK = CFG["risk_threshold"]

CAT = {1: "C1 Confirmed stunting", 2: "C2 Uncertain, high risk", 3: "C3 Uncertain, low risk",
       4: "C4 Not stunted, high risk", 5: "C5 Not stunted, low risk"}
ACTION = {1: "Refer to health centre for clinical assessment & stunting management; enrol in PMT/nutrition programme",
          2: "Priority re-measurement within 14 days by a second trained measurer (standard board); provisional counselling",
          3: "Re-measure at the next monthly posyandu session; routine counselling",
          4: "Not stunted: intensified prevention - monthly growth monitoring, counselling on modifiable risks (feeding, WASH, ANC)",
          5: "Routine growth monitoring and health education"}
COLORS = {1: "#B2182B", 2: "#EF8A62", 3: "#F4C28F", 4: "#67A9CF", 5: "#2166AC"}


def categorise(o, sigma=SIGMA_CM_MAIN, alpha=ALPHA_MAIN, t=T_RISK, haz=None):
    haz = o.haz.values if haz is None else haz
    p = norm.cdf((STUNT_CUT - haz) / (sigma * o.z_per_cm.values))
    hi = o.p_cal.values >= t
    c = np.where(p >= 1 - alpha, 1, np.where(p <= alpha, np.where(hi, 4, 5), np.where(hi, 2, 3)))
    return c, p


def draft_categorise(o):
    """Draft manuscript matrix. Methods version (HAZ-only outside zone) and
    Results-table version (also requires P_cal ranges outside zone)."""
    z, p = o.haz.values, o.p_cal.values
    inb = (z >= -2.10) & (z <= -1.90)
    meth = np.select([z < -2.10, z > -1.90, inb & (p >= .6), inb & (p >= .3), inb], [1, 5, 2, 3, 4], 0)
    res = np.select([(z < -2.10) & (p >= .6), inb & (p >= .6), inb & (p >= .3) & (p < .6), inb & (p < .3),
                     (z > -1.90) & (p < .3)], [1, 2, 3, 4, 5], 0)   # 0 = no category assigned
    return meth, res


rows, prof_rows, stab_rows = [], [], []
for part in ("test", "hold"):
    o = P[part].copy(); w = o[W].values
    c, p = categorise(o); o["cat"] = c; o["p_stunt_true"] = p
    o["draft_meth"], o["draft_res"] = draft_categorise(o)
    P[part] = o
    # stability: simulate one independent re-measurement (sigma = 1 cm) and re-categorise, 50 reps
    same = np.zeros(len(o))
    for r in range(50):
        hz2 = o.haz.values + rng.normal(0, SIGMA_CM_MAIN, len(o)) * o.z_per_cm.values
        c2, _ = categorise(o, haz=hz2)
        same += (c2 == c)
    o["stab"] = same / 50
    for k in range(1, 6):
        m = c == k
        if m.sum() == 0:
            continue
        ww = w[m]
        prof_rows.append(dict(partition="Locked test" if part == "test" else "Spatial holdout", category=CAT[k],
                              n=int(m.sum()), weighted_pct=100 * ww.sum() / w.sum(),
                              observed_stunting_pct=100 * np.average(o.stunted.values[m], weights=ww),
                              mean_haz=np.average(o.haz.values[m], weights=ww),
                              pct_haz_below_minus1=100 * np.average(o.haz.values[m] < -1, weights=ww),
                              mean_p_cal=np.average(o.p_cal.values[m], weights=ww),
                              mean_label_error_prob=100 * np.average(np.minimum(o.p_stunt_true.values[m], 1 - o.p_stunt_true.values[m]), weights=ww),
                              category_stability_pct=100 * np.average(o.stab.values[m], weights=ww),
                              action=ACTION[k]))
PR = pd.DataFrame(prof_rows).round(3)
PR.to_csv(os.path.join(RES_DIR, "T_category_profile.csv"), index=False)
print(PR.drop(columns="action").to_string())

# draft coverage
dr = []
for part in ("test", "hold"):
    o = P[part]; w = o[W].values
    for ver in ("draft_meth", "draft_res"):
        v = o[ver].values
        dr.append(dict(partition=part, version="Draft (Methods text)" if ver == "draft_meth" else "Draft (Results Table)",
                       unassigned_pct=100 * w[v == 0].sum() / w.sum(),
                       **{f"cat{k}_pct": 100 * w[v == k].sum() / w.sum() for k in range(1, 6)}))
    c = o.cat.values
    dr.append(dict(partition=part, version="Proposed", unassigned_pct=0.0,
                   **{f"cat{k}_pct": 100 * w[c == k].sum() / w.sum() for k in range(1, 6)}))
DR = pd.DataFrame(dr).round(3); DR.to_csv(os.path.join(RES_DIR, "T_draft_vs_proposed_coverage.csv"), index=False)
print(DR.to_string())

# sensitivity of category distribution to sigma and alpha (locked test)
sens = []
o = P["test"]; w = o[W].values
for sig in SIGMA_CM_SCENARIOS:
    for a in ALPHA_GRID:
        c, _ = categorise(o, sigma=sig, alpha=a)
        sens.append(dict(sigma_cm=sig, alpha=a, **{CAT[k]: 100 * w[c == k].sum() / w.sum() for k in range(1, 6)},
                         remeasure_pct=100 * w[(c == 2) | (c == 3)].sum() / w.sum()))
SE = pd.DataFrame(sens).round(2); SE.to_csv(os.path.join(RES_DIR, "T_category_sensitivity.csv"), index=False)
print(SE.to_string())

# cross-tab draft (methods) vs proposed
ct = pd.crosstab(P["test"].draft_meth.map({0: "none", 1: "D1 Stable Stunted", 2: "D2 Likely Stunted", 3: "D3 Uncertain",
                                            4: "D4 Likely Non-Stunted", 5: "D5 Stable Non-Stunted"}),
                 P["test"].cat.map(CAT), values=P["test"][W], aggfunc="sum", normalize="all") * 100
ct.round(2).to_csv(os.path.join(RES_DIR, "T_crosstab_draft_vs_proposed.csv")); print(ct.round(2))

# ================================================================== FIGURES
o = P["test"]; w = o[W].values
# F12 decision space: HAZ vs P_cal coloured by category
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
sub = o.sample(min(6000, len(o)), random_state=SEED)
for k in range(5, 0, -1):
    s = sub[sub.cat == k]
    ax[0].scatter(s.haz, s.p_cal, s=4, alpha=.55, color=COLORS[k], label=CAT[k], rasterized=True)
ax[0].axhline(T_RISK, color="k", ls="--", lw=1); ax[0].axvline(-2, color="k", lw=.8)
ax[0].text(1.9, T_RISK + .01, f"t* = {T_RISK:.3f}", ha="right", fontsize=7)
ax[0].set_xlim(-5, 2); ax[0].set_xlabel("Observed HAZ"); ax[0].set_ylabel("Calibrated risk P_cal (ML, no anthropometry)")
ax[0].set_title("(a) Decision space, locked test"); ax[0].legend(fontsize=7, markerscale=3, loc="upper right")
pr = PR[PR.partition == "Locked test"]
yy = np.arange(5)[::-1]
ax[1].barh(yy, pr.weighted_pct, color=[COLORS[k] for k in range(1, 6)], height=.6)
for y_, (_, r) in zip(yy, pr.iterrows()):
    ax[1].text(r.weighted_pct + .5, y_, f"{r.weighted_pct:.1f}%  (stunted {r.observed_stunting_pct:.0f}%, stable {r.category_stability_pct:.0f}%)", va="center", fontsize=7)
ax[1].set_yticks(yy); ax[1].set_yticklabels(pr.category, fontsize=8); ax[1].set_xlim(0, 80)
ax[1].set_xlabel("Weighted share of children (%)"); ax[1].set_title("(b) Category distribution")
fig.savefig(os.path.join(FIG_DIR, "F12_decision_space.png")); plt.close(fig)

# F11 derivation matrix 3x2 -> 5
fig, ax = plt.subplots(figsize=(8.6, 4.2)); ax.axis("off"); ax.set_xlim(-0.2, 10); ax.set_ylim(0, 6.2)
ax.text(5.9, 5.95, "Axis B: calibrated ML risk (pre-measurement, no anthropometry)", ha="center", fontsize=9, weight="bold")
ax.text(4.35, 5.45, f"High risk  (P_cal >= {T_RISK:.2f})", ha="center", fontsize=8.5)
ax.text(7.55, 5.45, f"Low risk  (P_cal < {T_RISK:.2f})", ha="center", fontsize=8.5)
ax.text(0.15, 2.6, "Axis A: measurement certainty\nP(true HAZ < -2 | observed, sigma)", fontsize=8.5, weight="bold", va="center", ha="center", rotation=90)
rowsA = [("A1 Confirmed stunted\np >= 1 - alpha", 4.1), ("A2 Uncertain\nalpha < p < 1 - alpha", 2.55), ("A3 Confirmed not stunted\np <= alpha", 1.0)]
for lbl, y_ in rowsA:
    ax.text(2.55, y_ + .55, lbl, ha="right", va="center", fontsize=8)
pct = pr.set_index("category").weighted_pct
def cell(x, y_, wdt, k, txt):
    ax.add_patch(FancyBboxPatch((x, y_), wdt, 1.1, boxstyle="round,pad=0.02", fc=COLORS[k], ec="white", lw=2, alpha=.9))
    ax.text(x + wdt / 2, y_ + .55, f"{CAT[k]}\n{txt}\n({pct[CAT[k]]:.1f}% of children)", ha="center", va="center",
            fontsize=7.6, color="white" if k in (1, 5) else "black", weight="bold")
cell(2.75, 4.1, 6.4, 1, "Action: refer & manage (risk tier does not change action)")
cell(2.75, 2.55, 3.15, 2, "Priority re-measurement <= 14 d")
cell(6.0, 2.55, 3.15, 3, "Re-measure at next posyandu")
cell(2.75, 1.0, 3.15, 4, "Intensified prevention")
cell(6.0, 1.0, 3.15, 5, "Routine monitoring")
ax.text(5.9, 0.35, f"3 x 2 = 6 cells; the two A1 cells share one action -> 5 mutually exclusive, exhaustive categories "
        f"(sigma = {SIGMA_CM_MAIN} cm, alpha = {ALPHA_MAIN})", ha="center", fontsize=7.5, style="italic")
fig.savefig(os.path.join(FIG_DIR, "F11_category_derivation.png")); plt.close(fig)

# F13 draft vs proposed coverage
fig, ax = plt.subplots(figsize=(8.5, 2.9))
d = DR[DR.partition == "test"].set_index("version")
left = np.zeros(len(d))
labels = ["unassigned"] + [f"cat{k}_pct" for k in range(1, 6)]
cols = ["#BBBBBB"] + [COLORS[k] for k in range(1, 6)]
names = ["No category (gap)", "Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"]
for lab, col, nm in zip(["unassigned_pct"] + labels[1:], cols, names):
    ax.barh(d.index, d[lab], left=left, color=col, label=nm, height=.55, edgecolor="white")
    left += d[lab].values
for i, v in enumerate(d.unassigned_pct):
    if v > 1:
        ax.text(v / 2, i, f"{v:.1f}%", ha="center", va="center", fontsize=8, weight="bold")
ax.set_xlabel("Weighted share of children, locked test (%)"); ax.set_xlim(0, 100)
ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(.5, 1.25))
fig.savefig(os.path.join(FIG_DIR, "F13_draft_vs_proposed.png")); plt.close(fig)

# F14 sensitivity heat table: remeasure share by sigma x alpha
fig, ax = plt.subplots(figsize=(4.6, 2.8))
hm = SE.pivot(index="sigma_cm", columns="alpha", values="remeasure_pct")
im = ax.imshow(hm.values, cmap="Blues", aspect="auto")
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        ax.text(j, i, f"{hm.values[i, j]:.1f}%", ha="center", va="center", fontsize=9,
                color="white" if hm.values[i, j] > hm.values.max() * .6 else "black")
ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns); ax.set_yticks(range(hm.shape[0]))
ax.set_yticklabels([f"{s} cm" for s in hm.index]); ax.set_xlabel("alpha (tolerated label-error probability)")
ax.set_ylabel("Assumed error SD"); ax.set_title("Children routed to re-measurement (C2+C3)"); ax.grid(False)
fig.savefig(os.path.join(FIG_DIR, "F14_sensitivity_heatmap.png")); plt.close(fig)
print("done")
