"""06_summary_figures.py - comparison figure: leakage / validation-scheme comparators
and draft-reported vs re-run key numbers."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from common import *
from plot_style import apply_style, COLORBLIND_SAFE_PALETTE as PAL
apply_style(10)
c = pd.read_csv(os.path.join(RES_DIR, "T_leakage_and_cv_comparators.csv"))
lab = {"HGB weighted, PSU-grouped CV (proposed)": "Leakage-free, PSU-grouped CV",
       "HGB weighted + child height & weight (LEAKY)": "LEAKY: + child height & weight",
       "HGB weighted, random (non-grouped) CV": "Leakage-free, random CV",
       "HGB weighted, without birth anthropometry": "Leakage-free, no birth weight/length"}
c["label"] = c.model.map(lab)
fig, ax = plt.subplots(figsize=(8, 2.9))
y = np.arange(len(c))
ax.barh(y + .18, c.oof_auroc, .34, color=PAL[0], label="Development OOF")
ax.barh(y - .18, c.test_auroc, .34, color=PAL[1], label="Locked test")
for i, r in c.iterrows():
    ax.text(max(r.oof_auroc, r.test_auroc) + .005, i, f"{r.test_auroc:.3f}", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(c.label); ax.set_xlim(.5, 1.05); ax.axvline(.5, color="#888", lw=1)
ax.set_xlabel("Weighted AUROC (HGB, survey-weighted training)"); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Target leakage inflates discrimination; validation scheme matters less here")
fig.savefig(os.path.join(FIG_DIR, "F16_leakage_comparators.png")); plt.close(fig)
print("ok")
