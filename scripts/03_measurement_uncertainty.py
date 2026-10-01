"""
03_measurement_uncertainty.py
Stage 3 - Anthropometric measurement uncertainty (Axis A of the framework).

 A. How large is 1 cm of length/height error in HAZ units, by age and sex? (WHO LMS)
 B. Deterministic perturbation stress test in CENTIMETRES (+/-0.5, +/-1.0 cm and
    +/-0.7 cm recumbent/standing position error) vs the draft's HAZ-shift test.
 C. Monte Carlo simulation of random measurement error (sigma = 0.5/1.0/1.5 cm):
    how many children would receive the wrong stunting label?
 D. Re-measurement policy comparison: which children should be re-measured?
    P0 none | P1 fixed band HAZ -2 +/- 0.10 SD (draft) | P2 fixed band tuned to the
    same budget as P3 | P3 proposed per-child probability band
    alpha < P(true HAZ < -2 | observed) < 1 - alpha.

Simulation truth = recorded (position-corrected) height; simulated observation =
truth + N(0, sigma^2). Survey weights are used for all population estimates.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from common import *
from plot_style import apply_style, COLORBLIND_SAFE_PALETTE as PAL

apply_style(10)
rng = np.random.default_rng(SEED)
A = pd.read_pickle(ANALYTIC); d = A["df"].reset_index(drop=True)
h, M, S, w = d.height_adj.values, d.lms_M.values, d.lms_S.values, d[W].values
age = d.age_months.values; zpc = d.z_per_cm.values; haz = d.haz.values
y_true = (haz < STUNT_CUT).astype(int)
AGEG = np.select([age < 6, age < 24], ["0-5 mo", "6-23 mo"], "24-59 mo")
GROUPS = ["0-5 mo", "6-23 mo", "24-59 mo"]
wp = lambda m: 100 * w[m].sum() / w.sum()


def zfromh(hh):
    return (hh / M - 1.0) / S


# ------------------------------------------------------------------ A. z per cm
lms = load_lms()
lms["m_months"] = lms.age / 30.4375
lms["sd_per_cm"] = 1 / (lms.m * lms.s)
lms["cm_per_0.10SD"] = 0.10 * lms.m * lms.s
tabA = []
for mo in [0, 3, 6, 12, 18, 24, 36, 48, 59]:
    r = lms[lms.age == int(round(mo * 30.4375))]
    for sx, lab in [(1, "Boys"), (2, "Girls")]:
        q = r[r.sex == sx].iloc[0]
        tabA.append(dict(age_months=mo, sex=lab, median_cm=q.m, sd_cm=q.m * q.s,
                         SD_per_1cm=q.sd_per_cm, cm_equiv_0p10SD=q["cm_per_0.10SD"]))
TA = pd.DataFrame(tabA).round(3); TA.to_csv(os.path.join(RES_DIR, "T_sd_per_cm_by_age.csv"), index=False)
print(TA)

# ------------------------------------------------------------------ B. deterministic stress test
rowsB = []
scen = [("Height -1.0 cm", -1.0, None), ("Height -0.5 cm", -0.5, None), ("Height +0.5 cm", 0.5, None),
        ("Height +1.0 cm", 1.0, None), ("Position error -0.7 cm", -0.7, None), ("Position error +0.7 cm", 0.7, None),
        ("Draft: HAZ -0.10 SD", None, -0.10), ("Draft: HAZ -0.05 SD", None, -0.05),
        ("Draft: HAZ +0.05 SD", None, 0.05), ("Draft: HAZ +0.10 SD", None, 0.10)]
for lbl, dcm, dz in scen:
    z2 = zfromh(h + dcm) if dcm is not None else haz + dz
    sw = ((z2 < STUNT_CUT).astype(int) != y_true)
    row = dict(scenario=lbl, overall_switch_pct=wp(sw),
               prevalence_after_pct=100 * np.average(z2 < STUNT_CUT, weights=w),
               n_switched=int(sw.sum()))
    for g in GROUPS:
        m = AGEG == g
        row[f"switch_{g}_pct"] = 100 * w[m & sw].sum() / w[m].sum()
    rowsB.append(row)
TB = pd.DataFrame(rowsB).round(3); TB.to_csv(os.path.join(RES_DIR, "T_perturbation_cm_vs_haz.csv"), index=False)
print(TB)
prev0 = 100 * np.average(y_true, weights=w)

# ------------------------------------------------------------------ C/D. Monte Carlo
R = 100
SIGMAS = SIGMA_CM_SCENARIOS
KGRID = np.round(np.arange(0.02, 0.62, 0.02), 2)          # fixed band half-widths (SD)
AGRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]  # proposed alpha grid


def evaluate(flag, obs_lab, h_obs, sig):
    """Remeasure flagged children once (independent error) and average the two readings."""
    err = obs_lab != y_true
    h2 = h + rng.normal(0, sig, len(h))
    lab2 = (zfromh((h_obs + h2) / 2) < STUNT_CUT).astype(int)
    final = np.where(flag, lab2, obs_lab)
    return dict(burden=wp(flag), err_before=wp(err), capture=100 * w[flag & err].sum() / max(w[err].sum(), 1e-9),
                err_after=wp(final != y_true), yield_pct=100 * w[flag & err].sum() / max(w[flag].sum(), 1e-9))


mc_rows, curve_rows = [], []
for sig in SIGMAS:
    for r in range(R):
        h_obs = h + rng.normal(0, sig, len(h))
        z_obs = zfromh(h_obs); lab = (z_obs < STUNT_CUT).astype(int); err = lab != y_true
        # C. misclassification
        row = dict(sigma_cm=sig, rep=r, misclass_pct=wp(err),
                   false_stunted_pct=wp(err & (lab == 1)), missed_stunted_pct=wp(err & (lab == 0)),
                   prev_obs_pct=100 * np.average(lab, weights=w))
        for g in GROUPS:
            m = AGEG == g; row[f"misclass_{g}_pct"] = 100 * w[m & err].sum() / w[m].sum()
        mc_rows.append(row)
        if r >= 30:
            continue  # policy curves use first 30 replicates (stable to 3 d.p.)
        pst = norm.cdf((STUNT_CUT - z_obs) / (sig * zpc))
        for k in KGRID:
            f = np.abs(z_obs - STUNT_CUT) <= k
            curve_rows.append(dict(sigma_cm=sig, rep=r, policy="Fixed HAZ band", param=k, **evaluate(f, lab, h_obs, sig)))
        for a in AGRID:
            f = (pst > a) & (pst < 1 - a)
            curve_rows.append(dict(sigma_cm=sig, rep=r, policy="Proposed probability band", param=a, **evaluate(f, lab, h_obs, sig)))
        curve_rows.append(dict(sigma_cm=sig, rep=r, policy="No re-measurement", param=0,
                               **evaluate(np.zeros(len(h), bool), lab, h_obs, sig)))
    print("sigma", sig, "done")

MC = pd.DataFrame(mc_rows)
TC = MC.groupby("sigma_cm").agg(["mean", lambda x: np.percentile(x, 2.5), lambda x: np.percentile(x, 97.5)])
TC.columns = [f"{a}_{'mean' if b == 'mean' else ('lo' if b == '<lambda_0>' else 'hi')}" for a, b in TC.columns]
TC = TC.drop(columns=[c for c in TC.columns if c.startswith("rep_")]).round(3)
TC.to_csv(os.path.join(RES_DIR, "T_montecarlo_misclassification.csv")); print(TC.T)

CV = pd.DataFrame(curve_rows).groupby(["sigma_cm", "policy", "param"]).mean(numeric_only=True).drop(columns="rep").reset_index()
CV.round(3).to_csv(os.path.join(RES_DIR, "T_policy_curves.csv"), index=False)

# head-to-head table at sigma = 1.0 cm: draft band vs proposed (alpha=0.10) vs fixed band with equal burden
H = []
for sig in SIGMAS:
    c = CV[CV.sigma_cm == sig]
    none = c[c.policy == "No re-measurement"].iloc[0]
    draft = c[(c.policy == "Fixed HAZ band") & (np.isclose(c.param, 0.10))].iloc[0]
    prop = c[(c.policy == "Proposed probability band") & (np.isclose(c.param, ALPHA_MAIN))].iloc[0]
    fb = c[c.policy == "Fixed HAZ band"]; eq = fb.iloc[(fb.burden - prop.burden).abs().argmin()]
    for lbl, rr in [("P0 No re-measurement (hard threshold)", none), ("P1 Draft fixed band HAZ -2 +/- 0.10 SD", draft),
                    (f"P2 Fixed band +/- {eq.param:.2f} SD (budget-matched to P3)", eq),
                    (f"P3 Proposed per-child band (alpha = {ALPHA_MAIN})", prop)]:
        H.append(dict(sigma_cm=sig, policy=lbl, remeasure_burden_pct=rr.burden, errors_captured_pct=rr.capture,
                      yield_pct=rr.yield_pct, misclass_before_pct=rr.err_before, misclass_after_pct=rr.err_after,
                      error_reduction_pct=100 * (rr.err_before - rr.err_after) / rr.err_before))
TH = pd.DataFrame(H).round(3); TH.to_csv(os.path.join(RES_DIR, "T_policy_head_to_head.csv"), index=False)
print(TH.to_string())

# robustness: proposed rule built with sigma_assumed = 1.0 while true sigma differs
rob = []
for sig in SIGMAS:
    for r in range(20):
        h_obs = h + rng.normal(0, sig, len(h)); z_obs = zfromh(h_obs); lab = (z_obs < STUNT_CUT).astype(int)
        pst = norm.cdf((STUNT_CUT - z_obs) / (SIGMA_CM_MAIN * zpc))
        f = (pst > ALPHA_MAIN) & (pst < 1 - ALPHA_MAIN)
        rob.append(dict(true_sigma=sig, assumed_sigma=SIGMA_CM_MAIN, **evaluate(f, lab, h_obs, sig)))
TR = pd.DataFrame(rob).groupby(["true_sigma", "assumed_sigma"]).mean().reset_index().round(3)
TR.to_csv(os.path.join(RES_DIR, "T_policy_sigma_misspecification.csv"), index=False); print(TR)

# age-specific capture at sigma = 1.0 cm: fixed band vs proposed at equal burden
ag = []
k_eq = float(TH[(TH.sigma_cm == SIGMA_CM_MAIN) & TH.policy.str.startswith("P2")].policy.str.extract(r"([0-9.]+) SD")[0].iloc[0])
for r in range(30):
    h_obs = h + rng.normal(0, SIGMA_CM_MAIN, len(h)); z_obs = zfromh(h_obs); lab = (z_obs < STUNT_CUT).astype(int)
    err = lab != y_true
    pst = norm.cdf((STUNT_CUT - z_obs) / (SIGMA_CM_MAIN * zpc))
    flags = {"P1 draft +/-0.10 SD": np.abs(z_obs - STUNT_CUT) <= 0.10,
             f"P2 fixed +/-{k_eq:.2f} SD": np.abs(z_obs - STUNT_CUT) <= k_eq,
             "P3 proposed (alpha=0.10)": (pst > ALPHA_MAIN) & (pst < 1 - ALPHA_MAIN)}
    for pn, f in flags.items():
        for g in GROUPS:
            m = AGEG == g
            ag.append(dict(policy=pn, age_group=g, rep=r, burden=100 * w[m & f].sum() / w[m].sum(),
                           capture=100 * w[m & f & err].sum() / w[m & err].sum()))
TG = pd.DataFrame(ag).groupby(["policy", "age_group"]).mean(numeric_only=True).drop(columns="rep").reset_index().round(2)
TG.to_csv(os.path.join(RES_DIR, "T_policy_capture_by_age.csv"), index=False); print(TG)

# ================================================================== FIGURES
# F03: SD per cm and cm-equivalent of 0.10 SD by age
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
for k, (sx, lab) in enumerate([(1, "Boys"), (2, "Girls")]):
    q = lms[lms.sex == sx]
    ax[0].plot(q.m_months, q.sd_per_cm, color=PAL[k], lw=2, label=lab)
    ax[1].plot(q.m_months, q["cm_per_0.10SD"], color=PAL[k], lw=2, label=lab)
ax[0].set_xlabel("Age (months)"); ax[0].set_ylabel("HAZ units per 1 cm error"); ax[0].set_title("(a) Impact of a 1 cm error")
for t, ls in [(0.5, ":"), (1.0, "--")]:
    ax[1].axhline(t, color="#555", ls=ls, lw=1)
    ax[1].text(59, t + 0.02, f"field error SD {t} cm", ha="right", fontsize=7, color="#555")
ax[1].set_xlabel("Age (months)"); ax[1].set_ylabel("cm equivalent of 0.10 SD"); ax[1].set_ylim(0, 1.2)
ax[1].set_title("(b) Draft band +/-0.10 SD in cm"); ax[0].legend(); ax[1].legend(loc="center right")
fig.savefig(os.path.join(FIG_DIR, "F03_sd_per_cm.png")); plt.close(fig)

# F04: switching by scenario and age group
fig, ax = plt.subplots(figsize=(9, 3.4))
tb = TB.set_index("scenario"); x = np.arange(len(tb)); bw_ = 0.26
for k, g in enumerate(GROUPS):
    ax.bar(x + (k - 1) * bw_, tb[f"switch_{g}_pct"], width=bw_ - 0.03, color=PAL[k], label=g)
ax.set_xticks(x); ax.set_xticklabels(tb.index, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Children whose label switches (%)"); ax.set_title("Label switching under centimetre vs HAZ perturbation")
ax.legend(title="Age")
fig.savefig(os.path.join(FIG_DIR, "F04_label_switching.png")); plt.close(fig)

# F09: Monte Carlo misclassification by sigma and age
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
x = np.arange(len(SIGMAS))
for k, g in enumerate(GROUPS):
    ax[0].bar(x + (k - 1) * 0.26, TC[f"misclass_{g}_pct_mean"], 0.23, color=PAL[k], label=g,
              yerr=[TC[f"misclass_{g}_pct_mean"] - TC[f"misclass_{g}_pct_lo"], TC[f"misclass_{g}_pct_hi"] - TC[f"misclass_{g}_pct_mean"]], capsize=2)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{s} cm" for s in SIGMAS]); ax[0].set_xlabel("Random measurement error SD")
ax[0].set_ylabel("Wrong stunting label (%)"); ax[0].set_title("(a) Misclassification by age"); ax[0].legend(fontsize=8)
ax[1].bar(x - 0.18, TC["false_stunted_pct_mean"], 0.34, color=PAL[1], label="Falsely labelled stunted")
ax[1].bar(x + 0.18, TC["missed_stunted_pct_mean"], 0.34, color=PAL[0], label="Stunting missed")
for i, s in enumerate(SIGMAS):
    ax[1].text(i, max(TC.loc[s, "false_stunted_pct_mean"], TC.loc[s, "missed_stunted_pct_mean"]) + 0.15,
               f"prev {TC.loc[s, 'prev_obs_pct_mean']:.2f}%", ha="center", fontsize=7)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{s} cm" for s in SIGMAS]); ax[1].set_xlabel("Random measurement error SD")
ax[1].set_ylabel("% of children"); ax[1].set_title(f"(b) Error direction (true prev {prev0:.2f}%)"); ax[1].legend(fontsize=8)
fig.savefig(os.path.join(FIG_DIR, "F09_montecarlo_misclassification.png")); plt.close(fig)

# F10: burden-capture trade-off curves
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
for a, sig in zip(ax, SIGMAS):
    c = CV[CV.sigma_cm == sig]
    fb = c[c.policy == "Fixed HAZ band"]; pp = c[c.policy == "Proposed probability band"]
    a.plot(fb.burden, fb.capture, "-", color=PAL[1], lw=1.8, label="Fixed HAZ band (vary width)")
    a.plot(pp.burden, pp.capture, "-o", color=PAL[0], lw=1.8, ms=3.5, label="Proposed per-child band (vary alpha)")
    d1 = fb[np.isclose(fb.param, 0.10)].iloc[0]; d2 = pp[np.isclose(pp.param, ALPHA_MAIN)].iloc[0]
    a.plot(d1.burden, d1.capture, "s", color=PAL[1], ms=8, mec="k"); a.annotate("draft +/-0.10 SD", (d1.burden, d1.capture), xytext=(4, -12), textcoords="offset points", fontsize=7)
    a.plot(d2.burden, d2.capture, "D", color=PAL[0], ms=8, mec="k"); a.annotate(f"alpha={ALPHA_MAIN}", (d2.burden, d2.capture), xytext=(4, -12), textcoords="offset points", fontsize=7)
    a.set_xlim(0, 30); a.set_title(f"Error SD = {sig} cm"); a.set_xlabel("Children re-measured (%)")
ax[0].set_ylabel("Label errors captured (%)"); ax[0].legend(fontsize=7, loc="lower right")
fig.savefig(os.path.join(FIG_DIR, "F10_policy_tradeoff.png")); plt.close(fig)
print("done")
