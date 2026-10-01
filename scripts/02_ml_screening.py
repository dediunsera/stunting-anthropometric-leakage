"""
02_ml_screening.py
Stage 2 - Survey-aware, leakage-free risk model (Axis B of the framework).

 * spatial holdout (3 provinces) + PSU-grouped locked test set (20% PSUs)
 * 5-fold PSU-grouped CV on the development set -> out-of-fold (OOF) predictions
 * models: prevalence baseline, LR (unweighted / survey-weighted),
           RF (weighted), HGB (unweighted / weighted)
 * comparators: leaky model (child weight/height), random (non-grouped) CV,
                model without birth anthropometry
 * post-hoc calibration fitted on OOF only (Platt, isotonic), frozen for test/holdout
 * risk-tier threshold fixed on development OOF (weighted Youden J)

Outputs: results/T_model_*.csv, results/pred_*.pkl, figures/F_*.png
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve
from common import *
from plot_style import apply_style, COLORBLIND_SAFE_PALETTE as PAL

apply_style(10)
t0 = time.time()
A = pd.read_pickle(ANALYTIC)
df, FEAT, BIRTH = A["df"].reset_index(drop=True).copy(), A["features"], A["birth_anthro"]
CAT = [c for c in FEAT if df[c].dtype == object or str(df[c].dtype).startswith("str")]
NUM = [c for c in FEAT if c not in CAT]
for c in CAT:
    df[c] = df[c].astype(object)
print(f"features {len(FEAT)} (cat {len(CAT)}, num {len(NUM)})")

# ------------------------------------------------------------------ splits
hold = df.province.isin(SPATIAL_HOLDOUT).values
rest = np.where(~hold)[0]
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
tr_i, te_i = next(gss.split(rest, groups=df[PSU].values[rest]))
split = np.array(["holdout"] * len(df), dtype=object)
split[rest[tr_i]] = "dev"; split[rest[te_i]] = "test"
df["split"] = split
assert not set(df.loc[df.split == "dev", PSU]) & set(df.loc[df.split == "test", PSU])
spl = df.groupby("split").agg(n=("stunted", "size"), n_psu=(PSU, "nunique"),
                               prev_unw=("stunted", "mean"))
spl["prev_w"] = df.groupby("split").apply(lambda g: np.average(g.stunted, weights=g[W]))
spl.to_csv(os.path.join(RES_DIR, "T_split_summary.csv")); print(spl)

dev = df[df.split == "dev"].reset_index(drop=True)
tst = df[df.split == "test"].reset_index(drop=True)
hol = df[df.split == "holdout"].reset_index(drop=True)


# ------------------------------------------------------------------ model factory
def pre_linear(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat)])


def pre_tree(num, cat):
    return ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), cat)])


def make(name, num, cat):
    if name.startswith("LR"):
        return Pipeline([("pre", pre_linear(num, cat)),
                         ("clf", LogisticRegression(C=0.1, max_iter=3000))])
    if name.startswith("RF"):
        return Pipeline([("pre", pre_linear(num, cat)),
                         ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                        max_features="sqrt", n_jobs=-1,
                                                        random_state=SEED))])
    if name.startswith("HGB"):
        catmask = [False] * len(num) + [True] * len(cat)
        return Pipeline([("pre", pre_tree(num, cat)),
                         ("clf", HistGradientBoostingClassifier(
                             max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                             min_samples_leaf=50, l2_regularization=1.0,
                             categorical_features=catmask, early_stopping=False,
                             random_state=SEED))])
    raise ValueError(name)


def fit(model, X, y, w):
    model.fit(X, y, clf__sample_weight=None if w is None else w / w.mean())
    return model


MODELS = {  # name: (weighted?, features)
    "LR (unweighted)": (False, FEAT),
    "LR (survey-weighted)": (True, FEAT),
    "RF (survey-weighted)": (True, FEAT),
    "HGB (unweighted)": (False, FEAT),
    "HGB (survey-weighted)": (True, FEAT),
}
LEAKY_EXTRA = ["child_weight_kg", "height_raw"]
dev["child_weight_kg"] = np.nan  # placeholder, filled below from raw
raw = pd.read_pickle(RAW_CACHE).set_index("ID ART")["J01.c.Berat Badan (kg)"]
for d in (df, dev, tst, hol):
    d["child_weight_kg"] = d["ID ART"].map(raw).values

COMPARATORS = {
    "HGB weighted + child height & weight (LEAKY)": (True, FEAT + LEAKY_EXTRA, "grouped"),
    "HGB weighted, random (non-grouped) CV": (True, FEAT, "random"),
    "HGB weighted, without birth anthropometry": (True, [f for f in FEAT if f not in BIRTH], "grouped"),
}


import joblib, hashlib
CACHE = os.path.join(DATA_DIR, "cache"); os.makedirs(CACHE, exist_ok=True)


def oof_predict(name, weighted, feats, scheme="grouped"):
    key = hashlib.md5(f"{name}|{weighted}|{scheme}|{len(feats)}|{sorted(feats)}".encode()).hexdigest()[:12]
    fp = os.path.join(CACHE, f"oof_{key}.joblib")
    if os.path.exists(fp):
        return joblib.load(fp)
    res = _oof_predict(name, weighted, feats, scheme)
    joblib.dump(res, fp)
    return res


def _oof_predict(name, weighted, feats, scheme="grouped"):
    num = [c for c in feats if c not in CAT]; cat = [c for c in feats if c in CAT]
    X, y, w, g = dev[feats], dev.stunted.values, dev[W].values, dev[PSU].values
    oof = np.zeros(len(dev))
    cv = GroupKFold(n_splits=N_FOLDS) if scheme == "grouped" else KFold(N_FOLDS, shuffle=True, random_state=SEED)
    it = cv.split(X, y, groups=g) if scheme == "grouped" else cv.split(X, y)
    for k, (a, b) in enumerate(it):
        m = fit(make(name, num, cat), X.iloc[a], y[a], w[a] if weighted else None)
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    full = fit(make(name, num, cat), X, y, w if weighted else None)
    return oof, full


def platt_fit(p, y, w):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    X = np.column_stack([np.ones_like(lp), lp])
    b = __import__("common")._wlogreg(X, y.astype(float), w)
    return lambda q: 1 / (1 + np.exp(-(b[0] + b[1] * np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))))))


def iso_fit(p, y, w):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y, sample_weight=w)
    return ir.predict


rows, preds = [], {}
yd, wd = dev.stunted.values, dev[W].values
for name, (weighted, feats) in MODELS.items():
    t = time.time()
    oof, full = oof_predict(name, weighted, feats)
    pt, ph = full.predict_proba(tst[feats])[:, 1], full.predict_proba(hol[feats])[:, 1]
    cal_p, cal_i = platt_fit(oof, yd, wd), iso_fit(oof, yd, wd)
    preds[name] = dict(oof=oof, test=pt, hold=ph, model=full, platt=cal_p, iso=cal_i)
    for part, y_, w_, p_ in [("OOF (dev)", yd, wd, oof), ("Locked test", tst.stunted.values, tst[W].values, pt),
                             ("Spatial holdout", hol.stunted.values, hol[W].values, ph)]:
        for cal, f in [("raw", lambda x: x), ("Platt", cal_p), ("isotonic", cal_i)]:
            if part == "OOF (dev)" and cal != "raw":
                continue  # calibrators are fitted on OOF: do not report in-sample
            rows.append(dict(model=name, partition=part, calibration=cal, **all_metrics(y_, f(p_), w_)))
    print(f"{name:28s} OOF AUROC {w_auc(yd, oof, wd):.4f}  ({time.time()-t:.0f}s)")

# prevalence baseline
prev = np.average(yd, weights=wd)
for part, d in [("Locked test", tst), ("Spatial holdout", hol)]:
    rows.append(dict(model="Prevalence baseline", partition=part, calibration="raw",
                     **all_metrics(d.stunted.values, np.full(len(d), prev) + 1e-9 * np.random.rand(len(d)), d[W].values)))

comp_rows = []
for name, (weighted, feats, scheme) in COMPARATORS.items():
    t = time.time()
    oof, full = oof_predict("HGB", weighted, feats, scheme)
    pt = full.predict_proba(tst[feats])[:, 1]
    comp_rows.append(dict(model=name, cv_scheme=scheme, oof_auroc=w_auc(yd, oof, wd),
                          test_auroc=w_auc(tst.stunted.values, pt, tst[W].values),
                          optimism=w_auc(yd, oof, wd) - w_auc(tst.stunted.values, pt, tst[W].values)))
    print(f"{name:45s} OOF {comp_rows[-1]['oof_auroc']:.4f} test {comp_rows[-1]['test_auroc']:.4f} ({time.time()-t:.0f}s)")
ref = [r for r in rows if r["model"] == "HGB (survey-weighted)" and r["partition"] in ("OOF (dev)",)][0]
reft = [r for r in rows if r["model"] == "HGB (survey-weighted)" and r["partition"] == "Locked test" and r["calibration"] == "raw"][0]
comp_rows.insert(0, dict(model="HGB weighted, PSU-grouped CV (proposed)", cv_scheme="grouped",
                         oof_auroc=ref["auroc"], test_auroc=reft["auroc"], optimism=ref["auroc"] - reft["auroc"]))
pd.DataFrame(comp_rows).round(4).to_csv(os.path.join(RES_DIR, "T_leakage_and_cv_comparators.csv"), index=False)

M = pd.DataFrame(rows)
M.round(4).to_csv(os.path.join(RES_DIR, "T_model_metrics_all.csv"), index=False)

# ------------------------------------------------------------------ bootstrap CI on locked test (Platt)
ci_rows = []
for name in MODELS:
    p = preds[name]["platt"](preds[name]["test"])
    y_, w_, g_ = tst.stunted.values, tst[W].values, tst[PSU].values
    a_lo, a_hi = psu_bootstrap(y_, p, w_, g_, w_auc)
    e_lo, e_hi = psu_bootstrap(y_, p, w_, g_, w_ece)
    ci_rows.append(dict(model=name, auroc=w_auc(y_, p, w_), auroc_lo=a_lo, auroc_hi=a_hi,
                        ece=w_ece(y_, p, w_), ece_lo=e_lo, ece_hi=e_hi,
                        brier=w_brier(y_, p, w_), auprc=w_auprc(y_, p, w_)))
CI = pd.DataFrame(ci_rows); CI.round(4).to_csv(os.path.join(RES_DIR, "T_model_comparison_test_CI.csv"), index=False)
print(CI.round(4))

# ------------------------------------------------------------------ select final model
# pre-specified rule: highest survey-weighted OOF AUROC on the development set
oof_auc = {n: w_auc(yd, preds[n]["oof"], wd) for n in MODELS}
BEST = max(oof_auc, key=oof_auc.get)
print("selected model:", BEST, oof_auc)
calP = preds[BEST]["platt"]
p_oof_cal = calP(preds[BEST]["oof"])
# risk threshold fixed on development OOF: weighted Youden J
fpr, tpr, thr = roc_curve(yd, p_oof_cal, sample_weight=wd)
j = tpr - fpr; T_RISK = float(thr[np.argmax(j)])
save_json(dict(best_model=BEST, risk_threshold=T_RISK, prevalence_dev_w=prev), os.path.join(RES_DIR, "config_locked.json"))
print("locked risk threshold", T_RISK)

out = {}
for nm, d, key in [("dev", dev, "oof"), ("test", tst, "test"), ("hold", hol, "hold")]:
    o = d[["ID ART", "haz", "stunted", "height_adj", "height_raw", "position", "position_mismatch",
           "age_days", "age_months", "sex", "urban", "z_per_cm", "lms_M", "lms_S", "lms_L", W, PSU, "province"]].copy()
    o["p_raw"] = preds[BEST][key]; o["p_cal"] = calP(o["p_raw"].values)
    out[nm] = o
pd.to_pickle(out, os.path.join(DATA_DIR, "predictions.pkl"))

# threshold performance
thr_rows = []
for nm in ("test", "hold"):
    o = out[nm]; y_, w_ = o.stunted.values, o[W].values; hi = o.p_cal.values >= T_RISK
    sens = np.sum(w_[hi & (y_ == 1)]) / np.sum(w_[y_ == 1]); spec = np.sum(w_[~hi & (y_ == 0)]) / np.sum(w_[y_ == 0])
    ppv = np.sum(w_[hi & (y_ == 1)]) / np.sum(w_[hi]); npv = np.sum(w_[~hi & (y_ == 0)]) / np.sum(w_[~hi])
    thr_rows.append(dict(partition=nm, threshold=T_RISK, flagged_high_w_pct=100 * w_[hi].sum() / w_.sum(),
                         sensitivity=sens, specificity=spec, ppv=ppv, npv=npv))
pd.DataFrame(thr_rows).round(4).to_csv(os.path.join(RES_DIR, "T_risk_threshold_performance.csv"), index=False)
print(pd.DataFrame(thr_rows).round(3))

# ------------------------------------------------------------------ subgroup analysis (test+holdout, Platt)
sub_rows = []
ev = pd.concat([out["test"].assign(part="Locked test"), out["hold"].assign(part="Spatial holdout")])
groups = {"Age 0-5 mo": ev.age_months < 6, "Age 6-23 mo": ev.age_months.between(6, 24, inclusive="left"),
          "Age 24-59 mo": ev.age_months >= 24, "Boys": ev.sex == 1, "Girls": ev.sex == 2,
          "Urban": ev.urban == 1, "Rural": ev.urban == 0,
          "Locked test": ev.part == "Locked test", "Spatial holdout": ev.part == "Spatial holdout"}
for p_ in SPATIAL_HOLDOUT:
    groups["Holdout: " + p_] = ev.province == p_
for g, m in groups.items():
    e = ev[m.values]; y_, p_, w_ = e.stunted.values, e.p_cal.values, e[W].values
    lo, hi_ = psu_bootstrap(y_, p_, w_, e[PSU].values, w_auc, B=200)
    mt = all_metrics(y_, p_, w_)
    sub_rows.append(dict(group=g, auroc_lo=lo, auroc_hi=hi_, **mt))
SUB = pd.DataFrame(sub_rows); SUB.round(4).to_csv(os.path.join(RES_DIR, "T_subgroup.csv"), index=False)
print(SUB[["group", "n", "auroc", "auroc_lo", "auroc_hi", "ece", "cal_intercept"]].round(3))

# ------------------------------------------------------------------ grouped permutation importance (domain level)
from common import w_auc as _wa
feat_dom = pd.read_csv(os.path.join(RES_DIR, "T_leakage_audit.csv")).query("retained").set_index("column").domain
model = preds[BEST]["model"]
base = _wa(tst.stunted.values, model.predict_proba(tst[FEAT])[:, 1], tst[W].values)
rng = np.random.default_rng(SEED); imp = []
for dname in sorted(feat_dom.unique()):
    cols = [c for c in FEAT if feat_dom.get(c) == dname]
    drops = []
    for r in range(5):
        Xp = tst[FEAT].copy(); perm = rng.permutation(len(Xp))
        Xp[cols] = Xp[cols].values[perm]
        drops.append(base - _wa(tst.stunted.values, model.predict_proba(Xp)[:, 1], tst[W].values))
    imp.append(dict(domain=dname, n_features=len(cols), auroc_drop=np.mean(drops), sd=np.std(drops)))
single = []
for c in FEAT:
    Xp = tst[FEAT].copy(); Xp[c] = Xp[c].values[rng.permutation(len(Xp))]
    single.append(dict(feature=c, domain=feat_dom.get(c), auroc_drop=base - _wa(tst.stunted.values, model.predict_proba(Xp)[:, 1], tst[W].values)))
IMP = pd.DataFrame(imp).sort_values("auroc_drop", ascending=False)
IMP.round(5).to_csv(os.path.join(RES_DIR, "T_importance_domain.csv"), index=False)
SI = pd.DataFrame(single).sort_values("auroc_drop", ascending=False)
SI.round(5).to_csv(os.path.join(RES_DIR, "T_importance_feature.csv"), index=False)
print(IMP.round(4)); print(SI.head(15).round(4))

# ================================================================== FIGURES
# F5 model comparison (AUROC with CI | ECE with CI)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
order = CI.sort_values("auroc").model.tolist(); c = CI.set_index("model").loc[order]
yy = np.arange(len(order))
ax[0].errorbar(c.auroc, yy, xerr=[c.auroc - c.auroc_lo, c.auroc_hi - c.auroc], fmt="o", color=PAL[0], capsize=3)
ax[0].axvline(0.5, color="#888", ls="--", lw=1); ax[0].set_yticks(yy); ax[0].set_yticklabels(order)
ax[0].set_xlabel("Weighted AUROC (95% PSU-bootstrap CI)"); ax[0].set_title("Discrimination (locked test)")
ax[1].errorbar(c.ece, yy, xerr=[c.ece - c.ece_lo, c.ece_hi - c.ece], fmt="s", color=PAL[1], capsize=3)
ax[1].set_yticks(yy); ax[1].set_yticklabels([]); ax[1].set_xlabel("Weighted ECE after Platt (95% CI)")
ax[1].set_title("Calibration (locked test)")
fig.savefig(os.path.join(FIG_DIR, "F05_model_comparison.png")); plt.close(fig)

# F6 ROC overlay + DCA
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, name in enumerate(MODELS):
    f_, t_, _ = roc_curve(tst.stunted.values, preds[name]["test"], sample_weight=tst[W].values)
    ax[0].plot(f_, t_, lw=1.6, color=PAL[k], label=f"{name} ({w_auc(tst.stunted.values, preds[name]['test'], tst[W].values):.3f})")
f_, t_, _ = roc_curve(tst.stunted.values, np.zeros(len(tst)) + 1e-9 * np.random.rand(len(tst)), sample_weight=tst[W].values)
ax[0].plot([0, 1], [0, 1], "--", color="#888", lw=1)
ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("Sensitivity"); ax[0].set_title("Weighted ROC, locked test")
ax[0].legend(fontsize=7, loc="lower right")
ts = np.linspace(0.05, 0.60, 56)
yt, wt = tst.stunted.values, tst[W].values
pc = out["test"].p_cal.values
ax[1].plot(ts, net_benefit(yt, pc, wt, ts), color=PAL[0], lw=2, label=f"{BEST} + Platt")
ax[1].plot(ts, net_benefit(yt, np.ones(len(yt)), wt, ts), color=PAL[1], lw=1.5, label="Measure/act on all")
ax[1].plot(ts, np.zeros_like(ts), color="#555", lw=1.2, label="Act on none")
ax[1].axvline(T_RISK, color=PAL[2], ls=":", lw=1.5, label=f"Locked risk threshold = {T_RISK:.3f}")
ax[1].set_ylim(-0.05, 0.25); ax[1].set_xlabel("Threshold probability"); ax[1].set_ylabel("Net benefit")
ax[1].set_title("Decision curve, locked test"); ax[1].legend(fontsize=7)
fig.savefig(os.path.join(FIG_DIR, "F06_roc_dca.png")); plt.close(fig)

# F7 reliability diagram raw vs Platt vs isotonic on test and holdout, weighted bins + CI via Neff
def rel_bins(y, p, w, nb=10):
    b = np.clip((p * nb).astype(int), 0, nb - 1); r = []
    for k in range(nb):
        m = b == k
        if m.sum() < 5: continue
        ww = w[m]; neff = ww.sum() ** 2 / np.sum(ww ** 2); o = np.average(y[m], weights=ww)
        se = np.sqrt(o * (1 - o) / neff)
        r.append(dict(bin=f"[{k/nb:.1f},{(k+1)/nb:.1f})", n_raw=int(m.sum()), n_eff=neff, w_frac=ww.sum() / w.sum(),
                      mean_pred=np.average(p[m], weights=ww), obs=o, lo=max(0, o - 1.96 * se), hi=min(1, o + 1.96 * se)))
    return pd.DataFrame(r)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
bin_tabs = []
for a, (nm, d, key) in zip(ax, [("Locked test", tst, "test"), ("Spatial holdout", hol, "hold")]):
    for k, (lbl, f) in enumerate([("Raw", lambda x: x), ("Platt", preds[BEST]["platt"]), ("Isotonic", preds[BEST]["iso"])]):
        p = f(preds[BEST][key]); rb = rel_bins(d.stunted.values, p, d[W].values)
        rb.insert(0, "calibration", lbl); rb.insert(0, "partition", nm); bin_tabs.append(rb)
        a.errorbar(rb.mean_pred, rb.obs, yerr=[rb.obs - rb.lo, rb.hi - rb.obs], fmt="-o", ms=4, lw=1.4,
                   capsize=2, color=PAL[k], label=lbl)
    a.plot([0, 1], [0, 1], "--", color="#888", lw=1); a.set_xlim(0, 0.8); a.set_ylim(0, 0.8)
    a.set_xlabel("Mean predicted probability"); a.set_ylabel("Observed stunting (weighted)"); a.set_title(nm)
    a.legend(fontsize=8)
pd.concat(bin_tabs).round(4).to_csv(os.path.join(RES_DIR, "T_calibration_bins.csv"), index=False)
fig.savefig(os.path.join(FIG_DIR, "F07_calibration.png")); plt.close(fig)

# F8 subgroup forest + domain importance
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
s = SUB.iloc[::-1]; yy = np.arange(len(s))
ax[0].errorbar(s.auroc, yy, xerr=[s.auroc - s.auroc_lo, s.auroc_hi - s.auroc], fmt="o", color=PAL[0], capsize=2)
ax[0].axvline(SUB.set_index("group").loc["Locked test", "auroc"], color="#888", ls=":", lw=1)
ax[0].set_yticks(yy); ax[0].set_yticklabels([f"{g} (n={n:,})" for g, n in zip(s.group, s.n)], fontsize=7.5)
ax[0].set_xlabel("Weighted AUROC (95% CI)"); ax[0].set_title("Subgroup discrimination")
im = IMP.iloc[::-1]
ax[1].barh(im.domain, im.auroc_drop, xerr=im.sd, color=PAL[2], height=0.6)
ax[1].set_xlabel("AUROC drop (domain permuted)"); ax[1].set_title("Domain importance (test)")
ax[1].tick_params(axis="y", labelsize=8)
fig.savefig(os.path.join(FIG_DIR, "F08_subgroup_importance.png")); plt.close(fig)

print(f"done in {time.time()-t0:.0f}s")
