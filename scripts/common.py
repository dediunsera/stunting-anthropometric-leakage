"""
common.py - shared configuration and helper functions
Reliable and Survey-Aware ML for Childhood Stunting Risk Screening
Under Anthropometric Measurement Uncertainty (SKI 2023)

All scripts import this module. Change DATA_DIR / OUT_DIR for Google Colab
(see 00_colab_setup.md or the notebook's first cell).
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

# ------------------------------------------------------------------ paths
BASE_DIR = os.environ.get(
    "SKI_PROJECT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_DIR = os.path.join(BASE_DIR, "data")
RES_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
for _d in (RES_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

RAW_XLSX = os.path.join(DATA_DIR, "SKI 2023 Balita 0-59 Bulan_asli labels.xlsx")
RAW_CACHE = os.path.join(DATA_DIR, "ski_raw.pkl")
LMS_FILE = os.path.join(DATA_DIR, "lenanthro.txt")
ANALYTIC = os.path.join(DATA_DIR, "analytic.pkl")

# ------------------------------------------------------------------ config
SEED = 42
N_FOLDS = 5
TEST_FRAC = 0.20                                   # share of PSUs in locked test set
SPATIAL_HOLDOUT = ["Sumatera Barat", "Bali", "Maluku"]
STUNT_CUT = -2.0
# measurement-error scenarios (random error SD in cm); see manuscript Sec. 2.6
SIGMA_CM_SCENARIOS = [0.5, 1.0, 1.5]
SIGMA_CM_MAIN = 1.0
ALPHA_MAIN = 0.10          # tolerated probability that a confirmed label is wrong
ALPHA_GRID = [0.05, 0.10, 0.20]
POSITION_BIAS_CM = 0.7     # WHO recumbent-vs-standing difference

W = "w"            # survey weight column in analytic file
PSU = "psu"
STRATA = "strata"


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


# ------------------------------------------------------------------ LMS
def load_lms():
    lms = pd.read_csv(LMS_FILE, sep="\t")
    lms.columns = [c.strip().lower() for c in lms.columns]
    return lms  # sex(1=M,2=F), age(days 0..1856), l, m, s, loh


def lms_lookup(lms, sex, age_days):
    key = pd.DataFrame({"sex": sex.astype(int), "age": age_days.astype(int)})
    out = key.merge(lms[["sex", "age", "l", "m", "s"]], on=["sex", "age"], how="left")
    return out["l"].values, out["m"].values, out["s"].values


def haz_from_height(h, L, M, S):
    """WHO LMS z-score. For length/height L==1 so z=(h/M-1)/S."""
    return (np.power(h / M, L) - 1.0) / (L * S)


def z_per_cm(M, S, L=1.0):
    """dz/dh at the observed point (L=1 -> 1/(M*S)); SD units per 1 cm."""
    return 1.0 / (M * S)


# ------------------------------------------------------------------ weighted metrics
def w_auc(y, p, w):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p, sample_weight=w)


def w_auprc(y, p, w):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, p, sample_weight=w)


def w_brier(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def w_ece(y, p, w, n_bins=10):
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece, W_ = 0.0, np.sum(w)
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        wb = w[m]
        ece += wb.sum() / W_ * abs(np.average(y[m], weights=wb) - np.average(p[m], weights=wb))
    return float(ece)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _wlogreg(X, y, w, offset=None, iters=50):
    """Weighted logistic regression via IRLS (unpenalised). Returns coef."""
    n, k = X.shape
    beta = np.zeros(k)
    off = np.zeros(n) if offset is None else offset
    for _ in range(iters):
        eta = X @ beta + off
        mu = 1 / (1 + np.exp(-eta))
        Wd = w * mu * (1 - mu) + 1e-12
        z = eta - off + (y - mu) / (mu * (1 - mu) + 1e-12)
        XtW = X.T * Wd
        new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def cal_slope_intercept(y, p, w):
    """Calibration slope (logit recalibration) and calibration-in-the-large
    intercept (slope fixed at 1 via offset). Ideal: slope 1, intercept 0."""
    lp = _logit(p)
    X = np.column_stack([np.ones_like(lp), lp])
    slope = _wlogreg(X, y.astype(float), w)[1]
    intercept = _wlogreg(np.ones((len(lp), 1)), y.astype(float), w, offset=lp)[0]
    return float(slope), float(intercept)


def all_metrics(y, p, w):
    y = np.asarray(y).astype(int); p = np.asarray(p, float); w = np.asarray(w, float)
    try:
        s, i = cal_slope_intercept(y, p, w)
    except np.linalg.LinAlgError:      # constant predictions (prevalence baseline)
        s, i = np.nan, float(np.log(np.average(y, weights=w) / (1 - np.average(y, weights=w)))
                             - np.log(p.mean() / (1 - p.mean())))
    return dict(n=int(len(y)), prev_w=float(np.average(y, weights=w)),
                auroc=w_auc(y, p, w), auprc=w_auprc(y, p, w),
                brier=w_brier(y, p, w), ece=w_ece(y, p, w),
                cal_slope=s, cal_intercept=i)


def psu_bootstrap(y, p, w, psu, fn, B=200, seed=SEED):
    """Cluster (PSU) bootstrap percentile CI for metric fn(y,p,w)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); w = np.asarray(w); psu = np.asarray(psu)
    codes, inv = np.unique(psu, return_inverse=True)
    idx_by = np.split(np.argsort(inv), np.cumsum(np.bincount(inv))[:-1])
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(codes), len(codes))
        idx = np.concatenate([idx_by[k] for k in pick])
        try:
            vals.append(fn(y[idx], p[idx], w[idx]))
        except ValueError:
            pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def net_benefit(y, p, w, thresholds):
    W_ = np.sum(w)
    out = []
    for t in thresholds:
        pos = p >= t
        tp = np.sum(w[pos & (y == 1)]) / W_
        fp = np.sum(w[pos & (y == 0)]) / W_
        out.append(tp - fp * t / (1 - t))
    return np.array(out)


def prob_true_stunted(haz_obs, sigma_z):
    """Measurement-error model: true HAZ ~ N(haz_obs, sigma_z^2) (flat prior).
    Returns P(true HAZ < -2 | observed)."""
    return norm.cdf((STUNT_CUT - haz_obs) / sigma_z)
