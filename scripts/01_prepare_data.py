"""
01_prepare_data.py
Stage 1 - Data ingestion, audit, outcome construction (WHO LMS HAZ),
leakage audit and locked predictor set.

Outputs:
  data/analytic.pkl                 analytic dataset (features + design + outcome)
  results/T_flow_exclusion.csv      sample flow (for flow diagram)
  results/T_leakage_audit.csv       every raw column -> role / decision / reason
  results/T_feature_domains.csv     retained predictors by domain
  results/T_sample_profile.csv      weighted profile of analytic sample
"""
import re
import numpy as np
import pandas as pd
from common import *

np.random.seed(SEED)

# ------------------------------------------------------------------ load
if os.path.exists(RAW_CACHE):
    raw = pd.read_pickle(RAW_CACHE)
else:
    raw = pd.read_excel(RAW_XLSX)
    raw.to_pickle(RAW_CACHE)
print("raw shape", raw.shape)
df = raw.copy()

C_H = "J02.b.Tinggi/Panjang Badan (cm)"
C_POS = "J02.c.KHUSUS UNTUK BALITA, (Posisi pengukuran TB/PB)"
C_WT = "J01.c.Berat Badan (kg)"
C_SEX = "4. Jenis Kelamin"
C_DOB = "6. Tanggal Lahir"
C_DOI = "2. Tanggal Pengumpulan data: (tgl-bln)"
C_PROV = "1.Provinsi"
C_KAB = "2. Kabupaten/Kota"


def parse_dmy(x):
    s = x.astype("Int64").astype(str).str.zfill(8)
    return pd.to_datetime(s, format="%d%m%Y", errors="coerce")


flow = [("Raw SKI 2023 under-five records", len(df))]

# ------------------------------------------------------------------ age
dob, doi = parse_dmy(df[C_DOB]), parse_dmy(df[C_DOI])
df["age_days"] = (doi - dob).dt.days
# fallback to reported completed months when dates are invalid
fallback = (df["7. Umur Bulan"] * 30.4375 + df["7. Umur hari"]).round()
bad_date = df["age_days"].isna() | (df["age_days"] < 0) | (df["age_days"] > 1856)
df.loc[bad_date, "age_days"] = fallback[bad_date]
df["age_months"] = df["age_days"] / 30.4375
df["sex"] = df[C_SEX].map({"Laki-laki": 1, "Prempuan": 2})

m = df["age_days"].between(0, 1856)
df = df[m]; flow.append(("Age 0-59 months (0-1856 days)", len(df)))

m = df[C_H].notna() & (df[C_H] > 0)
df = df[m]; flow.append(("Length/height measured", len(df)))

# ------------------------------------------------------------------ WHO position adjustment
h = df[C_H].astype(float).copy()
pos = df[C_POS]
standing_u2 = (df["age_days"] < 731) & (pos == "Berdiri")
lying_o2 = (df["age_days"] >= 731) & (pos == "Telentang")
h[standing_u2] += POSITION_BIAS_CM
h[lying_o2] -= POSITION_BIAS_CM
df["height_adj"] = h
df["position_mismatch"] = (standing_u2 | lying_o2).astype(int)

lms = load_lms()
L, M, S = lms_lookup(lms, df["sex"], df["age_days"])
df["lms_L"], df["lms_M"], df["lms_S"] = L, M, S
df["haz"] = haz_from_height(df["height_adj"].values, L, M, S)
df["z_per_cm"] = z_per_cm(M, S)

m = df["haz"].between(-6, 6)
flow.append(("Excluded biologically implausible HAZ (|HAZ|>6, WHO flag)", int((~m).sum())))
df = df[m]; flow.append(("Valid HAZ", len(df)))

m = df["Penimbang Populasi Individu"].notna() & df["Primary Sampling Unit"].notna()
df = df[m]; flow.append(("Complete survey design (weight, PSU, strata)", len(df)))

m = ~df["ID ART"].duplicated()
df = df[m]; flow.append(("Final analytic sample (unique child ID)", len(df)))

df["stunted"] = (df["haz"] < STUNT_CUT).astype(int)
df[W] = df["Penimbang Populasi Individu"].astype(float)
df[PSU] = df["Primary Sampling Unit"].astype(np.int64)
df[STRATA] = df["STRATA"].astype(np.int64)
df["province"] = df[C_PROV]
df["urban"] = (df["5. Klasifikasi Desa/Kelurahan"] == "Perkotaan").astype(int)

# ------------------------------------------------------------------ sentinel cleaning
SENT = {
    "H01.Berapa umur [NAMA] ketika pertama kali hamil?": [88, 98],
    "I04.Usia kehamilan saat [NAMA] dilahirkan": [88, 98],
    "I05.a.\tBerapa berat badan [NAMA] saat dilahirkan": [888, 8888, 9999],
    "I07.Berapa panjang badan [NAMA] saat dilahirkan": [0, 88, 88.8, 888],
    "I10.Salin dari catatan/dokumen lingkar kepala [NAMA]": [88, 88.8],
    "1.a.Waktu yang diperlukan dari Rumah ke faskes - Puskesmas": [995, 996, 997, 998, 999],
}
for c, codes in SENT.items():
    df.loc[df[c].isin(codes), c] = np.nan
for c in df.columns:
    if c.startswith(("150.a", "I50.e.Jika", "I50.f.Jika", "I50.h.Jika")):
        df.loc[df[c] == 88, c] = np.nan
# implausible birth weight / length
bw = "I05.a.\tBerapa berat badan [NAMA] saat dilahirkan"
bl = "I07.Berapa panjang badan [NAMA] saat dilahirkan"
df.loc[~df[bw].between(500, 6000), bw] = np.nan
df.loc[~df[bl].between(30, 60), bl] = np.nan

# ------------------------------------------------------------------ leakage audit
def domain_of(c):
    if c in ("age_months", "sex") or c.startswith(("I15.", "A12", "A22", "B01", "B07", "B16", "J01.b", "I16")):
        return "Child"
    if c.startswith(("I04", "I05", "I07", "I09", "I10", "I12", "I01", "H09", "H29", "H32", "H01")):
        return "Birth, ANC & neonatal care"
    if c.startswith(("I37", "I50", "150.")):
        return "Infant & young child feeding"
    if c.startswith(("8. Pendidikan", "9. Status", "11. Kepemilikan", "G0", "G1", "G2", "G3", "J03", "J07")):
        return "Maternal / caregiver"
    if c.startswith(("1.a.", "1. Apakah [RUMAH")):
        return "Health-service access"
    if re.match(r"^(1\.Apakah jenis|6\.|6\.b|7\.|8\.Apakah|11\.|12\.|14\.|17\.|18\.|19\.|20\.|21\.)", c):
        return "WASH"
    if c.startswith(("25.", "26.", "27.", "28.", "2.Luas", "urban")):
        return "Housing & area"
    if c.startswith(("1. Banyaknya", "2. Banyaknya", "3. Banyaknya", "4. Banyaknya", "5. Banyaknya",
                     "3. Hubungan", "1.Apakah ada anggota")):
        return "Household demography"
    return "Other"


rows = []
EXCL = {}
# Tier 1: deterministic outcome inputs / outcome
for c in [C_H, C_POS, "haz", "stunted", "height_adj", "position_mismatch", "z_per_cm",
          "lms_L", "lms_M", "lms_S"]:
    EXCL[c] = ("T1 Deterministic outcome component", "Defines HAZ / stunting label")
# Tier 2: concurrent child anthropometry
EXCL[C_WT] = ("T2 Concurrent child anthropometry", "Child weight measured with height; strongly collinear with length")
EXCL["J01.a"] = EXCL[C_WT]
# Tier 3: post-outcome proxies (nutrition-status driven programme)
for c in df.columns:
    if c.startswith("I49"):
        EXCL[c] = ("T3 Post-outcome proxy", "Supplementary feeding given BECAUSE of nutritional status")
    if c.startswith(("G03.b", "G03.2", "G03.3", "G03.4", "G03.5", "G03.6", "G03.7", "G03.8", "G03.9", "G03.c")):
        EXCL[c] = ("T4 Low-quality / conditional item", ">50% structurally missing free-text-like knowledge item")
# Tier 4 identifiers & design
for c in ["ID ART", "ID RT", "ID Ibu", "ID Anak Terakhir", "Penimbang Populasi Individu",
          "Penimbang Populasi RT", "Primary Sampling Unit", "STRATA", W, PSU, STRATA,
          C_PROV, C_KAB, "province", C_DOB, C_DOI, "age_days", "7. Umur Bulan", "7. Umur hari",
          "Kode umur", "1. No Urut ART", C_SEX, "5. Klasifikasi Desa/Kelurahan"]:
    EXCL[c] = ("Design / identifier", "Used for weighting, clustering, splitting or recoded")
EXCL["age_days"] = ("Design / identifier", "Recoded to age_months")
EXCL[C_SEX] = ("Design / identifier", "Recoded to sex")
EXCL["5. Klasifikasi Desa/Kelurahan"] = ("Design / identifier", "Recoded to urban")

cand = [c for c in df.columns]
feat = []
for c in cand:
    miss = df[c].isna().mean() if c in df else np.nan
    if c in EXCL:
        role, reason = EXCL[c]; keep = False
    elif df[c].dtype == object and df[c].nunique() > 50:
        role, reason, keep = "Excluded", "Free text (>50 levels)", False
    elif miss > 0.95:
        role, reason, keep = "Excluded", "Missing >95%", False
    elif df[c].nunique(dropna=True) <= 1:
        role, reason, keep = "Excluded", "Constant", False
    else:
        role, reason, keep = "Retained predictor", "Available before anthropometric confirmation", True
    note = ""
    if c in (bl, bw, "I10.Salin dari catatan/dokumen lingkar kepala [NAMA]"):
        note = "Birth record (pre-outcome); sensitivity model without birth anthropometry"
    if c.startswith(("J03", "J07")):
        note = "Maternal measurement (values adult-range, r(age)~0), NOT child"
    rows.append(dict(column=c, role=role, reason=reason, retained=keep,
                     domain=domain_of(c) if keep else "", missing=round(float(miss), 4), note=note))
    if keep:
        feat.append(c)

audit = pd.DataFrame(rows)
audit.to_csv(os.path.join(RES_DIR, "T_leakage_audit.csv"), index=False)
dom = audit[audit.retained].groupby("domain").size().rename("n_features").reset_index()
dom.to_csv(os.path.join(RES_DIR, "T_feature_domains.csv"), index=False)
print(dom)
print("excluded by role:\n", audit[~audit.retained].role.value_counts())

# ------------------------------------------------------------------ save analytic
keep_cols = feat + ["haz", "stunted", "height_adj", C_H, C_POS, "position_mismatch", "age_days",
                    "z_per_cm", "lms_L", "lms_M", "lms_S", W, PSU, STRATA, "province", "ID ART"]
keep_cols = list(dict.fromkeys(keep_cols))
an = df[keep_cols].copy()
an = an.rename(columns={C_H: "height_raw", C_POS: "position"})
an.attrs["features"] = feat
pd.to_pickle({"df": an, "features": feat, "birth_anthro": [bl, bw, "I10.Salin dari catatan/dokumen lingkar kepala [NAMA]"]}, ANALYTIC)

pd.DataFrame(flow, columns=["step", "n"]).to_csv(os.path.join(RES_DIR, "T_flow_exclusion.csv"), index=False)

# ------------------------------------------------------------------ profile
w = an[W].values
prof = []
def add(lbl, mask):
    prof.append(dict(characteristic=lbl, n=int(mask.sum()),
                     weighted_pct=100 * w[mask].sum() / w.sum(),
                     stunting_w_pct=100 * np.average(an.stunted[mask], weights=w[mask])))
add("All", np.ones(len(an), bool))
for lo, hi, lbl in [(0, 6, "0-5 mo"), (6, 24, "6-23 mo"), (24, 60, "24-59 mo")]:
    add("Age " + lbl, an.age_months.between(lo, hi, inclusive="left").values)
add("Boys", (an.sex == 1).values); add("Girls", (an.sex == 2).values)
add("Urban", (an.urban == 1).values); add("Rural", (an.urban == 0).values)
add("Position mismatch corrected", (an.position_mismatch == 1).values)
pd.DataFrame(prof).round(2).to_csv(os.path.join(RES_DIR, "T_sample_profile.csv"), index=False)
print(pd.DataFrame(prof).round(2))
print(pd.DataFrame(flow))
print("n features", len(feat), "| unweighted prev", an.stunted.mean(), "| weighted",
      np.average(an.stunted, weights=w))
