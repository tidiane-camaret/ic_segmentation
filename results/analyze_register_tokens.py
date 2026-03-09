"""
Analyze how register tokens correlate with case metadata.

For each metadata variable and each resolution level (0,1,2):
  - Classification tasks (LogReg): anatomical label, gender, pathology,
    manufacturer, scanner_model, institute, study_type, pathology_location, kvp_bin
  - Regression tasks (Ridge): age, kvp

Train on val cases, evaluate on test cases.
Results saved to register_token_probe_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    r2_score, mean_absolute_error
)
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation"
                   "/ANALYSIS_20251122/results/checkpoints"
                   "/2026-03-08_firm-shadow-201_register_tokens")
META_CSV    = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation"
                   "/ANALYSIS_20251122/data/totalseg/meta.csv")
OUT_CSV     = Path(__file__).parent / "register_token_probe_results.csv"

N_LEVELS = 3

# ── Load register tokens & metadata ───────────────────────────────────────────
print("Loading cases...")
case_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
print(f"  {len(case_dirs)} case dirs")

register_tokens_by_level = {lvl: {} for lvl in range(N_LEVELS)}
case_names, labels, dices = [], [], []

for case_dir in case_dirs:
    case_name = case_dir.name
    meta_file = case_dir / "metadata.npz"
    if not meta_file.exists():
        continue
    meta = np.load(meta_file, allow_pickle=True)

    for lvl in range(N_LEVELS):
        reg_file = case_dir / f"level{lvl}_register_tokens.npy"
        if reg_file.exists():
            register_tokens_by_level[lvl][case_name] = np.load(reg_file)[0]  # (D,)

    case_names.append(case_name)
    labels.append(str(meta["label_id"]))
    dices.append(float(meta["dice"]))

case_names  = np.array(case_names)
labels_arr  = np.array(labels)
dices_arr   = np.array(dices)
image_ids   = np.array([n.split("_")[0] for n in case_names])

all_tokens_by_level = {}
for lvl in range(N_LEVELS):
    tok = register_tokens_by_level[lvl]
    all_tokens_by_level[lvl] = np.stack([tok[n] for n in case_names], axis=0)  # (N, D)

print(f"  Loaded {len(case_names)} entries, levels: "
      f"{[all_tokens_by_level[l].shape for l in range(N_LEVELS)]}")

# ── Load meta & build splits ───────────────────────────────────────────────────
meta_df = pd.read_csv(META_CSV, sep=";").set_index("image_id")

val_ids  = set(meta_df[meta_df["split"] == "val"].index)
test_ids = set(meta_df[meta_df["split"] == "test"].index)

val_mask  = np.array([iid in val_ids  for iid in image_ids])
test_mask = np.array([iid in test_ids for iid in image_ids])
print(f"  Val entries: {val_mask.sum()}, Test entries: {test_mask.sum()}")

# ── Coarsen study_type to body region ─────────────────────────────────────────
def coarsen_study_type(s):
    if pd.isna(s): return np.nan
    s = s.lower()
    if "head"   in s or "brain" in s: return "head"
    if "neck"   in s and "thorax" not in s: return "neck"
    if "thorax" in s and "abdomen" not in s: return "thorax"
    if "abdomen" in s and "thorax" not in s: return "abdomen"
    if "pelvis" in s and "abdomen" not in s: return "pelvis"
    if "angio"  in s: return "angiography"
    if "neck"   in s or "thorax" in s or "abdomen" in s: return "multi_region"
    return "other"

meta_df["body_region"] = meta_df["study_type"].apply(coarsen_study_type)

# ── Task definitions ───────────────────────────────────────────────────────────
# Each task: (name, source, type)
# source="case" → per-entry label from case_names/labels_arr
# source="meta:col" → lookup image_id in meta_df[col]

TASKS_CLF = [
    ("anatomical_label",    "case:label",           50),
    ("gender",              "meta:gender",           2),
    ("pathology",           "meta:pathology",        8),
    ("manufacturer",        "meta:manufacturer",     3),
    ("scanner_model",       "meta:scanner_model",   20),
    ("institute",           "meta:institute",        10),
    ("body_region",         "meta:body_region",      7),
    ("pathology_location",  "meta:pathology_location", 28),
    ("kvp_bin",             "meta:kvp",              9),   # will bin kvp
]
TASKS_REG = [
    ("age",  "meta:age"),
    ("kvp",  "meta:kvp"),
]

def get_labels(source):
    """Return (y_array_or_None, valid_mask) for the full case list."""
    if source == "case:label":
        return labels_arr.copy(), np.ones(len(case_names), dtype=bool)

    col = source.split(":", 1)[1]
    vals = np.array([
        meta_df.loc[iid, col] if iid in meta_df.index else np.nan
        for iid in image_ids
    ])
    valid = ~pd.isna(pd.Series(vals)).values
    return vals, valid

# ── Run experiments ────────────────────────────────────────────────────────────
rows = []

for lvl in range(N_LEVELS):
    tokens = all_tokens_by_level[lvl]
    scaler = StandardScaler().fit(tokens[val_mask])
    tokens_s = scaler.transform(tokens)

    # --- Classification tasks ---
    for task_name, source, _ in TASKS_CLF:
        y_raw, valid = get_labels(source)
        mask = valid & (val_mask | test_mask)

        # Bin kvp for kvp_bin task
        if task_name == "kvp_bin":
            y_raw = np.where(valid, pd.cut(
                pd.to_numeric(y_raw, errors="coerce"),
                bins=[0, 75, 85, 95, 105, 115, 125, 135, 145, 999],
                labels=["70", "80", "90", "100", "110", "120", "130", "140", "150+"]
            ).astype(str), np.nan)
            valid = ~(pd.Series(y_raw) == "nan").values
            mask  = valid & (val_mask | test_mask)

        tr_mask = val_mask  & mask
        te_mask = test_mask & mask

        if tr_mask.sum() < 5 or te_mask.sum() < 5:
            print(f"  SKIP {task_name} lvl{lvl}: too few samples ({tr_mask.sum()} train)")
            continue

        le = LabelEncoder()
        le.fit(y_raw[mask])
        y_tr  = le.transform(y_raw[tr_mask])
        y_te  = le.transform(y_raw[te_mask])

        if len(np.unique(y_tr)) < 2:
            print(f"  SKIP {task_name} lvl{lvl}: only 1 class in training data")
            continue

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 multi_class="multinomial", n_jobs=-1)
        clf.fit(tokens_s[tr_mask], y_tr)

        preds = clf.predict(tokens_s[te_mask])
        acc   = accuracy_score(y_te, preds)
        bacc  = balanced_accuracy_score(y_te, preds)
        n_cls = len(le.classes_)
        chance = 1.0 / n_cls

        rows.append(dict(
            task=task_name, level=lvl, task_type="classification",
            metric_primary="balanced_acc", value_primary=bacc,
            metric_secondary="accuracy", value_secondary=acc,
            n_classes=n_cls, chance_level=chance,
            n_train=tr_mask.sum(), n_test=te_mask.sum()
        ))
        print(f"  {task_name:25s} lvl{lvl}  bacc={bacc:.3f}  acc={acc:.3f}"
              f"  (chance={chance:.3f}, n_cls={n_cls})")

    # --- Regression tasks ---
    for task_name, source in TASKS_REG:
        y_raw, valid = get_labels(source)
        y_num = pd.to_numeric(pd.Series(y_raw), errors="coerce").values
        valid  = valid & ~np.isnan(y_num)

        tr_mask = val_mask  & valid
        te_mask = test_mask & valid

        if tr_mask.sum() < 5 or te_mask.sum() < 5:
            print(f"  SKIP {task_name} lvl{lvl}: too few samples")
            continue

        reg = Ridge(alpha=1.0)
        reg.fit(tokens_s[tr_mask], y_num[tr_mask])

        preds = reg.predict(tokens_s[te_mask])
        r2    = r2_score(y_num[te_mask], preds)
        mae   = mean_absolute_error(y_num[te_mask], preds)
        r, _  = pearsonr(y_num[te_mask], preds)

        rows.append(dict(
            task=task_name, level=lvl, task_type="regression",
            metric_primary="pearson_r", value_primary=r,
            metric_secondary="r2", value_secondary=r2,
            n_classes=np.nan, chance_level=0.0,
            n_train=tr_mask.sum(), n_test=te_mask.sum()
        ))
        print(f"  {task_name:25s} lvl{lvl}  r={r:.3f}  r2={r2:.3f}  mae={mae:.2f}")

# ── Save ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} results to {OUT_CSV}")
print(df.pivot_table(index="task", columns="level", values="value_primary").round(3))
