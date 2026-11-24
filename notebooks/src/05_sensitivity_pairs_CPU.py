# %% [markdown]
# # RUNTIME: local PC, CPU (the human's order of 2026-11-19; plan section 10)
#
# Project: `qpain/paper-04` - Stage: `05 sensitivity: pair-averaged features` - Colab units: 0
#
# Planned sensitivity analysis of analysis_plan.md section 8, first item, run after the main results were seen
# (deviation 4, Jon's ruling in decisions/013): PLS and boosted trees tuned and fitted on the 325 pair-averaged
# shootout features (the same averaging TabPFN received; each family chooses its own preprocessing on source data),
# to see what the averaging costs. Scenario I, instrument 1 to 2 only (the
# direction of the primary comparison). Same frozen splits, same source-only tuning rule and grids as notebook 02,
# same 100 repetitions and the same U1 rule as notebook 03. TabPFN's predictions are read from notebook 03's outputs
# (not recomputed). No new-condition tablet is used by any tuning or fit; U1 uses the update tablets only.
#
# Run: `bash notebooks/run_nb.sh 05_sensitivity_pairs_CPU <log>`; `P4_SMOKE=1` for the smoke test (3 repetitions,
# one fold, one preprocessing, written outside outputs/).
# Source: `notebooks/src/05_sensitivity_pairs_CPU.py` (edit the source, rebuild with scripts/py-to-ipynb.py).

# %%
# Setup
import os, sys, json, platform, datetime
import importlib.metadata as ilmd
ROOT = os.environ.get("PAPER_ROOT", os.path.abspath(".."))
SMOKE = os.environ.get("P4_SMOKE", "0") == "1"
OUT_REAL = os.path.join(ROOT, "outputs")
OUT = os.path.join(ROOT, "logs", "smoke-05-out") if SMOKE else OUT_REAL
CACHE = os.path.join(ROOT, "data", "cache")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
NOTEBOOK = "05_sensitivity_pairs_CPU"

import numpy as np, pandas as pd
import p4_common as P, p4_models as M, p4_updates as U, p4_stats as S

SEED = 20261122                                       # notebook 02's seed (boosted-tree fits)
SPLITS_SHA_AUDITED = "034d226f695684bfd80be90445b2d9ba6b6def65ae71660c49acb3bc4dfa7a35"
VERSIONS = {m: ilmd.version(m) for m in ("numpy", "pandas", "scikit-learn", "xgboost")}
VERSIONS.update(python=platform.python_version(), p4_common=P.COMMON_VERSION, p4_models=M.MODELS_VERSION, p4_stats=S.STATS_VERSION)
print("Run at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "| SMOKE", SMOKE, VERSIONS)

# %%
# Inputs: audited splits, arrays bound to them, and notebook 03's verified predictions
with open(os.path.join(OUT_REAL, "p4_splits.json"), "rb") as f:
    split_bytes = f.read()
assert P.sha256_bytes(split_bytes) == SPLITS_SHA_AUDITED, "p4_splits.json is not the audited file"
SPLITS = json.loads(split_bytes)
with np.load(os.path.join(CACHE, "p4_arrays.npz")) as npz:
    A = {k: npz[k] for k in npz.files}
assert P.array_digest(A) == SPLITS["arrays_sha256"], "the cached arrays are not the ones the frozen splits index"
with open(os.path.join(OUT_REAL, "p4_complete_03.json"), encoding="utf-8") as f:
    MARK03 = json.load(f)
assert P.sha256_file(os.path.join(OUT_REAL, "p4_pred_s1_12.npz")) == MARK03["files"]["p4_pred_s1_12.npz"], "p4_pred_s1_12.npz differs from notebook 03's marker"
with np.load(os.path.join(OUT_REAL, "p4_pred_s1_12.npz")) as z:
    PRED03 = {k: z[k] for k in z.files}
LAB03 = [str(l) for l in PRED03["labels"]]

S1 = SPLITS["scenario_I"]
s1_train = np.array(S1["train"])
s1_test = np.flatnonzero(A["s_part"] == "test")
assert len(s1_train) == 195 and len(s1_test) == 460 and not (set(s1_train.tolist()) & set(s1_test.tolist()))
assert set(A["s_part"][s1_train].tolist()) == {"calibrate", "validate"}
GRID = [n for n in S1["grid"] if n > 0]
assert GRID == [5, 10, 20, 40, 80]

# %%
# Source-only tuning on the pair-averaged features (instrument 1 training tablets), notebook 02's rule and grids
x_src_raw, y_src = A["s_x1"][s1_train], A["s_assay"][s1_train].astype(np.float64)
pos = {int(r): i for i, r in enumerate(s1_train)}
FOLDS = []
for rpt in S1["cv_folds"]:
    for fold in rpt:
        te = np.array(sorted(pos[int(r)] for r in fold))
        FOLDS.append((np.setdiff1d(np.arange(len(s1_train)), te), te))
assert len(FOLDS) == 15
folds = FOLDS[:1] if SMOKE else FOLDS
PREPS = M.PREPROCESSINGS[:1] if SMOKE else M.PREPROCESSINGS


def feat325(x_raw, prep):
    x = M.bin_pairs(M.preprocess(x_raw, prep))
    assert x.shape[1] == 325
    return x


rows = []
for prep in PREPS:
    x = feat325(x_src_raw, prep)
    table = M.pls_cv(x, y_src, folds)
    if len(folds) > 1:
        k, mean, se = M.one_se_choice(table)
    else:
        k, mean, se = int(np.argmin(table[0])), table[0], np.zeros(table.shape[1])
    rows.append(dict(family="pls", preprocessing=prep, setting=dict(n_components=k + 1), cv_rmse=float(mean[k]), cv_se=float(se[k])))
    for params in (M.XGB_GRID[:1] if SMOKE else M.XGB_GRID):
        sc, trees = M.xgb_cv(x, y_src, folds, params, None, SEED)
        sc = np.asarray(sc, dtype=np.float64)
        rows.append(dict(family="xgb", preprocessing=prep, setting=dict(**params, n_trees=int(np.median(trees))), cv_rmse=float(sc.mean()),
                         cv_se=float(sc.std(ddof=1) / np.sqrt(len(sc))) if len(sc) > 1 else 0.0))
for order, r in enumerate(rows):
    r["order"] = order
CHOSEN = {fam: M.choose(rows, fam) for fam in ("pls", "xgb")}
for fam, r in CHOSEN.items():
    print(fam, "325 features | chosen", r["preprocessing"], r["setting"], "| CV RMSE", round(r["cv_rmse"], 4))

# %%
# Source models, predictions for the new-instrument test part (U0), and U1 per repetition and n (notebook 03's rule)
c = CHOSEN
x_pls = feat325(x_src_raw, c["pls"]["preprocessing"])
x_xgb = feat325(x_src_raw, c["xgb"]["preprocessing"])
pls = M.pls_fit(x_pls, y_src, c["pls"]["setting"]["n_components"])
xgb = M.xgb_fit(x_xgb, y_src, c["xgb"]["setting"], int(c["xgb"]["setting"]["n_trees"]), SEED)
x_new = A["s_x2"][s1_test]
base = {"pls325": M.pls_predict(pls, feat325(x_new, c["pls"]["preprocessing"])),
        "xgb325": np.asarray(xgb.predict(feat325(x_new, c["xgb"]["preprocessing"])), dtype=np.float64)}
for k, v in base.items():
    M.require_finite(v, k)
posn = {int(r): i for i, r in enumerate(s1_test)}
y_new = A["s_assay"][s1_test].astype(np.float64)

reps = S1["repetitions"][:3] if SMOKE else S1["repetitions"][:100]
assert SMOKE or len(reps) == len(PRED03["rep"]) == 100
labels, preds, ytrue = None, [], []
for i, rep in enumerate(reps):
    assert int(PRED03["rep"][i]) == rep["rep"]
    ev_rows = np.asarray(rep["eval"])
    assert len(rep["pool_order"]) == 100 and len(ev_rows) == 360 and len(set(rep["pool_order"])) == 100 and set(rep["pool_order"]) | set(ev_rows.tolist()) == set(s1_test.tolist()) and not (set(rep["pool_order"]) & set(ev_rows.tolist()))
    assert np.array_equal(ev_rows, PRED03["eval_rows"][i]), "evaluation tablets differ from notebook 03's"
    ev = np.array([posn[int(r)] for r in ev_rows])
    out = {}
    for fam, p in base.items():
        out[f"{fam}|U0|-|0"] = p[ev]
        for n in GRID:
            upd = np.array([posn[int(r)] for r in rep["pool_order"][:n]])
            assert not (set(upd.tolist()) & set(ev.tolist()))
            a, b = U.slope_bias(p[upd], y_new[upd])
            out[f"{fam}|U1|-|{n}"] = a + b * p[ev]
    for lab in ("tabpfn|U1|-|10", "pls|U1|-|10", "xgb|U1|-|10", "tabpfn|U0|-|0", "pls|U0|-|0", "xgb|U0|-|0"):
        out[lab] = PRED03["pred"][i, LAB03.index(lab)].astype(np.float64)
    labels = labels or sorted(out)
    preds.append(np.stack([np.float32(out[l]) for l in labels]))         # stored precision, as notebook 03
    assert np.array_equal(y_new[ev], PRED03["y_true"][i].astype(np.float64)), "reference values differ from notebook 03's"
    ytrue.append(PRED03["y_true"][i].astype(np.float64))
sc = S.scores(np.stack(preds), np.stack(ytrue), labels)

# %%
# Paired ratios at n = 10 (primary comparison's n) and RMSE summaries at every n
def rm(lab):
    return sc[sc["label"] == lab].sort_values("rep_index")["rmse"].to_numpy()


PAIRS = [("tabpfn|U1|-|10", "pls325|U1|-|10", "TabPFN U1 vs PLS-325 U1"), ("tabpfn|U1|-|10", "pls|U1|-|10", "TabPFN U1 vs PLS U1 (primary, as notebook 04)"),
         ("xgb325|U1|-|10", "pls325|U1|-|10", "Boosted trees-325 U1 vs PLS-325 U1"), ("tabpfn|U1|-|10", "xgb325|U1|-|10", "TabPFN U1 vs boosted trees-325 U1"),
         ("pls325|U1|-|10", "pls|U1|-|10", "PLS-325 U1 vs PLS (650) U1"), ("xgb325|U1|-|10", "xgb|U1|-|10", "Boosted trees-325 U1 vs boosted trees (650) U1"),
         ("pls325|U0|-|0", "pls|U0|-|0", "PLS-325 U0 vs PLS (650) U0"), ("xgb325|U0|-|0", "xgb|U0|-|0", "Boosted trees-325 U0 vs boosted trees (650) U0")]
RATIOS = pd.DataFrame([dict(comparison=t, numerator=a, denominator=b, n=int(a.split("|")[-1]), **S.paired_ratio(rm(a), rm(b))) for a, b, t in PAIRS])
RMSE = pd.DataFrame([dict(label=l, **S.describe(rm(l))) for l in labels])
TUNING = pd.DataFrame([dict(family=r["family"], preprocessing=r["preprocessing"], setting=json.dumps(r["setting"], sort_keys=True), cv_rmse=r["cv_rmse"], cv_se=r["cv_se"],
                            chosen=(r is CHOSEN[r["family"]])) for r in rows])
print(RATIOS.round(3).to_string(index=False))
print(RMSE.round(3).to_string(index=False))

# %%
# Write outputs, manifest rows (full run only) and the completion marker
if os.path.exists(os.path.join(OUT, "p4_complete_05.json")):
    os.remove(os.path.join(OUT, "p4_complete_05.json"))
TABLES = {"p4_sens_pairs_ratios.csv": (RATIOS, "Sensitivity (plan section 8, first item; run after the results, deviation 4): paired RMSE ratios at n = 10 (U1) and n = 0 (U0), instrument 1 to 2, PLS and boosted trees on the 325 pair-averaged features against TabPFN and the 650-feature models"),
          "p4_sens_pairs_rmse.csv": (RMSE, "Sensitivity: RMSE over the 100 repetitions per method and n, PLS and boosted trees on 325 pair-averaged features (U0, U1), with the notebook 03 methods compared"),
          "p4_sens_pairs_tuning.csv": (TUNING, "Sensitivity: source-only tuning on 325 pair-averaged instrument-1 features (notebook 02's folds, grids and rule); CV RMSE is optimistic")}
for name, (df, _) in TABLES.items():
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(df.to_csv(index=False, lineterminator="\n"))
if not SMOKE:
    man_path = os.path.join(OUT, "MANIFEST.md")
    HEADER = ["| file | notebook | cell | description |", "|---|---|---|---|"]
    with open(man_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    other = [l for l in lines if l.startswith("|") and l not in HEADER and f"| {NOTEBOOK} |" not in l]
    new = [f"| {n} | {NOTEBOOK} | sensitivity | {d} |" for n, (_, d) in TABLES.items()]
    new.append(f"| p4_complete_05.json | {NOTEBOOK} | outputs | Completion marker of notebook 05: input hashes and the hash of each output file |")
    with open(man_path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(HEADER + other + new) + "\n")
    os.replace(man_path + ".tmp", man_path)
marker = dict(notebook=NOTEBOOK, smoke=SMOKE, versions=VERSIONS, inputs=dict(splits=SPLITS_SHA_AUDITED, pred_s1_12=MARK03["files"]["p4_pred_s1_12.npz"]),
              files={n: P.sha256_file(os.path.join(OUT, n)) for n in TABLES})
with open(os.path.join(OUT, "p4_complete_05.json.tmp"), "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(marker, sort_keys=True, indent=1))
os.replace(os.path.join(OUT, "p4_complete_05.json.tmp"), os.path.join(OUT, "p4_complete_05.json"))
print("DONE", NOTEBOOK, "| SMOKE", SMOKE)
