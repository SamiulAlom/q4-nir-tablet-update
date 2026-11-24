# %% [markdown]
# # RUNTIME: local PC, NVIDIA RTX 3060 GPU (the human's order of 2026-11-19; plan section 10)
#
# Project: `qpain/paper-04` - Stage: `02 source-only tuning` - Colab units: 0
#
# Chooses, for each model family, the preprocessing and settings by cross-validation on source-condition tablets only
# (analysis_plan.md section 4 and its clarifications in section 10), and trains the source CNNs. Five source sets are
# tuned separately: the shootout training tablets on instrument 1 and on instrument 2 (the two directions of
# Scenario I), the Copenhagen source scales (Scenario II), and each single source scale (for the pseudo-transfer of
# plan section 5, whose choice of update settings is made in notebook 03).
# No new-condition tablet is used by any fit, scaling, early stopping, score or choice in this notebook: the cell
# "rows this notebook may use" builds the allowed row sets from the frozen split file and every fit goes through them.
#
# Run: `jupyter nbconvert --to notebook --execute` with `PAPER_ROOT` set; `P4_SMOKE=1` for the smoke test.
# Source: `notebooks/src/02_tune_source_GPU.py` (edit the source, rebuild with scripts/py-to-ipynb.py).

# %%
# Setup
import os, sys, json, time, shutil, platform, datetime, subprocess
import importlib.metadata as ilmd
os.environ["HF_HUB_OFFLINE"] = "1"                   # the foundation model's weights are a local, verified file; nothing is fetched
ROOT = os.environ.get("PAPER_ROOT", os.path.abspath(".."))
SMOKE = os.environ.get("P4_SMOKE", "0") == "1"       # smoke test: one fold, one preprocessing, one setting per family
SMOKE_TEST = SMOKE
OUT_REAL, CACHE, CKPT = (os.path.join(ROOT, d) for d in ("outputs", "data/cache", "checkpoints"))
# The smoke test never writes to outputs/ or to the real parts and checkpoints, and always starts from nothing
OUT = os.path.join(ROOT, "logs", "smoke-02-out") if SMOKE else OUT_REAL
PARTS = os.path.join(CKPT, "p4_tuning_parts_smoke" if SMOKE else "p4_tuning_parts")
CNN_DIR = os.path.join(CKPT, "p4_cnn_smoke" if SMOKE else "p4_cnn")
if SMOKE:
    for d in (OUT, PARTS, CNN_DIR):
        shutil.rmtree(d, ignore_errors=True)
for d in (OUT, PARTS, CNN_DIR):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
NOTEBOOK = "02_tune_source_GPU"

import numpy as np, pandas as pd, torch
import p4_common as P, p4_models as M

SEED = 20261122
CNN_SEEDS = [0, 1, 2]                   # three training seeds, used in tuning and for the final source CNNs
SPLITS_SHA_AUDITED = "034d226f695684bfd80be90445b2d9ba6b6def65ae71660c49acb3bc4dfa7a35"   # notebook 01, run local-run-01
FORCE_CPU = os.environ.get("P4_FORCE_CPU", "0") == "1"      # CPU smoke test while another project holds the GPU
assert SMOKE or not FORCE_CPU, "the full run must use the GPU: P4_FORCE_CPU is for the smoke test only"
DEVICE = "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"
assert DEVICE == "cuda" or SMOKE, "no CUDA device for the full run"
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
VERSIONS = {m: ilmd.version(m) for m in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "torch", "tabpfn")}
VERSIONS.update(python=platform.python_version(), device=DEVICE, torch_cuda=str(torch.version.cuda), cudnn=str(torch.backends.cudnn.version()),
                cudnn_deterministic=str(torch.backends.cudnn.deterministic), cudnn_benchmark=str(torch.backends.cudnn.benchmark),
                p4_common=P.COMMON_VERSION, p4_models=M.MODELS_VERSION)
if DEVICE == "cuda":
    VERSIONS["gpu"] = torch.cuda.get_device_name(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True)
    VERSIONS["gpu_driver"] = smi.stdout.strip() if smi.returncode == 0 else "unknown"
print("Run at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "| SMOKE", SMOKE)
print(VERSIONS)
T0 = time.time()

# %%
# Arrays and frozen splits; the splits must be the audited ones and must belong to these arrays
with open(os.path.join(OUT_REAL, "p4_splits.json"), "rb") as f:
    split_bytes = f.read()
SPLITS_SHA = P.sha256_bytes(split_bytes)
assert SPLITS_SHA == SPLITS_SHA_AUDITED, "p4_splits.json is not the file written and audited by notebook 01"
SPLITS = json.loads(split_bytes)
with np.load(os.path.join(CACHE, "p4_arrays.npz")) as npz:
    A = {k: npz[k] for k in npz.files}
assert P.array_digest(A) == SPLITS["arrays_sha256"], "the cached arrays are not the ones the frozen splits index"
WEIGHTS = M.tabpfn_weights(os.path.join(CKPT, "tabpfn"), download=False)       # read only; hash checked (decision 002)

# Run fingerprint: parts and checkpoints written under another fingerprint are never reused
code_hashes = {f: P.sha256_file(os.path.join(ROOT, "notebooks", f)) for f in ("src/02_tune_source_GPU.py", "p4_models.py", "p4_common.py")}
FINGERPRINT = P.sha256_bytes(json.dumps(dict(
    code=code_hashes, splits=SPLITS_SHA, arrays=SPLITS["arrays_sha256"], seed=SEED, cnn_seeds=CNN_SEEDS, smoke=SMOKE,
    versions=VERSIONS, weights=M.TABPFN_SHA256,
    grids=dict(prep=M.PREPROCESSINGS, pls=M.PLS_MAX_COMPONENTS, xgb=M.XGB_GRID, xgb_fixed=[M.XGB_MAX_TREES, M.XGB_PATIENCE, M.XGB_COLSAMPLE],
               cnn=M.CNN_GRID, cnn_fixed=[M.CNN_MAX_EPOCHS, M.CNN_PATIENCE, M.CNN_BATCH, M.CNN_WEIGHT_DECAY, M.CNN_AUG, M.INNER_FRACTION],
               tabpfn=M.TABPFN_ESTIMATORS)), sort_keys=True).encode("utf-8"))
print("splits sha256", SPLITS_SHA[:16], "| arrays digest ok | weights ok | run fingerprint", FINGERPRINT[:16])

# %%
# Rows this notebook may use: source-condition training tablets only.
# Each source set: spectra, target, group labels (or None) and cross-validation folds as (train, test) positions
# inside the set, with a repeat and fold number. New-condition rows are collected only to assert that no source set
# touches them.
S1, S2, PT = SPLITS["scenario_I"], SPLITS["scenario_II"], SPLITS["pseudo_transfer"]
s1_train = np.array(S1["train"])
s1_new = set(i for r in S1["repetitions"] for i in r["pool_order"] + r["eval"])
assert len(set(s1_train.tolist())) == len(s1_train) == 195 and not (set(s1_train.tolist()) & s1_new)
assert set(A["s_part"][s1_train].tolist()) == {"calibrate", "validate"}
c_group, c_scale = A["c_group"], A["c_scale"]


def folds_from_rows(train_rows, row_folds, repeat):
    """Folds given as lists of row numbers -> (repeat, fold, train positions, test positions); must partition train_rows."""
    pos = {int(r): i for i, r in enumerate(train_rows)}
    flat = [int(r) for fold in row_folds for r in fold]
    assert len(flat) == len(set(flat)) == len(train_rows) and set(flat) == set(pos), "folds must cover every source row exactly once"
    out = []
    for f, fold in enumerate(row_folds):
        te = np.array(sorted(pos[int(r)] for r in fold))
        out.append((repeat, f, np.setdiff1d(np.arange(len(train_rows)), te), te))
    return out


def folds_from_groups(train_rows, group_folds):
    g = c_group[train_rows]
    flat = [x for fold in group_folds for x in fold]
    assert len(flat) == len(set(flat)) and set(flat) == set(g.tolist()), "group folds must cover every source group exactly once"
    out = []
    for f, fold in enumerate(group_folds):
        te = np.flatnonzero(np.isin(g, fold))
        assert te.size == 10 * len(fold), "a fold must hold its groups whole"
        out.append((0, f, np.setdiff1d(np.arange(len(train_rows)), te), te))
    return out


SETS = {}
assert len(S1["cv_folds"]) == 3 and all(len(rpt) == 5 for rpt in S1["cv_folds"])
s1_folds = [f for r, rpt in enumerate(S1["cv_folds"]) for f in folds_from_rows(s1_train, rpt, r)]       # 3 repeats x 5 folds
for inst in ("1", "2"):
    SETS[f"s1_inst{inst}"] = dict(x=A[f"s_x{inst}"][s1_train], y=A["s_assay"][s1_train], groups=None, folds=s1_folds,
                                  rows=s1_train, dataset="shootout", n_folds=15)
s2_train = np.array(S2["train"])
assert set(c_scale[s2_train].tolist()) == set(S2["source_scales"]) and S2["new_scale"] not in S2["source_scales"]
SETS["s2_source"] = dict(x=A["c_x"][s2_train], y=A["c_content"][s2_train], groups=c_group[s2_train],
                         folds=folds_from_groups(s2_train, S2["cv_group_folds"]), rows=s2_train, dataset="copenhagen", n_folds=5)
for name, key, scale, n_folds in (("pseudo_scale0", "scale0_to_scale1", 0, 7), ("pseudo_scale1", "scale1_to_scale0", 1, 4)):
    rows = np.array(PT[key]["train"])
    assert set(c_scale[rows].tolist()) == {scale} and len(set(rows.tolist())) == len(rows)
    SETS[name] = dict(x=A["c_x"][rows], y=A["c_content"][rows], groups=c_group[rows],
                      folds=folds_from_groups(rows, PT[key]["cv_group_folds"]), rows=rows, dataset="copenhagen", n_folds=n_folds)
for name, s in SETS.items():
    assert len(s["folds"]) == s["n_folds"], f"{name}: unexpected number of folds"
    for _, _, tr, te in s["folds"]:
        assert not (set(tr.tolist()) & set(te.tolist())) and len(tr) + len(te) == len(s["rows"])
        if s["groups"] is not None:
            assert not (set(s["groups"][tr].tolist()) & set(s["groups"][te].tolist())), f"{name}: a group is on both sides of a fold"
    if s["dataset"] == "copenhagen":
        assert S2["new_scale"] not in set(c_scale[s["rows"]].tolist()), f"{name}: new-condition scale in a source set"
    print(name, "| tablets", len(s["rows"]), "| folds", len(s["folds"]), "| groups", None if s["groups"] is None else len(set(s["groups"].tolist())))

# %%
# Tuning. One part file per (source set, family), bound to the run fingerprint, so a stopped run resumes and a part
# from other code, settings, splits, seeds, packages or device is rejected. Every fold's score is kept.
PREPS = M.PREPROCESSINGS[:1] if SMOKE else M.PREPROCESSINGS


def features(s, prep, family):
    x = M.preprocess(s["x"], prep)           # per spectrum: nothing is fitted across rows
    return M.bin_pairs(x) if (family == "tabpfn" and s["dataset"] == "shootout") else x      # plan section 4


def mean_se(scores):
    scores = np.asarray(scores, dtype=np.float64)
    M.require_finite(scores, "fold scores")
    return float(scores.mean()), (float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0)


def tune(name, family):
    """Returns (candidate rows, fold rows). A candidate is one preprocessing with one setting of the family's grid."""
    s = SETS[name]
    all_folds = s["folds"][:1] if SMOKE else s["folds"]
    folds = [(tr, te) for _, _, tr, te in all_folds]
    ids = [(r, f) for r, f, _, _ in all_folds]
    cands, fold_rows = [], []

    def add(prep, setting, scores, seed=-1, extra=None, extra_name=""):
        for i, (r, f) in enumerate(ids):
            fold_rows.append(dict(source_set=name, family=family, preprocessing=prep, setting=json.dumps(setting, sort_keys=True), seed=seed,
                                  repeat=r, fold=f, rmse=float(scores[i]), kept=int(extra[i]) if extra is not None else -1, kept_what=extra_name))

    for prep in PREPS:
        x, y = features(s, prep, family), s["y"]
        if family == "pls":
            table = M.pls_cv(x, y, folds)                                   # folds x components
            if len(folds) > 1:
                k, mean, se = M.one_se_choice(table)
            else:                                                           # smoke test: one fold, no standard error
                k, mean, se = int(np.argmin(table[0])), table[0], np.zeros(table.shape[1])
            for c in range(table.shape[1]):
                add(prep, dict(n_components=c + 1), table[:, c])
            cands.append(dict(preprocessing=prep, setting=dict(n_components=k + 1), cv_rmse=float(mean[k]), cv_se=float(se[k])))
        elif family == "xgb":
            for params in (M.XGB_GRID[:1] if SMOKE else M.XGB_GRID):
                sc, trees = M.xgb_cv(x, y, folds, params, s["groups"], SEED)
                add(prep, params, sc, extra=trees, extra_name="trees")
                m, se = mean_se(sc)
                cands.append(dict(preprocessing=prep, setting=dict(**params, n_trees=int(np.median(trees))), cv_rmse=m, cv_se=se))
        elif family == "cnn":
            for params in (M.CNN_GRID[:1] if SMOKE else M.CNN_GRID):
                per_seed, per_seed_ep = [], []
                for seed in (CNN_SEEDS[:1] if SMOKE else CNN_SEEDS):         # same inner splits for every seed
                    sc, ep = M.cnn_cv(x, y, folds, params["lr"], s["groups"], SEED, seed, DEVICE)
                    add(prep, params, sc, seed=seed, extra=ep, extra_name="epochs")
                    per_seed.append(sc)
                    per_seed_ep.append(ep)
                fold_mean = np.mean(per_seed, axis=0)                       # per fold, the mean RMSE over seeds
                m, se = mean_se(fold_mean)
                cands.append(dict(preprocessing=prep, setting=dict(**params, epochs=int(np.median(np.concatenate(per_seed_ep)))),
                                  cv_rmse=m, cv_se=se, cv_rmse_by_seed=[float(np.mean(v)) for v in per_seed]))
        elif family == "tabpfn":
            sc = M.tabpfn_cv(x, y, folds, WEIGHTS, SEED, DEVICE)
            add(prep, dict(n_estimators=M.TABPFN_ESTIMATORS), sc, seed=SEED)
            m, se = mean_se(sc)
            cands.append(dict(preprocessing=prep, setting=dict(n_estimators=M.TABPFN_ESTIMATORS), cv_rmse=m, cv_se=se))
    for order, r in enumerate(cands):
        r.update(source_set=name, family=family, order=order, folds=len(folds), tablets=int(len(s["rows"])))
    return cands, fold_rows


ALL, FOLD_ROWS, TIMES = [], [], []
for name in SETS:
    for family in M.FAMILIES:
        part = os.path.join(PARTS, f"{name}__{family}.json")
        saved = None
        if os.path.exists(part):
            with open(part, encoding="utf-8") as f:
                saved = json.load(f)
            assert saved["fingerprint"] == FINGERPRINT and saved["source_set"] == name and saved["family"] == family, \
                f"{part} was written by another run (code, settings, splits, seeds, packages or device): delete the parts folder"
        if saved is None:
            t = time.time()
            cands, fold_rows = tune(name, family)
            saved = dict(fingerprint=FINGERPRINT, source_set=name, family=family, seconds=time.time() - t, rows=cands, fold_rows=fold_rows)
            with open(part + ".tmp", "w", encoding="utf-8") as f:
                json.dump(saved, f)
            os.replace(part + ".tmp", part)
        M.require_finite([r["cv_rmse"] for r in saved["rows"]] + [r["rmse"] for r in saved["fold_rows"]], f"tuning part {name} {family}")
        assert all(r["tablets"] == len(SETS[name]["rows"]) for r in saved["rows"])
        ALL += saved["rows"]
        FOLD_ROWS += saved["fold_rows"]
        TIMES.append(dict(source_set=name, family=family, seconds=round(saved["seconds"], 1)))
        print(f"{name:14s} {family:7s} {saved['seconds']:7.1f} s | best cv rmse {min(r['cv_rmse'] for r in saved['rows']):.4f}", flush=True)

# %%
# Chosen settings per source set and family (lowest mean cross-validated RMSE; ties by the fixed order).
# The chosen row's cross-validation RMSE is the score that selected it, so it is optimistic; it is labelled as such
# wherever it is used as a benchmark (plan section 10).
CHOSEN = {}
for name in SETS:
    CHOSEN[name] = {}
    for family in M.FAMILIES:
        r = M.choose([r for r in ALL if r["source_set"] == name], family)
        CHOSEN[name][family] = dict(preprocessing=r["preprocessing"], setting=r["setting"], cv_rmse_selected=r["cv_rmse"], cv_se=r["cv_se"],
                                    candidates_searched=sum(1 for q in ALL if q["source_set"] == name and q["family"] == family),
                                    **({"cv_rmse_by_seed": r["cv_rmse_by_seed"]} if "cv_rmse_by_seed" in r else {}))
cv_df = pd.DataFrame([dict(source_set=r["source_set"], family=r["family"], preprocessing=r["preprocessing"],
                           setting=json.dumps(r["setting"], sort_keys=True), cv_rmse=r["cv_rmse"], cv_se=r["cv_se"], folds=r["folds"],
                           tablets=r["tablets"],
                           chosen=(CHOSEN[r["source_set"]][r["family"]]["preprocessing"] == r["preprocessing"]
                                   and CHOSEN[r["source_set"]][r["family"]]["setting"] == r["setting"])) for r in ALL])
assert (cv_df.groupby(["source_set", "family"])["chosen"].sum() == 1).all(), "exactly one chosen row per source set and family"
fold_df = pd.DataFrame(FOLD_ROWS)
print(cv_df[cv_df["chosen"]].drop(columns=["chosen", "folds"]).round(4).to_string(index=False))

# %%
# Source CNNs: three seeds per source set, trained on all its tablets for the chosen number of epochs (no early
# stopping here, so no tablet is held back). Saved as plain tensors (loaded with weights_only=True) with a sidecar
# file that carries the run fingerprint, the settings and the checkpoint's hash.
ckpt_rows = []
for name in SETS:
    c = CHOSEN[name]["cnn"]
    x, y = features(SETS[name], c["preprocessing"], "cnn"), SETS[name]["y"]
    for seed in (CNN_SEEDS[:1] if SMOKE else CNN_SEEDS):
        path = os.path.join(CNN_DIR, f"{name}__seed{seed}.pt")
        side = path + ".json"
        if os.path.exists(path) and os.path.exists(side):
            with open(side, encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["fingerprint"] == FINGERPRINT and meta["sha256"] == P.sha256_file(path) and meta["source_set"] == name \
                and meta["seed"] == seed and meta["preprocessing"] == c["preprocessing"] and meta["setting"] == c["setting"], \
                f"{path} was written by another run or was changed: delete the CNN folder"
        else:
            model, scaler, _ = M.cnn_train(x, y, c["setting"]["lr"], SEED + 100 + seed, DEVICE, epochs=max(1, c["setting"]["epochs"]))
            fitted = M.rmse(y, M.cnn_predict(model, scaler, x, DEVICE))
            torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, x_mean=torch.tensor(scaler.x_mean),
                            x_sd=torch.tensor(scaler.x_sd), y_mean=torch.tensor(scaler.y_mean), y_sd=torch.tensor(scaler.y_sd)), path + ".tmp")
            os.replace(path + ".tmp", path)
            meta = dict(fingerprint=FINGERPRINT, sha256=P.sha256_file(path), source_set=name, seed=seed, preprocessing=c["preprocessing"],
                        setting=c["setting"], tablets=int(len(y)), fitted_rmse=fitted)
            with open(side + ".tmp", "w", encoding="utf-8") as f:
                json.dump(meta, f)
            os.replace(side + ".tmp", side)
        # reload check: the saved tensors give back the same predictions structure
        blob = torch.load(path, map_location="cpu", weights_only=True)
        assert set(blob) == {"state", "x_mean", "x_sd", "y_mean", "y_sd"} and blob["x_mean"].shape[0] == x.shape[1]
        ckpt_rows.append(dict(source_set=name, seed=seed, file=os.path.relpath(path, ROOT).replace("\\", "/"), sha256=meta["sha256"],
                              epochs=c["setting"]["epochs"], rmse_on_own_training_tablets=round(float(meta["fitted_rmse"]), 6)))
print(pd.DataFrame(ckpt_rows).drop(columns=["sha256"]).to_string(index=False))

# %%
# Write outputs: every file to a temporary name first; all are renamed only after all were written; a completion
# marker with their hashes is written last, and notebook 03 refuses outputs without a matching marker.
chosen_doc = dict(fingerprint=FINGERPRINT, splits_sha256=SPLITS_SHA, arrays_sha256=SPLITS["arrays_sha256"], code_sha256=code_hashes, seed=SEED,
                  cnn_seeds=CNN_SEEDS, smoke=SMOKE,
                  tabpfn=dict(repo=M.TABPFN_REPO, commit=M.TABPFN_COMMIT, file=M.TABPFN_FILE, sha256=M.TABPFN_SHA256,
                              n_estimators=M.TABPFN_ESTIMATORS, package=VERSIONS["tabpfn"], model_version="v2", seed=SEED),
                  chosen=CHOSEN, cnn_checkpoints=ckpt_rows)
texts = {
    "p4_tuning_cv.csv": cv_df.to_csv(index=False, lineterminator="\n"),
    "p4_tuning_folds.csv": fold_df.to_csv(index=False, lineterminator="\n"),
    "p4_cnn_checkpoints.csv": pd.DataFrame(ckpt_rows).to_csv(index=False, lineterminator="\n"),
    "p4_tuning_seconds.csv": pd.DataFrame(TIMES).to_csv(index=False, lineterminator="\n"),
    "p4_versions_02.csv": pd.DataFrame([dict(item=k, value=v) for k, v in VERSIONS.items()]).to_csv(index=False, lineterminator="\n"),
    "p4_tuning_chosen.json": json.dumps(chosen_doc, sort_keys=True, indent=1),
}
described = {
    "p4_tuning_cv.csv": ("tuning", "Source-only cross-validation: mean RMSE of every candidate (preprocessing and setting) per source set and family, full precision, with the chosen row marked"),
    "p4_tuning_folds.csv": ("tuning", "Source-only cross-validation, every fold: RMSE per candidate, repeat, fold and seed, with the trees or epochs kept by early stopping; all PLS component counts"),
    "p4_cnn_checkpoints.csv": ("source CNNs", "Source CNN checkpoints (three seeds per source set): file, hash, epochs, RMSE on their own training tablets"),
    "p4_tuning_seconds.csv": ("tuning", "Wall-clock seconds of the tuning per source set and family on this PC"),
    "p4_versions_02.csv": ("versions", "Package versions, GPU, driver, CUDA and cuDNN settings of the notebook 02 run"),
    "p4_tuning_chosen.json": ("tuning", "Chosen preprocessing and settings per source set and family, the number of candidates searched, the pinned foundation-model weights, and hashes of code, splits, arrays and CNN checkpoints"),
}
for name, text in texts.items():
    with open(os.path.join(OUT, name + ".tmp"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
for name in texts:
    os.replace(os.path.join(OUT, name + ".tmp"), os.path.join(OUT, name))
marker = dict(notebook=NOTEBOOK, fingerprint=FINGERPRINT, smoke=SMOKE, files={n: P.sha256_file(os.path.join(OUT, n)) for n in texts})
with open(os.path.join(OUT, "p4_complete_02.json.tmp"), "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(marker, sort_keys=True, indent=1))
os.replace(os.path.join(OUT, "p4_complete_02.json.tmp"), os.path.join(OUT, "p4_complete_02.json"))
if not SMOKE:
    rows = [(n, NOTEBOOK, described[n][0], described[n][1]) for n in texts]
    rows.append(("p4_complete_02.json", NOTEBOOK, "outputs", "Completion marker of notebook 02: run fingerprint and the hash of each of its output files"))
    man_path = os.path.join(OUT, "MANIFEST.md")
    HEADER = ["| file | notebook | cell | description |", "|---|---|---|---|"]
    with open(man_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    other = [l for l in lines if l.startswith("|") and l not in HEADER and f"| {NOTEBOOK} |" not in l]
    with open(man_path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(HEADER + other + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]) + "\n")
    os.replace(man_path + ".tmp", man_path)
print("DONE", NOTEBOOK, "| SMOKE", SMOKE, "| minutes", round((time.time() - T0) / 60, 1))
