# %% [markdown]
# # RUNTIME: CPU (local PC) - no GPU is used
#
# Project: `qpain/paper-04` - Stage: `01 audit and frozen splits` - Colab units: 0 (runs on this PC, plan section 10)
#
# Checks the hashes of the two downloaded archives, reads the spectra and reference values, gives every tablet one
# identifier, describes the groups, and writes the frozen repetition splits of analysis_plan.md section 3 (and the
# source-only tuning folds and pseudo-transfer splits of sections 4 and 5). No model is fitted and no spectrum is
# related to a reference value. Raw files are only read. Everything is computed and checked first; files are written
# in the last cells, so a run that fails a check leaves the earlier outputs as they were.
#
# Run: `jupyter nbconvert --to notebook --execute` with `PAPER_ROOT` set to the project folder.
# Source: `notebooks/src/01_audit_splits_CPU.py` (edit the source, rebuild with scripts/py-to-ipynb.py).

# %%
# Setup: paths from PAPER_ROOT and the constants fixed by analysis_plan.md
import os, sys, io, json, platform, datetime
import importlib.metadata as ilmd
ROOT = os.environ.get("PAPER_ROOT", os.path.abspath(".."))
RAW, OUT, CACHE = (os.path.join(ROOT, d) for d in ("data/raw", "outputs", "data/cache"))
for d in (OUT, CACHE):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
NOTEBOOK = "01_audit_splits_CPU"

import numpy as np, pandas as pd
from scipy.io import loadmat
import p4_common as P

SEED = 20261122
# Scenario I (plan section 3): 460 new-condition tablets, update pool 100, evaluation 360, 200 repetitions
S1_POOL, S1_REPS, S1_GRID = 100, 200, [0, 5, 10, 20, 40, 80]
S1_FOLDS, S1_FOLD_REPEATS = 5, 3
# Scenario II (plan section 3): source = scale codes 0 and 1, new condition = scale code 2
S2_SOURCE_SCALES, S2_NEW_SCALE, S2_ORDERS, S2_PER_GROUP = (0, 1), 2, 3, [0, 1, 2, 4, 10]
S2_FOLDS = 5
# Pseudo-transfer inside the source condition (plan section 5), used only to choose update settings for Scenario II
PSEUDO_CHOICES_0_TO_1, PSEUDO_PER_GROUP, PSEUDO_FOLDS_SCALE1 = 27, 2, 4      # n = 8 = 2 tablets from each of 4 pool groups
# Pairing check of the two instruments (decision 003, fixed before this notebook was first run)
PAIR_TOP, PAIR_MIN_SHARE, PAIR_CONTROL_FACTOR = 5, 0.5, 5.0

VERSIONS = {m: ilmd.version(m) for m in ("numpy", "pandas", "scipy")}
print("Run at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "| Python", platform.python_version())
print("SEED", SEED, "| common", P.COMMON_VERSION, "|", VERSIONS)
TABLES = []          # (file name, data frame, cell, description): written together in the last cell


def table(df, name, cell, description):
    TABLES.append((name, df, cell, description))
    print("prepared", name, df.shape)


# %%
# Hash check of the two archives, then read the data files from the archives in memory (nothing is written to data/raw)
shoot_zip = os.path.join(RAW, "idrc2002_shootout", "nir_shootout_2002.mat_.zip")
tab_zip = os.path.join(RAW, "ucph_tablets", "Tablets.zip")
hashes = []
for path, expected in ((shoot_zip, P.SHOOTOUT_SHA256), (tab_zip, P.TABLETS_SHA256)):
    got = P.sha256_file(path)
    assert got == expected, f"hash mismatch for {path}: {got}"
    hashes.append((os.path.relpath(path, ROOT).replace("\\", "/"), os.path.getsize(path), got, "archive as downloaded"))

shoot_mat_bytes = P.read_zip_member(open(shoot_zip, "rb").read(), "nir_shootout_2002.mat")
inner_zip = P.read_zip_member(open(tab_zip, "rb").read(), "Tablets_matlab.zip")
tab_mat_bytes = P.read_zip_member(inner_zip, "NIRdata_tablets.MAT")
hashes.append(("nir_shootout_2002.mat (inside the archive)", len(shoot_mat_bytes), P.sha256_bytes(shoot_mat_bytes), "data file read"))
hashes.append(("Tablets_matlab.zip/NIRdata_tablets.MAT (inside the archive)", len(tab_mat_bytes), P.sha256_bytes(tab_mat_bytes), "data file read"))

S = P.load_shootout(loadmat(io.BytesIO(shoot_mat_bytes)))
C = P.load_copenhagen(loadmat(io.BytesIO(tab_mat_bytes)))
for name, arr in (("shootout x1", S["x1"]), ("shootout x2", S["x2"]), ("shootout assay", S["assay"]),
                  ("copenhagen x", C["x"]), ("copenhagen content", C["content"])):
    assert np.isfinite(arr).all(), f"{name}: missing or infinite values"
print("shootout:", S["x1"].shape, "| copenhagen:", C["x"].shape)

# %%
# Shootout: parts and identifiers
part_rows = []
for p in P.SHOOTOUT_PARTS:
    y = S["assay"][S["part"] == p]
    part_rows.append(dict(part=p, tablets=int(y.size), assay_min=y.min(), assay_max=y.max(), assay_mean=y.mean(),
                          assay_sd=y.std(ddof=1)))
part_df = pd.DataFrame(part_rows)
assert part_df["tablets"].tolist() == [155, 40, 460], part_df["tablets"].tolist()
assert S["wavelength_nm"][0] == 600 and S["wavelength_nm"][-1] == 1898 and S["wavelength_nm"].size == 650
table(part_df.round(3), "p4_shootout_parts.csv", "shootout", "Shootout: tablets and assay summary in the organisers' three parts")

# %%
# Shootout: is row i of instrument 1 the same tablet as row i of instrument 2?
# The file has one reference table per part for both instruments and no other identifier, so the pairing rests on
# the publisher's layout and cannot be proved here. This spectra-only check can only support it or flag it.
# For each part: share of rows whose own partner is among the PAIR_TOP most similar rows in both directions, for the
# rows as given and for two wrong pairings made from the same spectra (random shuffle; neighbouring rows swapped).
# Rule fixed before the first run (decision 003): the pairing is "supported" in a part if that share is at least
# PAIR_MIN_SHARE and at least PAIR_CONTROL_FACTOR times the larger of the two control shares. If any part is not
# supported, notebook 02 is not started: the manager decides (plan deviation) whether the methods that need paired
# spectra are dropped. Rows that fail in both directions are listed as suspect; none is removed.
rng = np.random.default_rng(SEED + 1)
align_rows, suspects = [], []
for p in P.SHOOTOUT_PARTS:
    idx = np.flatnonzero(S["part"] == p)
    n = idx.size
    swap = np.arange(n)
    swap[: n - n % 2] = swap[: n - n % 2].reshape(-1, 2)[:, ::-1].ravel()       # (0,1)->(1,0), (2,3)->(3,2), ...
    shares = {}
    for label, order in (("rows as given", np.arange(n)), ("control: shuffled pairing", rng.permutation(n)),
                         ("control: neighbouring rows swapped", swap)):
        rep = P.pairing_report(S["x1"][idx], S["x2"][idx][order])
        both = (rep["rank_ab"] <= PAIR_TOP) & (rep["rank_ba"] <= PAIR_TOP)
        shares[label] = float(both.mean())
        align_rows.append(dict(part=p, pairing=label, tablets=int(n), partner_top1_both_directions=float(((rep["rank_ab"] == 1) & (rep["rank_ba"] == 1)).mean()),
                               partner_in_top5_both_directions=shares[label], median_rank_1_to_2=float(np.median(rep["rank_ab"])),
                               median_rank_2_to_1=float(np.median(rep["rank_ba"])), tied_rows=int(rep["tied"].sum()),
                               invalid_rows=int(rep["invalid"].sum())))
        if label == "rows as given":
            bad = (rep["rank_ab"] > PAIR_TOP) & (rep["rank_ba"] > PAIR_TOP)
            suspects += [dict(tablet=S["tablet"][idx][i], part=p, rank_1_to_2=int(rep["rank_ab"][i]), rank_2_to_1=int(rep["rank_ba"][i]))
                         for i in np.flatnonzero(bad)]
    control = max(shares["control: shuffled pairing"], shares["control: neighbouring rows swapped"])
    ok = shares["rows as given"] >= PAIR_MIN_SHARE and shares["rows as given"] >= PAIR_CONTROL_FACTOR * control
    align_rows.append(dict(part=p, pairing="RULE: supported" if ok else "RULE: NOT supported", tablets=int(n)))
align_df = pd.DataFrame(align_rows)
PAIRING_SUPPORTED = not align_df["pairing"].eq("RULE: NOT supported").any()
table(align_df.round(4), "p4_shootout_alignment.csv", "alignment",
      "Shootout: spectra-only check of the row pairing of the two instruments per part, against two wrong pairings, and the pre-set rule's result")
table(pd.DataFrame(suspects, columns=["tablet", "part", "rank_1_to_2", "rank_2_to_1"]), "p4_shootout_alignment_suspects.csv", "alignment",
      "Shootout: tablets whose partner spectrum is outside the 5 most similar rows in both directions (listed, not removed)")
print(align_df.round(3).to_string(index=False))
print("PAIRING SUPPORTED IN EVERY PART:", PAIRING_SUPPORTED, "| suspect rows:", len(suspects))

# %%
# Copenhagen: groups, types and scales (reference values are summarised, not modelled)
groups = P.group_table(C["group"], C["type"], C["scale"])
g_df = pd.DataFrame(groups, columns=["group", "tablets", "type", "scale"])
assert len(g_df) == 31 and (g_df["tablets"] == 10).all(), "expected 31 groups of 10 tablets"
g_df["content_min"] = [C["content"][C["group"] == g].min() for g in g_df["group"]]
g_df["content_max"] = [C["content"][C["group"] == g].max() for g in g_df["group"]]
g_df["content_mean"] = [C["content"][C["group"] == g].mean() for g in g_df["group"]]
table(g_df.round(4), "p4_copenhagen_groups.csv", "copenhagen", "Copenhagen set: the 31 tablet groups with type, scale code and content range")

src_mask = np.isin(C["scale"], S2_SOURCE_SCALES)
new_mask = C["scale"] == S2_NEW_SCALE
src_max, src_min = C["content"][src_mask].max(), C["content"][src_mask].min()
scale_rows = []
for s in sorted(set(C["scale"].tolist())):
    m = C["scale"] == s
    scale_rows.append(dict(scale=s, groups=int(g_df["scale"].eq(s).sum()), tablets=int(m.sum()),
                           content_min=C["content"][m].min(), content_max=C["content"][m].max(),
                           above_source_max=int((C["content"][m] > src_max).sum()),
                           below_source_min=int((C["content"][m] < src_min).sum())))
table(pd.DataFrame(scale_rows).round(4), "p4_copenhagen_scales.csv", "copenhagen",
      "Copenhagen set: groups, tablets and content range per scale code; tablets outside the source-condition range (scale codes 0 and 1)")
print(pd.DataFrame(scale_rows).round(3).to_string(index=False))
assert int(src_mask.sum()) == 190 and int(new_mask.sum()) == 120

# %%
# Arrays for the later notebooks and their digest (names, shapes, types, identifiers, groups and axes included)
ARRAYS = dict(s_x1=S["x1"], s_x2=S["x2"], s_assay=S["assay"], s_part=S["part"], s_tablet=S["tablet"],
              s_wavelength_nm=S["wavelength_nm"], c_x=C["x"], c_content=C["content"], c_type=C["type"], c_scale=C["scale"],
              c_group=C["group"], c_tablet=C["tablet"], c_wavenumber_cm1=C["wavenumber_cm1"])
ARRAYS_SHA = P.array_digest(ARRAYS)
print("arrays digest", ARRAYS_SHA)

# %%
# Frozen splits. Positions are row numbers inside each dataset's arrays; tablet identifiers are stored beside them.
# Each list of repetitions comes from one seeded random stream and is frozen as a list: the splits are defined by the
# stored file, not by re-drawing. If the cost rule of plan section 7 applies, Scenario I keeps repetitions 0-99 and
# Scenario II keeps orders 0-1 (then order 0) within every one of the 81 choices of pool groups.
train_pos = np.flatnonzero(S["part"] != "test")          # calibration + validation: 195 source training tablets
test_pos = np.flatnonzero(S["part"] == "test")           # 460 new-condition tablets
assert train_pos.size == 195 and test_pos.size == 460

s1 = P.splits_scenario_one(test_pos.size, S1_POOL, S1_REPS, SEED + 10)
for rep in s1:                                            # positions inside the test part -> row numbers
    rep["pool_order"] = test_pos[rep["pool_order"]].tolist()
    rep["eval"] = test_pos[rep["eval"]].tolist()
s1_folds = [[train_pos[f].tolist() for f in rpt] for rpt in P.repeated_kfold(train_pos.size, S1_FOLDS, S1_FOLD_REPEATS, SEED + 11)]


def by_type(scale_codes):
    sel = g_df[g_df["scale"].isin(scale_codes)]
    return {int(t): sel.loc[sel["type"] == t, "group"].tolist() for t in sorted(sel["type"].unique())}


members = {g: np.flatnonzero(C["group"] == g).tolist() for g in g_df["group"]}
s2 = P.splits_by_group(by_type([S2_NEW_SCALE]), members, S2_ORDERS, SEED + 20)
s2_folds = P.group_folds(by_type(S2_SOURCE_SCALES), S2_FOLDS, SEED + 21)
pseudo_01 = P.splits_by_group(by_type([1]), members, 1, SEED + 30, max_choices=PSEUDO_CHOICES_0_TO_1)
pseudo_10 = P.splits_by_group(by_type([0]), members, 1, SEED + 31)
# Tuning folds inside each pseudo-source scale, so that base settings for pseudo-transfer are chosen without the
# other scale: scale 0 has 7 groups (one group per fold), scale 1 has 12 (4 folds of 3 groups)
pseudo_folds_0 = [[g] for g in sorted(g_df.loc[g_df["scale"] == 0, "group"])]
pseudo_folds_1 = P.group_folds(by_type([1]), PSEUDO_FOLDS_SCALE1, SEED + 32)

splits = {
    "version": 2, "seed": SEED, "arrays_sha256": ARRAYS_SHA,
    "scenario_I": {"train": train_pos.tolist(), "grid": S1_GRID, "pool_size": S1_POOL, "repetitions": s1, "cv_folds": s1_folds,
                   "tablet": S["tablet"].tolist()},
    "scenario_II": {"source_scales": list(S2_SOURCE_SCALES), "new_scale": S2_NEW_SCALE, "per_group_grid": S2_PER_GROUP,
                    "train": np.flatnonzero(src_mask).tolist(), "repetitions": s2, "cv_group_folds": s2_folds,
                    "tablet": C["tablet"].tolist(), "group": C["group"].tolist()},
    "pseudo_transfer": {"per_group": PSEUDO_PER_GROUP,
                        "scale0_to_scale1": {"train": np.flatnonzero(C["scale"] == 0).tolist(), "repetitions": pseudo_01,
                                             "cv_group_folds": pseudo_folds_0},
                        "scale1_to_scale0": {"train": np.flatnonzero(C["scale"] == 1).tolist(), "repetitions": pseudo_10,
                                             "cv_group_folds": pseudo_folds_1}},
}

# %%
# Assertions, row by row: every list holds valid, distinct rows of the right partition; training, update and
# evaluation tablets are disjoint; in the group-structured splits no evaluation tablet shares a group with an update
# or training tablet; the numbers of repetitions, choices, orders and folds are exactly the planned ones.
def distinct_rows(rows, allowed, what):
    assert all(isinstance(i, int) for i in rows), f"{what}: row numbers must be whole numbers"
    assert len(set(rows)) == len(rows), f"{what}: repeated row"
    assert set(rows) <= allowed, f"{what}: row outside its partition"


ids1 = S["tablet"]
test_set, train_set = set(test_pos.tolist()), set(train_pos.tolist())
assert not (test_set & train_set) and len(set(ids1[train_pos].tolist()) & set(ids1[test_pos].tolist())) == 0
assert [r["rep"] for r in s1] == list(range(S1_REPS))
for rep in s1:
    distinct_rows(rep["pool_order"], test_set, "Scenario I pool")
    distinct_rows(rep["eval"], test_set, "Scenario I evaluation")
    assert len(rep["pool_order"]) == S1_POOL and len(rep["eval"]) == test_pos.size - S1_POOL
    assert not (set(rep["pool_order"]) & set(rep["eval"])) and set(rep["pool_order"]) | set(rep["eval"]) == test_set
assert len({tuple(r["pool_order"]) for r in s1}) == S1_REPS, "Scenario I repetitions must differ"
assert len(s1_folds) == S1_FOLD_REPEATS
for rpt in s1_folds:
    assert len(rpt) == S1_FOLDS and [len(f) for f in rpt] == [39] * S1_FOLDS
    flat = [i for f in rpt for i in f]
    distinct_rows(flat, train_set, "Scenario I folds")
    assert set(flat) == train_set, "cross-validation folds must partition the training tablets"


def check_group_reps(reps, train_rows, new_scale, n_choices, n_orders, name):
    train_rows = [int(i) for i in train_rows]
    new_rows = set(np.flatnonzero(C["scale"] == new_scale).tolist())
    train_groups = set(C["group"][train_rows].tolist())
    new_groups = set(C["group"][sorted(new_rows)].tolist())
    assert not (train_groups & new_groups) and not (set(train_rows) & new_rows)
    assert [r["rep"] for r in reps] == list(range(len(reps))) and len(reps) == n_choices * n_orders, name
    seen = {}
    for rep in reps:
        pool_rows = [i for g in rep["pool_groups"] for i in rep["pool_order"][g]]
        distinct_rows(pool_rows, new_rows, f"{name} pool")
        distinct_rows(rep["eval"], new_rows, f"{name} evaluation")
        pool_groups, eval_groups = set(rep["pool_groups"]), set(rep["eval_groups"])
        assert len(rep["pool_groups"]) == len(pool_groups) == 4 and set(rep["pool_order"]) == pool_groups
        assert len({int(C["type"][members[g][0]]) for g in pool_groups}) == 4, "one pool group per tablet type"
        assert pool_groups | eval_groups == new_groups and not (pool_groups & eval_groups) and eval_groups
        assert all(sorted(rep["pool_order"][g]) == members[g] for g in pool_groups), "pool order must hold each pool group whole"
        assert sorted(rep["eval"]) == sorted(i for g in eval_groups for i in members[g]), "evaluation must hold each evaluation group whole"
        assert not (set(pool_rows) & set(rep["eval"]))
        seen.setdefault(tuple(rep["pool_groups"]), []).append(rep["order"])
    assert len(seen) == n_choices and all(sorted(v) == list(range(n_orders)) for v in seen.values()), f"{name}: orders per choice"


check_group_reps(s2, np.flatnonzero(src_mask), S2_NEW_SCALE, 81, S2_ORDERS, "Scenario II")
assert all(len(r["eval"]) == 80 and len(r["eval_groups"]) == 8 for r in s2)
check_group_reps(pseudo_01, np.flatnonzero(C["scale"] == 0), 1, PSEUDO_CHOICES_0_TO_1, 1, "pseudo 0 to 1")
check_group_reps(pseudo_10, np.flatnonzero(C["scale"] == 1), 0, 8, 1, "pseudo 1 to 0")
assert all(len(r["eval_groups"]) == 3 for r in pseudo_10)


def check_group_folds(folds, scale_codes, n_folds, name):
    expected = sorted(g_df.loc[g_df["scale"].isin(scale_codes), "group"].tolist())
    flat = [g for f in folds for g in f]
    assert len(folds) == n_folds and all(folds) and sorted(flat) == expected and len(set(flat)) == len(flat), name


check_group_folds(s2_folds, S2_SOURCE_SCALES, S2_FOLDS, "Scenario II folds")
check_group_folds(pseudo_folds_0, [0], 7, "pseudo folds scale 0")
check_group_folds(pseudo_folds_1, [1], PSEUDO_FOLDS_SCALE1, "pseudo folds scale 1")
print("Scenario I repetitions:", len(s1), "| Scenario II repetitions:", len(s2), "| pseudo-transfer:", len(pseudo_01), "+", len(pseudo_10))
print("Scenario II group folds:", s2_folds)

# %%
# Compare with an existing splits file BEFORE anything is written. The frozen splits may not change, and they are
# bound to the arrays by the digest stored inside them.
split_bytes = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode("utf-8")
SPLITS_SHA = P.sha256_bytes(split_bytes)
splits_path = os.path.join(OUT, "p4_splits.json")
cache_path = os.path.join(CACHE, "p4_arrays.npz")
if os.path.exists(splits_path):
    with open(splits_path, "rb") as f:
        assert f.read() == split_bytes, "p4_splits.json exists and differs from this run: nothing was written"
    print("p4_splits.json reproduced byte for byte")
summary = pd.DataFrame([
    dict(item="Scenario I source training tablets", value=int(train_pos.size)),
    dict(item="Scenario I new-condition tablets", value=int(test_pos.size)),
    dict(item="Scenario I repetitions", value=len(s1)),
    dict(item="Scenario I update pool / evaluation tablets per repetition", value=f"{S1_POOL} / {test_pos.size - S1_POOL}"),
    dict(item="Scenario II source tablets (groups)", value=f"{int(src_mask.sum())} ({int(g_df['scale'].isin(S2_SOURCE_SCALES).sum())})"),
    dict(item="Scenario II new-condition tablets (groups)", value=f"{int(new_mask.sum())} ({int(g_df['scale'].eq(S2_NEW_SCALE).sum())})"),
    dict(item="Scenario II repetitions (choices of pool groups x orders)", value=f"{len(s2)} (81 x {S2_ORDERS})"),
    dict(item="Scenario II evaluation tablets per repetition", value=80),
    dict(item="Pseudo-transfer repetitions, scale 0 to 1 / 1 to 0", value=f"{len(pseudo_01)} / {len(pseudo_10)}"),
    dict(item="Shootout row pairing supported by the spectra in every part (pre-set rule)", value=str(PAIRING_SUPPORTED)),
    dict(item="sha256 of p4_splits.json", value=SPLITS_SHA),
    dict(item="digest of the arrays the splits index (stored inside p4_splits.json)", value=ARRAYS_SHA),
])
table(summary, "p4_splits_summary.csv", "splits", "Counts of the frozen splits, the pairing rule's result, and the hashes that bind splits and arrays")
hashes.append(("data/cache/p4_arrays.npz (digest over all 13 arrays: names, shapes, types, values)", 0, ARRAYS_SHA,
               "arrays used by notebooks 02-04; they must check this digest against p4_splits.json before indexing"))
table(pd.DataFrame(hashes, columns=["file", "bytes", "sha256", "what"]), "p4_input_hashes.csv", "arrays",
      "SHA-256 of the two archives, of the data files read from them, and the digest of the arrays passed to the later notebooks")
table(pd.DataFrame([dict(package=k, version=v) for k, v in {**VERSIONS, "python": platform.python_version(), "p4_common": P.COMMON_VERSION}.items()]),
      "p4_versions_01.csv", "versions", "Package versions of the notebook 01 run")

# %%
# Write everything: each file goes to a temporary name first and is then renamed; the manifest is updated last.
def publish(path, write):
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def write_npz(tmp):
    with open(tmp, "wb") as f:
        np.savez(f, **ARRAYS)


publish(cache_path, write_npz)
back = np.load(cache_path)
assert P.array_digest({k: back[k] for k in back.files}) == ARRAYS_SHA, "the saved arrays do not round-trip"
if not os.path.exists(splits_path):
    publish(splits_path, lambda tmp: open(tmp, "wb").write(split_bytes))
    print("p4_splits.json written")
rows = [("p4_splits.json", NOTEBOOK, "splits", "Frozen splits: Scenario I (200 repetitions, cross-validation folds), Scenario II (243 repetitions, group folds), pseudo-transfer splits and folds inside the source condition; holds the digest of the arrays it indexes")]
for name, df, cell, description in TABLES:
    publish(os.path.join(OUT, name), lambda tmp, df=df: df.to_csv(tmp, index=False, lineterminator="\n"))
    rows.append((name, NOTEBOOK, cell, description))
man_path = os.path.join(OUT, "MANIFEST.md")
HEADER = ["| file | notebook | cell | description |", "|---|---|---|---|"]
lines = open(man_path, encoding="utf-8").read().splitlines() if os.path.exists(man_path) else []
other = [l for l in lines if l.startswith("|") and l not in HEADER and f"| {NOTEBOOK} |" not in l]
body = HEADER + other + [f"| {f} | {nb} | {cell} | {desc} |" for f, nb, cell, desc in rows]
publish(man_path, lambda tmp: open(tmp, "w", encoding="utf-8", newline="\n").write("\n".join(body) + "\n"))
print("files written:", len(rows), "| manifest rows of this notebook:", len(rows))
print("DONE", NOTEBOOK, "| pairing supported:", PAIRING_SUPPORTED)
