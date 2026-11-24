# %% [markdown]
# # RUNTIME: local PC, CPU (the human's order of 2026-11-19; plan section 10)
#
# Project: `qpain/paper-04` - Stage: `04 statistics and figures` - Colab units: 0
#
# Rebuilds every table and figure of the paper from the per-tablet prediction files of notebook 03 and the outputs of
# notebooks 01 and 02, as analysis_plan.md sections 6, 7 and 11 (v1.5 and deviations 1-2) say. No model is fitted and
# no significance test is computed. Every input is checked against the hash in its notebook's completion marker.
#
# Run: `jupyter nbconvert --to notebook --execute` with `PAPER_ROOT` set; `P4_SMOKE=1` reads the smoke test of notebook
# 03 (logs/smoke-03-out) and writes to logs/smoke-04-out, never to outputs/.
# Source: `notebooks/src/04_statistics_figures_CPU.py` (edit the source, rebuild with scripts/py-to-ipynb.py).

# %%
import os, sys, json, platform, datetime
import importlib.metadata as ilmd
ROOT = os.environ.get("PAPER_ROOT", os.path.abspath(".."))
SMOKE = os.environ.get("P4_SMOKE", "0") == "1"
OUT_REAL = os.path.join(ROOT, "outputs")
IN03 = os.path.join(ROOT, "logs", "smoke-03-out") if SMOKE else OUT_REAL
OUT = os.path.join(ROOT, "logs", "smoke-04-out") if SMOKE else OUT_REAL
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
NOTEBOOK = "04_statistics_figures_CPU"
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import p4_common as P, p4_models as M, p4_stats as S
VERSIONS = {m: ilmd.version(m) for m in ("numpy", "pandas", "matplotlib")}
VERSIONS.update(python=platform.python_version(), p4_stats=S.STATS_VERSION)
print("Run at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "| SMOKE", SMOKE, VERSIONS)

# %%
# Inputs, bound by hash to the completion markers of notebooks 02 and 03
with open(os.path.join(OUT_REAL, "p4_complete_02.json"), encoding="utf-8") as f:
    MARK02 = json.load(f)
for name, sha in MARK02["files"].items():
    assert P.sha256_file(os.path.join(OUT_REAL, name)) == sha, f"{name} differs from notebook 02's marker"
with open(os.path.join(IN03, "p4_complete_03.json"), encoding="utf-8") as f:
    MARK03 = json.load(f)
assert MARK03["notebook"] == "03_update_curves_GPU" and MARK03["smoke"] is SMOKE
if not SMOKE:
    assert (MARK03["S1_REPS"], MARK03["S2_ORDERS"]) == (100, 1), "deviation 2 fixed the design at 100 repetitions and 1 order"
    assert MARK03["repetitions"] == dict(pseudo_01=27, pseudo_10=8, s2=81, s1_12=100, s1_21=100)
for name, sha in MARK03["files"].items():
    assert P.sha256_file(os.path.join(IN03, name)) == sha, f"{name} differs from notebook 03's marker"
SPLITS_SHA_AUDITED = "034d226f695684bfd80be90445b2d9ba6b6def65ae71660c49acb3bc4dfa7a35"   # notebook 01
with open(os.path.join(OUT_REAL, "p4_splits.json"), "rb") as f:
    split_bytes = f.read()
assert P.sha256_bytes(split_bytes) == SPLITS_SHA_AUDITED
SPLITS = json.loads(split_bytes)
NEEDED = ["p4_update_settings.json", "p4_bench_s1.csv"] + [f"p4_pred_{u}.npz" for u in ("s1_12", "s1_21", "s2", "pseudo_01", "pseudo_10")]
assert all(n in MARK03["files"] for n in NEEDED), "notebook 03's marker must list every file read here"
with np.load(os.path.join(ROOT, "data", "cache", "p4_arrays.npz")) as npz:
    A = {k: npz[k] for k in npz.files}
assert P.array_digest(A) == SPLITS["arrays_sha256"]
with open(os.path.join(OUT_REAL, "p4_tuning_chosen.json"), encoding="utf-8") as f:
    CHOSEN = json.load(f)["chosen"]
_folds = pd.read_csv(os.path.join(OUT_REAL, "p4_tuning_folds.csv"))
CANDIDATES = _folds.groupby(["source_set", "family"])[["preprocessing", "setting"]].apply(lambda g: len(g.drop_duplicates())).to_dict()
with open(os.path.join(IN03, "p4_update_settings.json"), encoding="utf-8") as f:
    SETTINGS = json.load(f)
assert SETTINGS["fingerprint"] == MARK03["fingerprint"] and SETTINGS["smoke"] is SMOKE
assert SETTINGS["tuned_sha256"] == MARK02["files"]["p4_tuning_chosen.json"] and SETTINGS["splits_sha256"] == SPLITS_SHA_AUDITED
S1R = SPLITS["scenario_I"]["repetitions"][:(2 if SMOKE else MARK03["S1_REPS"])]
S2R = ([r for r in SPLITS["scenario_II"]["repetitions"] if r["choice"] == 0][:2] if SMOKE
       else [r for r in SPLITS["scenario_II"]["repetitions"] if r["order"] < MARK03["S2_ORDERS"]])
PTR = {u: SPLITS["pseudo_transfer"][k]["repetitions"][:(2 if SMOKE else None)] for u, k in (("pseudo_01", "scale0_to_scale1"), ("pseudo_10", "scale1_to_scale0"))}
DESIGN = dict(s1_12=S1R, s1_21=S1R, s2=S2R, **PTR)
UNITS = {}
for u in ("s1_12", "s1_21", "s2", "pseudo_01", "pseudo_10"):
    with np.load(os.path.join(IN03, f"p4_pred_{u}.npz")) as z:
        UNITS[u] = {k: z[k] for k in z.files}
    d = UNITS[u]
    assert d["pred"].dtype == np.float32 and np.isfinite(d["pred"]).all() and d["pred"].shape == (len(d["rep"]), len(d["labels"]), d["eval_rows"].shape[1])
    assert d["rep"].tolist() == [r["rep"] for r in DESIGN[u]], f"{u}: repetitions are not the frozen design's"
    assert all(d["eval_rows"][i].tolist() == r["eval"] for i, r in enumerate(DESIGN[u])), f"{u}: evaluation rows are not the frozen design's"
BENCH = pd.read_csv(os.path.join(IN03, "p4_bench_s1.csv"))
print("inputs verified:", {u: d["pred"].shape for u, d in UNITS.items()})

# %%
# Scores per repetition (CNN seeds averaged), and the main settings
FAMILIES = ["pls", "xgb", "cnn", "tabpfn"]
NAMES = dict(pls="PLS", xgb="Boosted trees", cnn="1D CNN", tabpfn="TabPFN")
U2_GRID = dict(pls="pls_copies", xgb="xgb_copies", cnn="cnn_steps", tabpfn="tabpfn_repeat")
MAIN = {"s1_12": {k: f"{v:g}" for k, v in SETTINGS["scenario_I"]["main"].items()},
        "s1_21": {k: f"{v:g}" for k, v in SETTINGS["scenario_I"]["main"].items()},
        "s2": {k: f"{v['chosen']:g}" for k, v in SETTINGS["scenario_II"].items()}}
SC = {u: S.by_family(S.scores(d["pred"], d["y_true"], d["labels"])) for u, d in UNITS.items()}
for u, d in UNITS.items():
    SC[u]["rep"] = d["rep"][SC[u]["rep_index"].to_numpy()]
GRID_N = {u: sorted(SC[u]["n"].unique().tolist()) for u in ("s1_12", "s1_21", "s2")}


def series(u, fam, update, setting, n, frame=None):
    """RMSE per repetition (ordered by repetition) for one method; U0 is looked up at n = 0."""
    df = SC[u] if frame is None else frame
    if n == 0:
        update, setting = "U0", "-"
    m = df[(df.family == fam) & (df["update"] == update) & (df.setting == setting) & (df.n == n)].sort_values("rep_index")
    assert len(m) > 0, (u, fam, update, setting, n)
    return m["rmse"].to_numpy(), m["rep_index"].to_numpy()


def main_methods(u):
    """(label, family, update, setting) of the methods shown in the paper: U1 for every family, each family's own
    update U2 at the main setting, and PLS with T at the main setting."""
    out = [(f"{NAMES[f]} U1", f, "U1", "-") for f in FAMILIES]
    out += [(f"{NAMES[f]} U2", f, "U2", MAIN[u][U2_GRID[f]]) for f in FAMILIES]
    return out + [("PLS T", "pls", "T", MAIN[u]["t_factor"])]


# %%
# Curves: RMSE and bias per method and n, over repetitions (median, 2.5th-97.5th percentiles); all settings kept
rows = []
for u in ("s1_12", "s1_21", "s2", "pseudo_01", "pseudo_10"):
    for (fam, upd, st, n), g in SC[u].groupby(["family", "update", "setting", "n"]):
        g = g.sort_values("rep_index")
        rows.append(dict(unit=u, family=fam, update=upd, setting=st, n=int(n), **S.describe(g["rmse"]),
                         bias_median=float(np.median(g["bias"])), main_setting=(u in MAIN and (upd in ("U0", "U1") or (upd == "U2" and st == MAIN[u][U2_GRID[fam]]) or (upd == "T" and st == MAIN[u]["t_factor"])))))
CURVES = pd.DataFrame(rows)

# %%
# Benchmarks and the opening table (plan sections 5, 6; v1.5; deviation 1)
def bench_rmse(unit):
    b = BENCH[BENCH.unit == unit].copy()
    b["fam"] = b["family_label"].str.replace(r"cnn\d", "cnn", regex=True)
    per = b.groupby(["fam", "family_label"]).apply(lambda g: float(np.sqrt(np.mean((g.prediction - g.reference) ** 2))), include_groups=False)
    out = per.groupby(level=0).mean()                                   # CNN: mean over the three seeds
    assert set(out.index) == set(FAMILIES) and (b.groupby("family_label").size() == 460).all()
    return out.to_dict()


BENCHMARK = {"s1_12": bench_rmse("s1_12"), "s1_21": bench_rmse("s1_21"),
             "s2": {f: CHOSEN["s2_source"][f]["cv_rmse_selected"] for f in FAMILIES}}
CV = {"s1_12": {f: CHOSEN["s1_inst1"][f]["cv_rmse_selected"] for f in FAMILIES},
      "s1_21": {f: CHOSEN["s1_inst2"][f]["cv_rmse_selected"] for f in FAMILIES},
      "s2": {f: CHOSEN["s2_source"][f]["cv_rmse_selected"] for f in FAMILIES}}
LEVEL_B = {"s1_12": BENCHMARK["s1_21"], "s1_21": BENCHMARK["s1_12"]}   # v1.5: the opposite direction's benchmark
rows = []
for u in ("s1_12", "s1_21", "s2"):
    for f in FAMILIES:
        v, _ = series(u, f, "U0", "-", 0)
        d = S.describe(v)
        rows.append(dict(unit=u, family=f, cv_rmse_source=CV[u][f], benchmark=BENCHMARK[u][f], level_b=LEVEL_B.get(u, {}).get(f, np.nan),
                         u0_median=d["median"], u0_p2_5=d["p2_5"], u0_p97_5=d["p97_5"], reps=d["reps"],
                         ratio_u0_to_cv=d["median"] / CV[u][f], candidates_searched=CANDIDATES[({"s1_12": "s1_inst1", "s1_21": "s1_inst2", "s2": "s2_source"}[u], f)],
                         cv_scheme=("repeated 5-fold, random (optimistic: selection score)" if u.startswith("s1") else "group-held-out 5-fold (optimistic: selection score)")))
OPENING = pd.DataFrame(rows)

# %%
# Paired comparisons (plan sections 6, 7). One specification serves the comparison table and the leave-one-group-out
# table: at n = 0 each family's U0 against PLS U0; at n > 0 each family's U1, every U2 setting and every T setting
# against PLS U1 at the same n, and every U2 setting against the same family's U1 (secondary question 3).
def settings_of(u, f, upd):
    return sorted(set(SC[u][(SC[u].family == f) & (SC[u]["update"] == upd)]["setting"]), key=float)


def is_main(u, f, upd, st):
    return upd in ("U0", "U1") or (upd == "U2" and st == MAIN[u][U2_GRID[f]]) or (upd == "T" and st == MAIN[u]["t_factor"])


def contrasts(u, n):
    """(method, setting, against, (family, update, setting), (family, update, setting), primary)"""
    if n == 0:
        return [(f"{NAMES[f]} U0", "-", "PLS U0", (f, "U0", "-"), ("pls", "U0", "-"), False) for f in FAMILIES[1:]]
    out = [(f"{NAMES[f]} U1", "-", "PLS U1", (f, "U1", "-"), ("pls", "U1", "-"), u == "s1_12" and n == 10) for f in FAMILIES[1:]]
    for f in FAMILIES:
        for st in settings_of(u, f, "U2"):
            out.append((f"{NAMES[f]} U2", st, "PLS U1", (f, "U2", st), ("pls", "U1", "-"), False))
            out.append((f"{NAMES[f]} U2", st, "same family, U1", (f, "U2", st), (f, "U1", "-"), False))
    out += [("PLS T", st, "PLS U1", ("pls", "T", st), ("pls", "U1", "-"), False) for st in settings_of(u, "pls", "T")]
    return out


def ratio_of(u, n, a_spec, b_spec, frame=None):
    a, ai = series(u, *a_spec, n, frame)
    b, bi = series(u, *b_spec, n, frame)
    assert (ai == bi).all() and np.isfinite(a).all() and np.isfinite(b).all() and (b > 0).all()
    return a, b


rows = []
for u in ("s1_12", "s1_21", "s2"):
    for n in GRID_N[u]:
        for method, st, against, a_spec, b_spec, prim in contrasts(u, n):
            a, b = ratio_of(u, n, a_spec, b_spec)
            rows.append(dict(unit=u, n=n, method=method, setting=st, against=against, main_setting=is_main(u, *a_spec), primary=prim, **S.paired_ratio(a, b)))
COMPARE = pd.DataFrame(rows)
assert int(COMPARE.primary.sum()) == 3, "the primary comparison has three contrasts"
PRIMARY = COMPARE[(COMPARE.unit == "s1_12") & (COMPARE.n == 10) & (COMPARE.against == "PLS U1") & COMPARE.main_setting].reset_index(drop=True)
assert len(PRIMARY) == 8, "three primary contrasts, four own updates and PLS with T, all against PLS with U1"

# %%
# Recovery budget (plan section 6, deviation 1): per repetition RMSE / benchmark of the family; first n with median
# ratio <= 1.10; "not reached"; reversal flag. n = 0 uses U0. Every setting is reported; main_setting marks the main one.
rows = []
for u in ("s1_12", "s1_21", "s2"):
    for f in FAMILIES:
        methods = [("U1", "-")] + [(upd, st) for upd, st in sorted(set(map(tuple, SC[u][(SC[u].family == f) & SC[u]["update"].isin(["U2", "T"])][["update", "setting"]].to_numpy())))]
        for upd, st in methods:
            med = [float(np.median(series(u, f, upd, st, n)[0] / BENCHMARK[u][f])) for n in GRID_N[u]]
            n_b, rev = S.budget(GRID_N[u], med)
            main = upd == "U1" or (upd == "U2" and st == MAIN[u][U2_GRID[f]]) or (upd == "T" and st == MAIN[u]["t_factor"])
            rows.append(dict(unit=u, family=f, update=upd, setting=st, main_setting=main, benchmark=BENCHMARK[u][f],
                             budget=("not reached" if n_b is None else str(n_b)), reversal=rev, **{f"median_ratio_n{n}": m for n, m in zip(GRID_N[u], med)}))
BUDGET = pd.DataFrame(rows)

# %%
# Scenario II: content range, groups, and the leave-one-group-out range of each comparison (plan sections 6, 7)
d2 = UNITS["s2"]


def masked(mask):
    """Scores on the masked tablets; repetitions with no such tablet are dropped (counted in reps), never others."""
    sc = S.scores(d2["pred"], d2["y_true"], d2["labels"], mask)
    sc = sc[sc.n_scored > 0]
    assert np.isfinite(sc.rmse).all() and np.isfinite(sc.bias).all()
    return S.by_family(sc)


lo, hi = float(d2["source_content_min"]), float(d2["source_content_max"])
masks = S.range_masks(d2["y_true"], lo, hi)
rows = []
for where, mk in masks.items():
    if not mk.any():
        rows.append(dict(where=where, tablets_distinct=0))
        continue
    fam_sc = masked(mk)
    distinct = int(len(set(d2["eval_rows"][mk].tolist())))
    for lab, f, upd, st in [("no update", f, "U0", "-") for f in FAMILIES] + main_methods("s2"):
        for n in GRID_N["s2"]:
            if (upd == "U0") != (n == 0):
                continue
            v, _ = series("s2", f, upd, st, n, fam_sc)
            rows.append(dict(where=where, source_min=lo, source_max=hi, family=f, update=upd, setting=st, n=n, tablets_distinct=distinct,
                             tablets_per_rep_median=float(np.median(mk.sum(axis=1))), **S.describe(v)))
RANGE = pd.DataFrame(rows)

gmask = S.drop_group_masks(d2["eval_group"])
grows = []
full = {(n, c[0], c[1], c[2]): float(np.median(np.divide(*ratio_of("s2", n, c[3], c[4])))) for n in GRID_N["s2"] for c in contrasts("s2", n)}
logo = {k: [] for k in full}
for g, keep in gmask.items():
    only = masked(~keep)                                                         # this group's tablets only
    gtype = int(d2["eval_type"][~keep][0])
    for lab, f, upd, st in [("no update", f, "U0", "-") for f in FAMILIES] + main_methods("s2"):
        for n in GRID_N["s2"]:
            if (upd == "U0") != (n == 0):
                continue
            v, _ = series("s2", f, upd, st, n, only)
            grows.append(dict(group=g, type=gtype, family=f, update=upd, setting=st, n=n, **S.describe(v)))
    drop = masked(keep)                                                          # this group left out
    for n in GRID_N["s2"]:
        for c in contrasts("s2", n):
            logo[(n, c[0], c[1], c[2])].append(float(np.median(np.divide(*ratio_of("s2", n, c[3], c[4], drop)))))
GROUPS = pd.DataFrame(grows)
LOGO = pd.DataFrame([dict(n=n, method=m, setting=st, against=ag, median_ratio=full[(n, m, st, ag)], leave_one_group_out_min=min(v),
                          leave_one_group_out_max=max(v), groups=len(v)) for (n, m, st, ag), v in logo.items()])
assert (LOGO.groups == len(gmask)).all()

# %%
# Difference spectra (plan section 6, v1.5): differences of means over the same set of rows; no tablet-level pairing
sg1 = lambda x: M.preprocess(x, "sg1")
wl, wn = A["s_wavelength_nm"], A["c_wavenumber_cm1"]
c_src = np.isin(A["c_scale"], [0, 1])
DIFF = dict(
    shootout_mean=A["s_x2"].mean(0) - A["s_x1"].mean(0), shootout_sg1=sg1(A["s_x2"]).mean(0) - sg1(A["s_x1"]).mean(0),
    copen_mean=A["c_x"][A["c_scale"] == 2].mean(0) - A["c_x"][c_src].mean(0), copen_sg1=sg1(A["c_x"][A["c_scale"] == 2]).mean(0) - sg1(A["c_x"][c_src]).mean(0))
rows = []
for key, ax in (("shootout_mean", wl), ("shootout_sg1", wl), ("copen_mean", wn), ("copen_sg1", wn)):
    for r in S.region_summary(ax, DIFF[key]):
        rows.append(dict(difference=key, **r))
REGIONS = pd.DataFrame(rows)

# %%
# Figures (result-figures skill): IEEE two-column page, Okabe-Ito colours plus markers, percentile bands named in the caption
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42, "lines.linewidth": 1.0, "axes.linewidth": 0.6})
COL = dict(pls="#000000", xgb="#E69F00", cnn="#0072B2", tabpfn="#009E73")
MK = dict(pls="o", xgb="s", cnn="^", tabpfn="D")


def curve(ax, u, f, upd, st, label, color, marker, ls="-"):
    xs = np.array(GRID_N[u], dtype=float)
    med, lo_, hi_ = [], [], []
    for n in GRID_N[u]:
        dd = S.describe(series(u, f, upd, st, n)[0])
        med.append(dd["median"]); lo_.append(dd["p2_5"]); hi_.append(dd["p97_5"])
    ax.fill_between(xs, lo_, hi_, color=color, alpha=0.12, linewidth=0)
    ax.plot(xs, med, ls, color=color, marker=marker, markersize=2.8, label=label)


def curves_figure(units, path, titles):
    fig, axes = plt.subplots(len(units), 2, figsize=(7.16, 0.4 + 2.1 * len(units)), layout="constrained", squeeze=False)
    letters = iter("abcdefgh")
    for i, u in enumerate(units):
        for j in range(2):
            ax = axes[i, j]
            for f in FAMILIES:
                if j == 0:
                    curve(ax, u, f, "U1", "-", NAMES[f], COL[f], MK[f])
                else:
                    curve(ax, u, f, "U2", MAIN[u][U2_GRID[f]], NAMES[f], COL[f], MK[f])
            if j == 1:
                curve(ax, u, "pls", "T", MAIN[u]["t_factor"], "PLS, Tikhonov-type update (T)", "#CC79A7", "v", "--")
            else:                                             # each family's benchmark (Scenario II: optimistic source CV)
                for f in FAMILIES:
                    ax.axhline(BENCHMARK[u][f], color=COL[f], linestyle=":", linewidth=0.8)
            ax.set_xticks(GRID_N[u], [str(n) for n in GRID_N[u]])
            ax.set_xlabel("labelled tablets from the new condition, n")
            ax.set_ylabel("RMSE, " + ("assay value" if u.startswith("s1") else "% w/w"))
            ax.set_title(f"({next(letters)}) " + titles[u] + (": slope and bias (U1)" if j == 0 else ": own update (U2) and T"), loc="left")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    h, l = axes[0, 1].get_legend_handles_labels()
    h.append(matplotlib.lines.Line2D([], [], color="#555555", linestyle=":", linewidth=0.8)); l.append("benchmark (left panels)")
    fig.legend(h, l, loc="outside lower center", ncol=6, frameon=False)
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=600)
    plt.close(fig)


TITLES = {"s1_12": "Instrument 1 to 2", "s1_21": "Instrument 2 to 1", "s2": "Scale codes 0+1 to 2"}
curves_figure(["s1_12", "s2"], os.path.join(OUT, "p4_fig_curves"), TITLES)
curves_figure(["s1_21"], os.path.join(OUT, "p4_fig_curves_s1_21"), TITLES)

fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.0), layout="constrained")
for i, (key, ax_v, xl) in enumerate((("shootout", wl, "wavelength, nm"), ("copen", wn, "wavenumber, cm$^{-1}$"))):
    for j, kind in enumerate(("mean", "sg1")):
        ax = axes[i, j]
        ax.plot(ax_v, DIFF[f"{key}_{kind}"], color="#000000", linewidth=0.7)
        ax.axhline(0, color="#999999", linewidth=0.4)
        ax.set_xlabel(xl)
        ax.set_title("(" + "abcd"[2 * i + j] + ") " + ("instr. 2 - 1" if key == "shootout" else "scale 2 - (0+1)") + (", spectra" if kind == "mean" else ", 1st der."), loc="left", fontsize=6)
        if key == "copen":
            ax.invert_xaxis()
axes[0, 0].set_ylabel("difference of means"); axes[1, 0].set_ylabel("difference of means")
fig.savefig(os.path.join(OUT, "p4_fig_diff_spectra.pdf")); fig.savefig(os.path.join(OUT, "p4_fig_diff_spectra.png"), dpi=600)
plt.close(fig)

# %%
# Write the tables, the manifest rows and the completion marker (manifest first, marker last; an old marker is removed first)
if os.path.exists(os.path.join(OUT, "p4_complete_04.json")):
    os.remove(os.path.join(OUT, "p4_complete_04.json"))
TABLES = {"p4_tab_curves.csv": (CURVES, "RMSE per method, setting and n over repetitions: median, 2.5th and 97.5th percentiles, median bias; CNN seeds averaged per repetition; main_setting marks the settings shown in the paper"),
          "p4_tab_opening.csv": (OPENING, "Opening table: source cross-validation RMSE (optimistic; Scenario II group-held-out), benchmark (Scenario I: source model on the 460 test-part tablets on the source instrument), level (b), and the no-update RMSE over repetitions"),
          "p4_tab_compare.csv": (COMPARE, "Paired RMSE ratios per repetition: each method against PLS with slope and bias at the same n, and each family's own update against its slope and bias; median, percentiles, share of repetitions lower"),
          "p4_tab_primary.csv": (PRIMARY, "Pre-specified primary comparison: instrument 1 to 2, n = 10, each family against PLS (all with slope and bias), with the other methods shown beside it"),
          "p4_tab_budget.csv": (BUDGET, "Recovery budget: median ratio of RMSE to the family's benchmark at every n, the first n at most 1.10 (or not reached), and the reversal flag; every setting"),
          "p4_tab_s2_range.csv": (RANGE, "Scenario II: RMSE over repetitions for evaluation tablets inside the source content range and above its largest value, with tablet counts"),
          "p4_tab_s2_groups.csv": (GROUPS, "Scenario II: RMSE per evaluation group over the repetitions in which the group is evaluated"),
          "p4_tab_s2_logo.csv": (LOGO, "Scenario II: median paired ratio against PLS with slope and bias, and its range when one evaluation group at a time is left out"),
          "p4_tab_diff_regions.csv": (REGIONS, "Difference spectra (differences of means over the same rows, no tablet pairing) summarised in five equal-width regions")}
FIGS = {"p4_fig_curves.pdf": "Figure: RMSE against n, instrument 1 to 2 and scale codes 0+1 to 2; medians with 2.5th-97.5th percentile bands over repetitions",
        "p4_fig_curves.png": "Same figure, 600 dpi",
        "p4_fig_curves_s1_21.pdf": "Figure: RMSE against n, instrument 2 to 1 (secondary direction)",
        "p4_fig_curves_s1_21.png": "Same figure, 600 dpi",
        "p4_fig_diff_spectra.pdf": "Figure: mean and first-derivative difference spectra between instruments and between scale codes",
        "p4_fig_diff_spectra.png": "Same figure, 600 dpi"}
for name, (df, _) in TABLES.items():
    num = df.select_dtypes(include=[np.number])
    assert not num.isin([np.inf, -np.inf]).any().any(), name
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(df.to_csv(index=False, lineterminator="\n"))
with open(os.path.join(OUT, "p4_versions_04.csv"), "w", encoding="utf-8", newline="\n") as f:
    f.write(pd.DataFrame([dict(item=k, value=v) for k, v in VERSIONS.items()]).to_csv(index=False, lineterminator="\n"))
files = list(TABLES) + list(FIGS) + ["p4_versions_04.csv"]
if not SMOKE:
    rows = [(n, NOTEBOOK, "tables", TABLES[n][1]) for n in TABLES] + [(n, NOTEBOOK, "figures", d) for n, d in FIGS.items()]
    rows += [("p4_versions_04.csv", NOTEBOOK, "versions", "Package versions of the notebook 04 run"),
             ("p4_complete_04.json", NOTEBOOK, "outputs", "Completion marker of notebook 04: input marker hashes and the hash of each output file")]
    man_path = os.path.join(OUT, "MANIFEST.md")
    HEADER = ["| file | notebook | cell | description |", "|---|---|---|---|"]
    with open(man_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    other = [l for l in lines if l.startswith("|") and l not in HEADER and f"| {NOTEBOOK} |" not in l]
    with open(man_path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(HEADER + other + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]) + "\n")
    os.replace(man_path + ".tmp", man_path)
marker = dict(notebook=NOTEBOOK, smoke=SMOKE, inputs=dict(complete_02=P.sha256_file(os.path.join(OUT_REAL, "p4_complete_02.json")),
              complete_03=P.sha256_file(os.path.join(IN03, "p4_complete_03.json"))), files={n: P.sha256_file(os.path.join(OUT, n)) for n in files})
with open(os.path.join(OUT, "p4_complete_04.json.tmp"), "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(marker, sort_keys=True, indent=1))
os.replace(os.path.join(OUT, "p4_complete_04.json.tmp"), os.path.join(OUT, "p4_complete_04.json"))
print("DONE", NOTEBOOK, "| SMOKE", SMOKE)
