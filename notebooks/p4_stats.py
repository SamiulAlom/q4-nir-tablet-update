"""Statistics of notebook 04 (analysis_plan.md sections 6 and 7): scores from stored per-tablet predictions, paired
ratios over repetitions, recovery budgets, range splits and region summaries. Pure functions on arrays; no file I/O.
No significance test anywhere (plan section 7)."""
import numpy as np
import pandas as pd

STATS_VERSION = "1"
THRESHOLD = 1.10          # recovery budget: median ratio to the benchmark at most 1.10 (plan section 6)
PCT = (2.5, 97.5)         # repetition percentiles (plan section 7)


def split_label(label):
    """'cnn1|U2|800|40' -> (family, seed, update, setting, n); seed is -1 for families without seeds."""
    lab, update, setting, n = str(label).split("|")
    fam, seed = ("cnn", int(lab[3:])) if lab.startswith("cnn") else (lab, -1)
    return fam, seed, update, setting, int(n)


def scores(pred, y_true, labels, keep=None):
    """RMSE and mean error (bias) per repetition and method label, from the stored 32-bit predictions.
    pred: (R, L, E); y_true: (R, E). keep: optional boolean mask (R, E) of the evaluation tablets to score.
    Returns a long table: rep_index, label, family, seed, update, setting, n, rmse, bias, n_scored."""
    pred = np.asarray(pred)
    y = np.asarray(y_true, dtype=np.float64)
    assert pred.ndim == 3 and y.shape == (pred.shape[0], pred.shape[2]) and len(labels) == pred.shape[1]
    err = pred.astype(np.float64) - y[:, None, :]
    if keep is None:
        keep = np.ones(y.shape, dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    assert keep.shape == y.shape
    cnt = keep.sum(axis=1)                                            # (R,)
    w = keep[:, None, :].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt((err ** 2 * w).sum(axis=2) / cnt[:, None])
        bias = (err * w).sum(axis=2) / cnt[:, None]
    parts = [split_label(l) for l in labels]
    R, L = rmse.shape
    df = pd.DataFrame(dict(rep_index=np.repeat(np.arange(R), L), label=np.tile(np.asarray(labels, dtype=str), R),
                           rmse=rmse.ravel(), bias=bias.ravel(), n_scored=np.repeat(cnt, L)))
    meta = pd.DataFrame(parts, columns=["family", "seed", "update", "setting", "n"])
    meta["label"] = np.asarray(labels, dtype=str)
    return df.merge(meta, on="label", how="left")


def by_family(score_df):
    """CNN seeds averaged per repetition (plan section 7: the three seeds' RMSEs are averaged; bias likewise).
    Returns rep_index, family, update, setting, n, rmse, bias."""
    keys = ["rep_index", "family", "update", "setting", "n"]
    out = score_df.groupby(keys, as_index=False).agg(rmse=("rmse", "mean"), bias=("bias", "mean"), seeds=("seed", "nunique"))
    bad = out[(out["family"] == "cnn") & (out["seeds"] != 3)]
    assert bad.empty, "every CNN score must average three seeds"
    return out.drop(columns="seeds")


def describe(values):
    """Median and the 2.5th and 97.5th percentiles over repetitions (linear interpolation)."""
    v = np.asarray(values, dtype=np.float64)
    assert v.ndim == 1 and v.size > 0 and np.isfinite(v).all()
    lo, hi = np.percentile(v, PCT)
    return dict(median=float(np.median(v)), p2_5=float(lo), p97_5=float(hi), reps=int(v.size))


def paired_ratio(a, b):
    """Paired ratio a/b over repetitions (same splits): median, percentiles, and the share of repetitions with a < b."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1 and (b > 0).all()
    d = describe(a / b)
    d["share_lower"] = float(np.mean(a < b))
    return d


def budget(n_grid, median_ratio, threshold=THRESHOLD):
    """Recovery budget: the first n on the grid whose median ratio is at most the threshold, or None ("not reached");
    reversal is True when a larger n on the grid exceeds the threshold again."""
    n_grid, med = list(n_grid), list(median_ratio)
    assert len(n_grid) == len(med) and n_grid == sorted(n_grid)
    first = next((i for i, m in enumerate(med) if m <= threshold), None)
    if first is None:
        return None, False
    return n_grid[first], any(m > threshold for m in med[first + 1:])


def range_masks(y_true, lo, hi):
    """Evaluation tablets inside the source-condition content range, above its largest value and below its smallest."""
    y = np.asarray(y_true, dtype=np.float64)
    return dict(inside=(y >= lo) & (y <= hi), above=y > hi, below=y < lo)


def drop_group_masks(eval_group):
    """One mask per evaluation group g: True for the tablets kept when g is left out (plan section 7)."""
    g = np.asarray(eval_group).astype(str)
    return {k: g != k for k in sorted(set(g.ravel().tolist()))}


def region_summary(axis, diff, n_regions=5):
    """Mean absolute value and the signed mean of a difference spectrum in equal-width regions of the axis."""
    axis, diff = np.asarray(axis, dtype=np.float64), np.asarray(diff, dtype=np.float64)
    assert axis.shape == diff.shape and axis.ndim == 1
    lo, hi = float(axis.min()), float(axis.max())
    edges = np.linspace(lo, hi, n_regions + 1)
    rows = []
    for i in range(n_regions):
        m = (axis >= edges[i]) & ((axis < edges[i + 1]) if i < n_regions - 1 else (axis <= edges[i + 1]))
        rows.append(dict(region=i + 1, start=float(edges[i]), end=float(edges[i + 1]), points=int(m.sum()),
                         mean_abs=float(np.mean(np.abs(diff[m]))), mean_signed=float(np.mean(diff[m]))))
    return rows
