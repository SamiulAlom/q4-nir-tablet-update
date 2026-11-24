"""Shared helpers for paper-04: reading the two tablet data files and building the frozen repetition splits.

Nothing here fits a model. Functions are pure (inputs in, values out) so that they can be tested without the data.
"""
import hashlib
import io
import itertools
import re
import zipfile

import numpy as np

COMMON_VERSION = "2"   # 2: stricter loaders, two-way pairing check, array digest (Codex review 001)

SHOOTOUT_SHA256 = "04c03a4314ab8e8f63e0ecc788b657b5d1b5450791ce465ad41ad90cb5d52125"
TABLETS_SHA256 = "b583e3ad176ad6571d2f38c63dc4af3e9a9f283d59d004f85f3ae85d7a212289"
SHOOTOUT_PARTS = ("calibrate", "validate", "test")


def sha256_file(path):
    """SHA-256 of a file, read in blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_zip_member(zip_bytes, name):
    """Bytes of one member of a zip archive held in memory."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        return z.read(name)


def load_shootout(mat):
    """Arrays from the loaded shootout .mat (scipy.io.loadmat result).

    Returns a dict: x1, x2 (tablets x 650, instruments 1 and 2), assay, part (calibrate / validate / test),
    tablet (one identifier per physical tablet, shared by its two spectra), wavelength_nm.
    A tablet's two spectra are taken to be the same row of the `_1` and `_2` objects of its part: the file gives one
    reference table per part for both instruments and no other identifier, so this rests on the publisher's layout.
    """
    x1, x2, assay, part, tablet, axes = [], [], [], [], [], []
    for p in SHOOTOUT_PARTS:
        a = np.asarray(mat[f"{p}_1"]["data"][0, 0], dtype=np.float64)
        b = np.asarray(mat[f"{p}_2"]["data"][0, 0], dtype=np.float64)
        y_obj = mat[f"{p}_Y"]
        y = np.asarray(y_obj["data"][0, 0], dtype=np.float64)
        names = [str(s).strip() for s in np.asarray(y_obj["label"][0, 0][1, 0]).ravel()]
        if names != ["weight", "hardness", "assay"]:
            raise ValueError(f"{p}: unexpected reference columns {names}")
        if not (a.ndim == 2 and a.shape == b.shape and y.shape == (a.shape[0], 3) and a.shape[1] == 650):
            raise ValueError(f"{p}: shapes do not agree: {a.shape} {b.shape} {y.shape}")
        for inst in ("1", "2"):
            axes.append(np.asarray(mat[f"{p}_{inst}"]["axisscale"][0, 0][1, 0], dtype=np.float64).ravel())
        x1.append(a)
        x2.append(b)
        assay.append(y[:, 2])
        part += [p] * a.shape[0]
        tablet += [f"S_{p[:3]}_{i + 1:03d}" for i in range(a.shape[0])]
    for ax in axes[1:]:
        if not np.array_equal(ax, axes[0]):
            raise ValueError("wavelength axes differ between parts or instruments")
    out = dict(x1=np.vstack(x1), x2=np.vstack(x2), assay=np.concatenate(assay), part=np.array(part),
               tablet=np.array(tablet), wavelength_nm=axes[0])
    if len(set(out["tablet"])) != out["x1"].shape[0]:
        raise ValueError("tablet identifiers are not unique")
    return out


def load_copenhagen(mat):
    """Arrays from the loaded Copenhagen NIR .mat.

    Returns a dict: x (tablets x 404), content (% w/w), type (1-4), scale (0-2), group (label prefix), tablet
    (identifier), wavenumber_cm1.
    """
    m = np.asarray(mat["Matrix"], dtype=np.float64)
    var = [str(s).strip() for s in mat["VarLabels"]]
    obj = [str(s).strip() for s in mat["ObjLabels"]]
    if [v.lower().replace(" ", "") for v in var[:3]] != ["active(%w/w)", "type", "scale"]:
        raise ValueError(f"unexpected first columns {var[:3]}")
    if len(set(obj)) != len(obj):
        raise ValueError("tablet labels are not unique")
    bad = [o for o in obj if not re.fullmatch(r"\d{2}_\d{2}", o)]
    if bad:
        raise ValueError(f"tablet labels not of the form GG_TT: {bad[:5]}")
    if m.ndim != 2 or m.shape != (len(obj), len(var)) or m.shape[1] != 407:
        raise ValueError(f"unexpected matrix shape {m.shape} for {len(obj)} labels and {len(var)} variables")
    if not (np.isfinite(m[:, :3]).all() and set(m[:, 1].tolist()) <= {1.0, 2.0, 3.0, 4.0} and set(m[:, 2].tolist()) <= {0.0, 1.0, 2.0}):
        raise ValueError("type must be 1-4 and scale 0-2, whole numbers")
    out = dict(x=m[:, 3:], content=m[:, 0], type=m[:, 1].astype(int), scale=m[:, 2].astype(int),
               group=np.array([o.split("_")[0] for o in obj]), tablet=np.array([f"C_{o}" for o in obj]),
               wavenumber_cm1=np.array([float(v) for v in var[3:]]))
    return out


def group_table(group, type_, scale):
    """One row per group: (group, n, type, scale). Raises if a group mixes types or scales."""
    rows = []
    for g in sorted(set(group.tolist())):
        idx = np.flatnonzero(group == g)
        t, s = set(type_[idx].tolist()), set(scale[idx].tolist())
        if len(t) != 1 or len(s) != 1:
            raise ValueError(f"group {g} mixes types {t} or scales {s}")
        rows.append((g, int(idx.size), int(t.pop()), int(s.pop())))
    return rows


def splits_scenario_one(n_new, n_pool, n_rep, seed):
    """Scenario I repetitions: positions 0..n_new-1 of the new-condition tablets.

    Each repetition: a random order of all positions; the first n_pool are the update pool in that order (the update
    set of size n is its first n entries, so sets are nested); the rest, sorted, are the evaluation set.
    """
    rng = np.random.default_rng(seed)
    reps = []
    for r in range(n_rep):
        order = rng.permutation(n_new)
        reps.append({"rep": r, "pool_order": order[:n_pool].tolist(), "eval": sorted(order[n_pool:].tolist())})
    return reps


def splits_by_group(groups_by_type, members, n_orders, seed, max_choices=None):
    """Repetitions for a group-structured new condition.

    groups_by_type: {type: [group, ...]} of the new condition. members: {group: [tablet position, ...]}.
    One group per type is the update pool; every other group is evaluated. All choices of pool groups are listed
    (or a seeded random subset of max_choices of them); each is repeated with n_orders random orders of the pool
    tablets within their group. The update set with k tablets per pool group is the first k of each order.
    """
    rng = np.random.default_rng(seed)
    types = sorted(groups_by_type)
    choices = list(itertools.product(*[sorted(groups_by_type[t]) for t in types]))
    if max_choices is not None and max_choices < len(choices):
        keep = sorted(rng.choice(len(choices), size=max_choices, replace=False).tolist())
        choices = [choices[i] for i in keep]
    all_groups = sorted(g for t in types for g in groups_by_type[t])
    reps = []
    for c, pool in enumerate(choices):
        eval_groups = [g for g in all_groups if g not in pool]
        for o in range(n_orders):
            orders = {g: rng.permutation(np.array(members[g])).tolist() for g in pool}
            reps.append({"rep": len(reps), "choice": c, "order": o, "pool_groups": list(pool), "pool_order": orders,
                         "eval_groups": eval_groups, "eval": sorted(i for g in eval_groups for i in members[g])})
    return reps


def group_folds(groups_by_type, n_folds, seed):
    """Group-held-out folds: groups are shuffled within type and dealt round-robin, so types are spread over folds."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    k = 0
    for t in sorted(groups_by_type):
        gs = sorted(groups_by_type[t])
        for g in rng.permutation(len(gs)).tolist():
            folds[k % n_folds].append(gs[g])
            k += 1
    return [sorted(f) for f in folds]


def repeated_kfold(n, n_folds, n_repeats, seed):
    """Repeated K-fold over positions 0..n-1: list of repeats, each a list of n_folds sorted position lists."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_repeats):
        order = rng.permutation(n)
        out.append([sorted(order[f::n_folds].tolist()) for f in range(n_folds)])
    return out


def _unit_rows(x):
    """Centre per variable, scale rows to unit length. Returns the scaled matrix and a mask of rows of (near) zero length."""
    c = x - x.mean(axis=0)
    norm = np.linalg.norm(c, axis=1, keepdims=True)
    scale = np.nanmedian(norm[:, 0])
    # "zero length" up to rounding: a row equal to the mean leaves only floating-point noise after centring
    zero = ~np.isfinite(norm[:, 0]) | (norm[:, 0] <= 1e-9 * scale) | ~np.isfinite(scale)
    return c / np.where(zero[:, None], 1.0, norm), zero


def pairing_report(xa, xb):
    """Spectra-only check of a row pairing between two instruments. Reference values are not used.

    Each matrix is centred per variable and each row scaled to unit length; similarity is the dot product.
    For row i, rank_ab is the number of rows of the second instrument at least as similar to a_i as its own partner
    b_i (so ties count against the pairing; 1 = the partner is strictly the most similar), and rank_ba the same from
    the second instrument's side. Rows of zero length or with a non-finite score get rank n (the worst).
    Returns a dict of arrays: rank_ab, rank_ba, tied (partner shares its score with another row), invalid.
    A correct pairing can still rank badly if the instruments respond differently; a wrong one cannot rank well
    unless spectra are near-identical. The result flags suspect rows; it does not prove identity.
    """
    a, zero_a = _unit_rows(np.asarray(xa, dtype=np.float64))
    b, zero_b = _unit_rows(np.asarray(xb, dtype=np.float64))
    sim = a @ b.T
    n = sim.shape[0]
    own = np.diag(sim).copy()
    invalid = zero_a | zero_b | ~np.isfinite(own) | ~np.isfinite(sim).all(axis=1) | ~np.isfinite(sim).all(axis=0)
    rank_ab = (sim >= own[:, None]).sum(axis=1)          # includes the partner itself, so the best value is 1
    rank_ba = (sim >= own[None, :]).sum(axis=0)
    tied = ((sim == own[:, None]).sum(axis=1) > 1) | ((sim == own[None, :]).sum(axis=0) > 1)
    rank_ab = np.where(invalid, n, rank_ab)
    rank_ba = np.where(invalid, n, rank_ba)
    return dict(rank_ab=rank_ab, rank_ba=rank_ba, tied=tied, invalid=invalid)


def array_digest(arrays):
    """One SHA-256 over named arrays: name, shape, canonical dtype and bytes (little-endian numbers, UTF-8 text).

    Covers every array handed to the later notebooks, so a split file can be bound to the exact rows it indexes.
    """
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.asarray(arrays[name])
        if a.dtype.kind in "US":
            kind, payload = "text", "\x1f".join(str(v) for v in a.ravel().tolist()).encode("utf-8")
        elif a.dtype.kind in "iu":
            kind, payload = "int64", np.ascontiguousarray(a, dtype="<i8").tobytes()
        elif a.dtype.kind == "f":
            kind, payload = "float64", np.ascontiguousarray(a, dtype="<f8").tobytes()
        else:
            raise TypeError(f"{name}: unsupported dtype {a.dtype}")
        h.update(f"{name}|{a.shape}|{kind}|".encode("utf-8"))
        h.update(payload)
    return h.hexdigest()
