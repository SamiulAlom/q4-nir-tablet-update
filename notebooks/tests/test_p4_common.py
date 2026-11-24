"""Tests of the split helpers on made-up indices (no data file needed).

Run:  .venv-local/Scripts/python.exe projects/qpain/paper-04/notebooks/tests/test_p4_common.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import p4_common as P


def test_scenario_one():
    reps = P.splits_scenario_one(460, 100, 20, seed=1)
    again = P.splits_scenario_one(460, 100, 20, seed=1)
    assert reps == again, "same seed must give the same splits"
    assert reps != P.splits_scenario_one(460, 100, 20, seed=2)
    for r in reps:
        pool, ev = set(r["pool_order"]), set(r["eval"])
        assert len(r["pool_order"]) == 100 == len(pool) and len(ev) == 360
        assert not pool & ev and pool | ev == set(range(460))
    assert reps[0]["pool_order"] != reps[1]["pool_order"]
    # every tablet is evaluated in some repetitions and pooled in others
    counts = np.zeros(460)
    for r in reps:
        counts[r["eval"]] += 1
    assert counts.min() >= 1 and counts.max() <= 20


def toy_groups():
    groups_by_type = {t: [f"{t}{c}" for c in "abc"] for t in (1, 2, 3, 4)}
    members, k = {}, 0
    for t in groups_by_type:
        for g in groups_by_type[t]:
            members[g] = list(range(k, k + 10))
            k += 10
    return groups_by_type, members


def test_group_splits():
    groups_by_type, members = toy_groups()
    reps = P.splits_by_group(groups_by_type, members, n_orders=3, seed=5)
    assert len(reps) == 243 and len({tuple(r["pool_groups"]) for r in reps}) == 81
    for r in reps:
        assert len(r["pool_groups"]) == 4 and len(r["eval_groups"]) == 8 and len(r["eval"]) == 80
        assert not set(r["pool_groups"]) & set(r["eval_groups"])
        pool_rows = {i for g in r["pool_groups"] for i in r["pool_order"][g]}
        assert len(pool_rows) == 40 and not pool_rows & set(r["eval"])
        for g in r["pool_groups"]:
            assert sorted(r["pool_order"][g]) == members[g]
    # the three orders of one choice differ, the pool groups do not
    assert reps[0]["pool_groups"] == reps[1]["pool_groups"] and reps[0]["pool_order"] != reps[1]["pool_order"]
    sub = P.splits_by_group(groups_by_type, members, n_orders=1, seed=5, max_choices=27)
    assert len(sub) == 27 and len({tuple(r["pool_groups"]) for r in sub}) == 27
    assert sub == P.splits_by_group(groups_by_type, members, n_orders=1, seed=5, max_choices=27)
    # unequal numbers of groups per type (as in the source scales)
    uneven = {1: ["1a"], 2: ["2a", "2b"], 3: ["3a", "3b"], 4: ["4a", "4b"]}
    mem = {g: [i * 10 + j for j in range(10)] for i, g in enumerate(sorted(g for v in uneven.values() for g in v))}
    reps = P.splits_by_group(uneven, mem, n_orders=1, seed=5)
    assert len(reps) == 8 and all(len(r["eval_groups"]) == 3 for r in reps)


def test_folds():
    groups_by_type = {1: list("abcd"), 2: list("efghi"), 3: list("jklmn"), 4: list("opqrs")}
    folds = P.group_folds(groups_by_type, 5, seed=3)
    flat = sorted(g for f in folds for g in f)
    assert flat == sorted(g for v in groups_by_type.values() for g in v)
    assert sorted(len(f) for f in folds) == [3, 4, 4, 4, 4]
    assert folds == P.group_folds(groups_by_type, 5, seed=3)
    rk = P.repeated_kfold(195, 5, 3, seed=4)
    assert len(rk) == 3
    for rpt in rk:
        assert sorted(i for f in rpt for i in f) == list(range(195)) and [len(f) for f in rpt] == [39] * 5
    assert rk[0] != rk[1]


def test_group_table_and_pairing():
    group = np.array(["01"] * 3 + ["02"] * 3)
    rows = P.group_table(group, np.array([1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1]))
    assert rows == [("01", 3, 1, 0), ("02", 3, 2, 1)]
    try:
        P.group_table(group, np.array([1, 1, 2, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1]))
        raise AssertionError("a mixed group must raise")
    except ValueError:
        pass
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 80))
    a = base + 0.05 * rng.normal(size=base.shape) + 3.0            # second instrument: offset, gain and noise
    b = 1.2 * base + 0.05 * rng.normal(size=base.shape) - 1.0
    rep = P.pairing_report(a, b)
    assert ((rep["rank_ab"] == 1) & (rep["rank_ba"] == 1)).mean() > 0.95 and not rep["invalid"].any()
    assert not np.shares_memory(a, a) or np.isclose(a.mean(), 3.0, atol=0.5), "inputs must not be changed"
    shuffled = P.pairing_report(a, b[rng.permutation(50)])
    assert ((shuffled["rank_ab"] <= 5) & (shuffled["rank_ba"] <= 5)).mean() < 0.3, "a shuffled pairing must not look aligned"
    swap = np.arange(50).reshape(-1, 2)[:, ::-1].ravel()
    swapped = P.pairing_report(a, b[swap])
    assert (swapped["rank_ab"] == 1).mean() < 0.05, "swapped neighbours must not rank first"
    # identical spectra: a swap cannot be told apart, so ties must count against the pairing
    dup_a, dup_b = a.copy(), b.copy()
    dup_a[1], dup_b[1] = dup_a[0], dup_b[0]
    tied = P.pairing_report(dup_a, dup_b)
    assert tied["tied"][0] and tied["tied"][1] and tied["rank_ab"][0] == 2 and tied["rank_ba"][1] == 2
    # a row equal to its instrument's mean has zero length: flagged and given the worst rank
    flat_a = a.copy()
    flat_a[3] = np.delete(flat_a, 3, axis=0).mean(axis=0)
    flat = P.pairing_report(flat_a, b)
    assert flat["invalid"][3] and flat["rank_ab"][3] == 50 and not flat["invalid"][[0, 1, 2]].any()


def test_array_digest():
    x = {"v": np.arange(6, dtype=">f8").reshape(2, 3), "k": np.array([1, 2], dtype=np.int32), "s": np.array(["a", "b"])}
    same = {"v": np.arange(6, dtype="<f8").reshape(2, 3), "k": np.array([1, 2], dtype=np.int64), "s": np.array(["a", "b"])}
    assert P.array_digest(x) == P.array_digest(same), "byte order and integer width must not matter"
    for key, other in (("v", np.arange(6, dtype="<f8").reshape(3, 2)), ("k", np.array([2, 1])), ("s", np.array(["b", "a"]))):
        assert P.array_digest({**same, key: other}) != P.array_digest(same), key
    renamed = {"w": same["v"], "k": same["k"], "s": same["s"]}
    assert P.array_digest(renamed) != P.array_digest(same)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
