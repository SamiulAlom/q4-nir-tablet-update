"""Tests of p4_stats.py on generated numbers (code tests only; no data file is read). Run: python test_p4_stats.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p4_stats as S


def test_split_label():
    assert S.split_label("cnn2|U2|800|40") == ("cnn", 2, "U2", "800", 40)
    assert S.split_label("pls|U0|-|0") == ("pls", -1, "U0", "-", 0)


def test_scores_and_family():
    labels = ["pls|U1|-|5", "cnn0|U1|-|5", "cnn1|U1|-|5", "cnn2|U1|-|5"]
    y = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    pred = np.zeros((2, 4, 4), dtype=np.float32)
    pred[:, 0, :] = y + np.array([1, -1, 1, -1])          # rmse 1, bias 0
    pred[:, 1, :] = y + 2                                   # rmse 2, bias 2
    pred[:, 2, :] = y + 4
    pred[:, 3, :] = y
    df = S.scores(pred, y, labels)
    r = df[(df.label == "pls|U1|-|5")]
    assert np.allclose(r.rmse, 1) and np.allclose(r.bias, 0)
    f = S.by_family(df)
    c = f[f.family == "cnn"]
    assert np.allclose(c.rmse, 2.0) and np.allclose(c.bias, 2.0)
    keep = np.array([[True, False, True, False]] * 2)       # only the +1 errors
    r = S.scores(pred, y, labels, keep)
    r = r[r.label == "pls|U1|-|5"]
    assert np.allclose(r.rmse, 1) and np.allclose(r.bias, 1) and (r.n_scored == 2).all()


def test_by_family_needs_three_seeds():
    labels = ["cnn0|U1|-|5", "cnn1|U1|-|5"]
    df = S.scores(np.zeros((1, 2, 3), np.float32), np.ones((1, 3)), labels)
    try:
        S.by_family(df)
        raise AssertionError("should have stopped")
    except AssertionError as e:
        assert "three seeds" in str(e)


def test_describe_and_ratio():
    d = S.describe(np.arange(1, 101, dtype=float))
    assert d["median"] == 50.5 and d["reps"] == 100 and abs(d["p2_5"] - 3.475) < 1e-9
    r = S.paired_ratio(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
    assert r["median"] == 1.0 and abs(r["share_lower"] - 1 / 3) < 1e-12


def test_budget():
    assert S.budget([0, 5, 10], [1.5, 1.05, 1.2]) == (5, True)
    assert S.budget([0, 5, 10], [1.5, 1.2, 1.1]) == (10, False)
    assert S.budget([0, 5, 10], [1.5, 1.2, 1.11]) == (None, False)
    assert S.budget([0, 5], [1.0, 1.0]) == (0, False)


def test_masks_and_regions():
    m = S.range_masks(np.array([1.0, 5.0, 9.0]), 2.0, 8.0)
    assert m["inside"].tolist() == [False, True, False] and m["above"].tolist() == [False, False, True] and m["below"].tolist() == [True, False, False]
    d = S.drop_group_masks(np.array([["a", "b"], ["b", "c"]]))
    assert sorted(d) == ["a", "b", "c"] and d["b"].tolist() == [[True, False], [False, True]]
    rows = S.region_summary(np.arange(10.0), np.r_[np.ones(5), -np.ones(5)], 2)
    assert rows[0]["points"] == 5 and rows[0]["mean_signed"] == 1 and rows[1]["mean_signed"] == -1 and rows[1]["mean_abs"] == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
