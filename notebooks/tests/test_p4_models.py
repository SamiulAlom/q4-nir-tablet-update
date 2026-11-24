"""Tests of the model helpers on generated spectra (no data file, no GPU, no foundation model needed).

Run:  .venv-local/Scripts/python.exe projects/qpain/paper-04/notebooks/tests/test_p4_models.py
The generated spectra are test inputs for the code only; nothing from them enters the paper.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import p4_models as M


def toy(n=120, p=100, seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    axis = np.linspace(0, 1, p)
    band = np.exp(-((axis - 0.4) / 0.05) ** 2)
    other = np.exp(-((axis - 0.7) / 0.1) ** 2)
    y = rng.uniform(4, 9, n)
    x = (y[:, None] * band + rng.uniform(1, 2, n)[:, None] * other + rng.normal(0, 0.3, n)[:, None]
         + rng.normal(0, noise, (n, p)))
    return x, y


def kfold(n, k=4):
    idx = np.arange(n)
    return [(np.setdiff1d(idx, idx[f::k]), idx[f::k]) for f in range(k)]


def test_preprocess():
    x, _ = toy()
    for name in M.PREPROCESSINGS:
        z = M.preprocess(x, name)
        assert z.shape == x.shape and np.isfinite(z).all(), name
    # per-spectrum: the result for a row does not depend on the other rows
    assert np.allclose(M.preprocess(x[:5], "snv_sg1"), M.preprocess(x, "snv_sg1")[:5])
    assert np.allclose(M.preprocess(x + 3.0, "snv"), M.preprocess(x, "snv")), "SNV removes an offset"
    assert np.allclose(M.preprocess(x + 3.0, "sg1"), M.preprocess(x, "sg1")), "a derivative removes an offset"
    assert not np.shares_memory(M.preprocess(x, "none"), x)
    flat = x.copy()
    flat[3] = 1.5                                   # a constant spectrum: stop, never drop the row
    for name, must_fail in (("snv", True), ("snv_sg1", True), ("sg1", False)):
        try:
            M.preprocess(flat, name)
            failed = False
        except ValueError:
            failed = True
        assert failed == must_fail, name
    bad = x.copy()
    bad[0, 0] = np.nan
    try:
        M.preprocess(bad, "none")
        raise AssertionError("a missing value must stop the run")
    except ValueError:
        pass
    b = M.bin_pairs(x)
    assert b.shape == (x.shape[0], x.shape[1] // 2) and np.allclose(b[:, 0], x[:, :2].mean(axis=1))


def test_inner_split():
    rows = np.arange(50)
    fit, stop = M.inner_split(rows, None, seed=1)
    assert len(stop) == 10 and not set(fit) & set(stop) and len(fit) + len(stop) == 50
    groups = np.repeat(np.arange(10), 5).astype(str)
    fit, stop = M.inner_split(rows, groups, seed=1)
    assert not set(groups[fit]) & set(groups[stop]) and len(set(groups[stop])) == 2
    assert (M.inner_split(rows, groups, seed=1)[1] == stop).all()


def test_one_se_and_pls():
    table = np.array([[3.0, 1.0, 0.95, 0.94], [3.2, 1.1, 1.05, 1.04], [2.8, 0.9, 0.85, 0.86]])
    k, mean, se = M.one_se_choice(table)
    # means 3.0, 1.0, 0.95, 0.947; the best is the last, its standard error 0.052, so the limit is 0.999:
    # candidate 2 (0.95) qualifies, candidate 1 (1.0) just does not
    assert k == 2 and abs(mean[3] + se[3] - 0.9987) < 1e-3, "the simplest candidate within one standard error of the best"
    wide = table.copy()
    wide[:, 3] = [0.80, 1.10, 0.94]                 # a noisier best: now candidate 1 is inside the limit
    assert M.one_se_choice(wide)[0] == 1
    x, y = toy()
    cv = M.pls_cv(x, y, kfold(len(y)))
    assert cv.shape == (4, 15) and cv[:, 2].mean() < cv[:, 0].mean()
    k, mean, _ = M.one_se_choice(cv)
    model = M.pls_fit(x[:90], y[:90], k + 1)
    assert M.rmse(y[90:], M.pls_predict(model, x[90:])) < 0.2 * y.std()


def test_xgb():
    x, y = toy()
    sc, trees = M.xgb_cv(x, y, kfold(len(y)), M.XGB_GRID[0], None, seed=3)
    assert sc.shape == (4,) and (trees >= 1).all() and (trees <= M.XGB_MAX_TREES).all() and sc.mean() < y.std()
    m = M.xgb_fit(x, y, M.XGB_GRID[0], 50, seed=3)
    assert m.predict(x).shape == (len(y),)


def test_cnn_cpu():
    x, y = toy(n=80, p=64)
    z = M.preprocess(x, "snv")
    fit, stop = M.inner_split(np.arange(80), None, seed=0)
    model, scaler, ep = M.cnn_train(z[fit], y[fit], 1e-3, seed=0, device="cpu", stop=(z[stop], y[stop]))
    assert 1 <= ep <= M.CNN_MAX_EPOCHS
    pred = M.cnn_predict(model, scaler, z[stop], "cpu")
    assert pred.shape == (len(stop),) and np.isfinite(pred).all() and M.rmse(y[stop], pred) < y.std()
    # the scaler comes from the training rows only
    assert np.allclose(scaler.x_mean, z[fit].mean(axis=0)) and abs(scaler.y_mean - y[fit].mean()) < 1e-12
    # two training seeds share the inner splits and differ only in training randomness
    folds = kfold(80, 2)
    a, _ = M.cnn_cv(z, y, folds, 1e-3, None, split_seed=5, train_seed=0, device="cpu")
    b, _ = M.cnn_cv(z, y, folds, 1e-3, None, split_seed=5, train_seed=0, device="cpu")
    c, _ = M.cnn_cv(z, y, folds, 1e-3, None, split_seed=5, train_seed=1, device="cpu")
    assert np.allclose(a, b) and not np.allclose(a, c)
    for bad_y in (np.full(len(fit), 3.0), y[fit][:1]):              # no variation, or a single row: scaling must refuse
        try:
            M.Scaler(z[fit][:len(bad_y)], bad_y)
            raise AssertionError("scaling must refuse")
        except ValueError:
            pass
    m2, s2, e2 = M.cnn_train(z[fit], y[fit], 1e-3, seed=0, device="cpu", epochs=3)
    assert e2 == 3
    again, _, _ = M.cnn_train(z[fit], y[fit], 1e-3, seed=0, device="cpu", epochs=3)
    assert np.allclose(M.cnn_predict(m2, s2, z[stop], "cpu"), M.cnn_predict(again, s2, z[stop], "cpu")), "same seed, same model on CPU"


def test_choose():
    rows = [dict(family="pls", preprocessing="snv", setting={}, cv_rmse=1.0, cv_se=0.1, order=1),
            dict(family="pls", preprocessing="none", setting={}, cv_rmse=1.0, cv_se=0.1, order=0),
            dict(family="pls", preprocessing="sg1", setting={}, cv_rmse=1.2, cv_se=0.1, order=2),
            dict(family="xgb", preprocessing="sg1", setting={}, cv_rmse=0.1, cv_se=0.1, order=0)]
    assert M.choose(rows, "pls")["preprocessing"] == "none", "a tie goes to the earlier preprocessing"
    assert M.choose(rows, "xgb")["preprocessing"] == "sg1"
    try:
        M.choose(rows + [dict(family="pls", preprocessing="sg2", setting={}, cv_rmse=float("nan"), cv_se=0.0, order=3)], "pls")
        raise AssertionError("a non-finite score must stop the choice")
    except ValueError:
        pass
    try:
        M.one_se_choice(np.array([[1.0, np.inf], [1.0, 2.0]]))
        raise AssertionError("a non-finite table must stop the rule")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
