"""Tests of p4_updates.py on generated numbers (code tests only; no data file is read). Run: python test_p4_updates.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p4_models as M
import p4_updates as U


def test_seed_for():
    assert U.seed_for("a", 1, 2) == U.seed_for("a", 1, 2) != U.seed_for("a", 1, 3)
    assert 0 <= U.seed_for("x") < 2 ** 31


def test_slope_bias():
    p = np.arange(10.0)
    a, b = U.slope_bias(p, 3 + 2 * p)
    assert abs(a - 3) < 1e-12 and abs(b - 2) < 1e-12
    a, b = U.slope_bias(p[:7], 3 + 2 * p[:7])              # 7 tablets: bias only; mean(3 + p) with mean(p) = 3
    assert b == 1.0 and abs(a - 6.0) < 1e-12
    a, b = U.slope_bias(np.full(10, 4.0), np.arange(10.0))   # equal predictions: bias only
    assert b == 1.0 and abs(a - 0.5) < 1e-12
    for bad in ((np.array([1.0, np.nan] * 5), np.ones(10)), (np.ones(3), np.ones(4))):
        try:
            U.slope_bias(*bad)
            raise AssertionError("should have stopped")
        except ValueError:
            pass


def test_tikhonov():
    rng = np.random.default_rng(0)
    n, p = 6, 4                                            # small enough for the primal closed form
    x, y, b_src = rng.normal(size=(n, p)), rng.normal(size=n), rng.normal(size=p)
    b, b0, lam = U.tikhonov(b_src, x, y, 1.0)
    xc, yc = x - x.mean(0), y - y.mean()
    assert abs(lam - np.mean(np.sum(xc ** 2, axis=1))) < 1e-12
    primal = np.linalg.solve(xc.T @ xc + lam * np.eye(p), xc.T @ yc + lam * b_src)
    assert np.allclose(b, primal, atol=1e-10)
    assert abs(b0 - (y.mean() - x.mean(0) @ b)) < 1e-12
    b_big, b0_big, _ = U.tikhonov(b_src, x, y, 1e12)       # huge penalty: the source vector, intercept re-estimated
    assert np.allclose(b_big, b_src, atol=1e-8) and abs(b0_big - (y.mean() - x.mean(0) @ b_src)) < 1e-8
    b_small, _, _ = U.tikhonov(b_src, x, y, 1e-12)         # no penalty: fits the centred update set (n - 1 < p here? no: n > p)
    ols = np.linalg.lstsq(xc, yc, rcond=None)[0]
    assert np.allclose(b_small, ols, atol=1e-5)
    # more variables than tablets (the real case)
    x2, y2, b2 = rng.normal(size=(5, 50)), rng.normal(size=5), rng.normal(size=50)
    b, b0, lam = U.tikhonov(b2, x2, y2, 0.1)
    xc, yc = x2 - x2.mean(0), y2 - y2.mean()
    assert np.allclose(b, np.linalg.solve(xc.T @ xc + lam * np.eye(50), xc.T @ yc + lam * b2), atol=1e-8)


def test_pls_linear_and_copies():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 30))
    y = x[:, :3].sum(axis=1) + 5 + 0.01 * rng.normal(size=40)
    model = M.pls_fit(x, y, 3)
    b, b0 = U.pls_linear(model, 30)
    assert np.allclose(x @ b + b0, M.pls_predict(model, x), atol=1e-9)
    xs, ys = U.stack_with_copies(x, y, x[:2] + 1, y[:2] + 1, 3)
    assert xs.shape == (46, 30) and ys.shape == (46,)
    assert np.array_equal(xs[:40], x) and np.array_equal(xs[40:42], xs[44:46]) and np.array_equal(ys[40:42], y[:2] + 1)


def test_cnn_finetune_and_load():
    import tempfile
    import torch
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 64)).cumsum(axis=1)
    y = x[:, 10] - x[:, 40] + rng.normal(scale=0.1, size=30)
    model, scaler, _ = M.cnn_train(x, y, 1e-3, 0, "cpu", epochs=2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    xu, yu, xe = x[:4] + 0.5, y[:4] + 1.0, x[20:] + 0.5
    a = U.cnn_finetune(model, scaler, x, y, xu, yu, xe, (3, 6), 1e-4, 7, "cpu")
    assert sorted(a) == [3, 6] and a[3].shape == (10,)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "the source model must not change"
    again = U.cnn_finetune(model, scaler, x, y, xu, yu, xe, (3, 6), 1e-4, 7, "cpu")
    assert np.array_equal(a[6], again[6]), "same seed, same result"
    short = U.cnn_finetune(model, scaler, x, y, xu, yu, xe, (3,), 1e-4, 7, "cpu")
    assert np.array_equal(a[3], short[3]), "a snapshot equals a separate run stopped at that step"
    other = U.cnn_finetune(model, scaler, x, y, xu, yu, xe, (6,), 1e-4, 8, "cpu")
    assert not np.array_equal(a[6], other[6])
    # checkpoint round trip in the layout notebook 02 writes
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pt")
        torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, x_mean=torch.tensor(scaler.x_mean),
                        x_sd=torch.tensor(scaler.x_sd), y_mean=torch.tensor(scaler.y_mean), y_sd=torch.tensor(scaler.y_sd)), path)
        m2, s2 = U.load_cnn(path, "cpu")
    p1, p2 = M.cnn_predict(model, scaler, x, "cpu"), M.cnn_predict(m2, s2, x, "cpu")
    assert np.allclose(p1, p2, rtol=1e-4, atol=1e-4), "a reloaded checkpoint predicts as the trained model"


def test_choose_setting():
    g = (1, 5, 20)
    assert U.choose_setting(g, [[3, 2, 1], [1, 2, 3.5]])[0] == 5          # averages 2, 2, 2.25: tie of 1 and 5 -> middle
    assert U.choose_setting(g, [[1, 2, 3], [1, 2, 3]])[0] == 1
    assert U.choose_setting(g, [[2, 3, 2], [2, 3, 2]])[0] == 1            # tie without the middle -> smaller
    assert U.choose_setting(g, [[3, 2, 1], [3, 2, 1]])[0] == 20


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASS")
