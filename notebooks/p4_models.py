"""Model families of paper-04 and their source-only tuning (analysis_plan.md section 4).

Four families predict tablet content from a near-infrared spectrum: PLS, boosted trees, a small 1D CNN and a tabular
foundation model. Everything here is fitted on the rows it is given; no function reads a data file or a split file.
Row sets are passed in by the caller, so the caller decides (and asserts) which tablets a fit may see.
"""
import hashlib

import numpy as np
from scipy.signal import savgol_filter

MODELS_VERSION = "2"   # 2: finite checks, three-seed CNN tuning, offline weights (Codex review 002)
FAMILIES = ("pls", "xgb", "cnn", "tabpfn")
PREPROCESSINGS = ("none", "snv", "sg1", "sg2", "snv_sg1")          # plan section 4, fixed list
PLS_MAX_COMPONENTS = 15
XGB_GRID = [dict(max_depth=d, learning_rate=lr) for d in (2, 3, 4) for lr in (0.03, 0.1)]
XGB_MAX_TREES, XGB_PATIENCE, XGB_COLSAMPLE = 500, 30, 0.3
CNN_GRID = [dict(lr=1e-3), dict(lr=3e-4)]
CNN_MAX_EPOCHS, CNN_PATIENCE, CNN_BATCH, CNN_WEIGHT_DECAY = 300, 30, 32, 1e-4
CNN_AUG = dict(offset=0.10, slope=0.05, scale=0.05)                 # in units of the standardised spectrum
INNER_FRACTION = 0.2                                                # inner split for early stopping
TABPFN_REPO, TABPFN_COMMIT = "Prior-Labs/TabPFN-v2-reg", "02a44705ff7790b50cded404bc59adb94da456a1"
TABPFN_FILE = "tabpfn-v2-regressor.ckpt"
TABPFN_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"   # decision 002
TABPFN_ESTIMATORS = 8


# ---------------------------------------------------------------- preprocessing (per spectrum, nothing is fitted)
def require_finite(values, what):
    """Stop with a message that names the place if any value is missing or infinite."""
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError(f"non-finite values in {what}")


def snv(x):
    """Standard normal variate: each spectrum minus its own mean, divided by its own standard deviation."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)


def preprocess(x, name):
    """Apply one of the fixed preprocessings to spectra (rows). Uses only the spectrum itself."""
    x = np.asarray(x, dtype=np.float64)
    require_finite(x, "spectra before preprocessing")
    if name in ("snv", "snv_sg1") and (x.std(axis=1, ddof=1) == 0).any():
        # policy: a constant spectrum stops the run; no row is ever dropped silently
        raise ValueError("a constant spectrum cannot be scaled by SNV")
    if name == "none":
        out = x.copy()
    elif name == "snv":
        out = snv(x)
    elif name == "sg1":
        out = savgol_filter(x, 15, 2, deriv=1, axis=1)
    elif name == "sg2":
        out = savgol_filter(x, 15, 2, deriv=2, axis=1)
    elif name == "snv_sg1":
        out = savgol_filter(snv(x), 15, 2, deriv=1, axis=1)
    else:
        raise ValueError(f"unknown preprocessing {name}")
    require_finite(out, f"spectra after preprocessing {name}")
    return out


def bin_pairs(x):
    """Average neighbouring variables in pairs (650 -> 325). Used for the foundation model on the shootout spectra."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] % 2:
        raise ValueError("an even number of variables is needed")
    return x.reshape(x.shape[0], -1, 2).mean(axis=2)


def rmse(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def inner_split(rows, groups, seed, fraction=INNER_FRACTION):
    """Split training rows into fit rows and early-stopping rows.

    rows: positions (any integers). groups: None, or one group label per row; whole groups then go to one side, so
    early stopping never sees a tablet of a group it was fitted on.
    Returns (fit, stop) as arrays of positions into `rows` (0..len(rows)-1).
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    if groups is None:
        order = rng.permutation(n)
        k = max(1, int(round(fraction * n)))
        return np.sort(order[k:]), np.sort(order[:k])
    groups = np.asarray(groups)
    names = sorted(set(groups.tolist()))
    if len(names) < 2:
        raise ValueError("at least two groups are needed for a group-held-out inner split")
    k = max(1, int(round(fraction * len(names))))
    held = set(np.array(names)[rng.permutation(len(names))[:k]].tolist())
    stop = np.flatnonzero(np.isin(groups, sorted(held)))
    return np.setdiff1d(np.arange(n), stop), stop


# ---------------------------------------------------------------- PLS
def pls_fit(x, y, n_components):
    from sklearn.cross_decomposition import PLSRegression
    return PLSRegression(n_components=int(n_components), scale=False, max_iter=1000).fit(x, y)


def pls_predict(model, x):
    return np.asarray(model.predict(x), dtype=np.float64).ravel()


def one_se_choice(fold_rmse):
    """One-standard-error rule. fold_rmse: array (folds, candidates), candidates ordered from simplest.

    Returns (index chosen, mean per candidate, standard error per candidate): the first candidate whose mean is at
    most the smallest mean plus the standard error of that minimum-RMSE candidate. Folds weigh equally. The standard
    error is a tuning heuristic (folds share training rows), not a calibrated uncertainty.
    """
    fold_rmse = np.asarray(fold_rmse, dtype=np.float64)
    require_finite(fold_rmse, "cross-validation table")
    if fold_rmse.ndim != 2 or fold_rmse.shape[0] < 2:
        raise ValueError("at least two folds are needed for a standard error")
    mean = fold_rmse.mean(axis=0)
    se = fold_rmse.std(axis=0, ddof=1) / np.sqrt(fold_rmse.shape[0])
    best = int(np.argmin(mean))
    return int(np.flatnonzero(mean <= mean[best] + se[best])[0]), mean, se


def pls_cv(x, y, folds):
    """Cross-validated RMSE of PLS for 1..PLS_MAX_COMPONENTS components. folds: list of (train, test) position arrays."""
    limit = min(PLS_MAX_COMPONENTS, min(len(tr) for tr, _ in folds) - 2, x.shape[1])
    if limit < 1:
        raise ValueError("too few training rows for PLS")
    out = np.empty((len(folds), limit))
    for f, (tr, te) in enumerate(folds):
        for k in range(1, limit + 1):
            out[f, k - 1] = rmse(y[te], pls_predict(pls_fit(x[tr], y[tr], k), x[te]))
    return out


# ---------------------------------------------------------------- boosted trees
def xgb_fit(x, y, params, n_trees, seed, stop=None):
    """Fit boosted trees. With stop=(x_stop, y_stop) training stops early and n_trees is the upper limit."""
    import xgboost
    model = xgboost.XGBRegressor(
        n_estimators=int(n_trees), max_depth=params["max_depth"], learning_rate=params["learning_rate"],
        colsample_bytree=XGB_COLSAMPLE, subsample=1.0, tree_method="hist", device="cpu", n_jobs=4, random_state=seed,
        early_stopping_rounds=XGB_PATIENCE if stop is not None else None)
    if stop is None:
        return model.fit(x, y)
    return model.fit(x, y, eval_set=[stop], verbose=False)


def xgb_cv(x, y, folds, params, groups, seed):
    """Returns (rmse per fold, trees used per fold). Early stopping uses an inner split of each training fold."""
    scores, trees = [], []
    for f, (tr, te) in enumerate(folds):
        fit, stop = inner_split(tr, None if groups is None else groups[tr], seed + f)
        m = xgb_fit(x[tr][fit], y[tr][fit], params, XGB_MAX_TREES, seed, stop=(x[tr][stop], y[tr][stop]))
        scores.append(rmse(y[te], m.predict(x[te])))
        trees.append(int(m.best_iteration) + 1)
    return np.array(scores), np.array(trees)


# ---------------------------------------------------------------- 1D CNN
def _torch():
    import torch
    return torch


def make_cnn(torch):
    nn = torch.nn

    class SpectrumCNN(nn.Module):
        """Three convolution blocks, global average pooling, one hidden layer. No batch normalisation (plan section 4)."""

        def __init__(self):
            super().__init__()
            blocks, c_in = [], 1
            for c_out in (16, 32, 64):
                blocks += [nn.Conv1d(c_in, c_out, kernel_size=7, padding=3), nn.ELU(), nn.MaxPool1d(2)]
                c_in = c_out
            self.features = nn.Sequential(*blocks)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1))

        def forward(self, x):                      # x: (batch, variables)
            h = self.features(x.unsqueeze(1)).mean(dim=2)
            return self.head(h).squeeze(1)

    return SpectrumCNN()


class Scaler:
    """Scaling fitted on training rows only: mean spectrum and one overall standard deviation; target mean and sd."""

    def __init__(self, x, y):
        if len(y) < 2:
            raise ValueError("at least two training rows are needed for scaling")
        self.x_mean = x.mean(axis=0)
        self.x_sd = float((x - self.x_mean).std())
        self.y_mean, self.y_sd = float(y.mean()), float(y.std(ddof=1))
        if not (np.isfinite(self.x_sd) and self.x_sd > 0 and np.isfinite(self.y_sd) and self.y_sd > 0 and np.isfinite(self.x_mean).all()):
            raise ValueError(f"scaling cannot be fitted: spectrum sd {self.x_sd}, target sd {self.y_sd}")

    def x(self, x):
        return ((x - self.x_mean) / self.x_sd).astype(np.float32)

    def y(self, y):
        return ((y - self.y_mean) / self.y_sd).astype(np.float32)

    def y_back(self, z):
        return np.asarray(z, dtype=np.float64) * self.y_sd + self.y_mean


def augment(torch, xb, gen):
    """Random offset, slope along the spectrum and scale, per spectrum (training only)."""
    n, p = xb.shape
    ramp = torch.linspace(-1.0, 1.0, p, device=xb.device)
    r = torch.randn(n, 3, generator=gen, device=xb.device)
    return xb * (1.0 + CNN_AUG["scale"] * r[:, 2:3]) + CNN_AUG["offset"] * r[:, 0:1] + CNN_AUG["slope"] * r[:, 1:2] * ramp


def cnn_train(x, y, lr, seed, device, epochs=None, stop=None):
    """Train the CNN on (x, y).

    With stop=(x_stop, y_stop): at most CNN_MAX_EPOCHS, the weights of the epoch with the lowest stop-set error are
    kept, and training ends after CNN_PATIENCE epochs without improvement. Without stop: exactly `epochs` epochs.
    The scaler is fitted on x, y only. Returns (model, scaler, epochs used).
    """
    torch = _torch()
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    order_rng = np.random.default_rng(seed)
    scaler = Scaler(x, y)
    xt = torch.tensor(scaler.x(x), device=device)
    yt = torch.tensor(scaler.y(y), device=device)
    model = make_cnn(torch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CNN_WEIGHT_DECAY)
    if stop is not None:
        xs = torch.tensor(scaler.x(stop[0]), device=device)
        ys = torch.tensor(scaler.y(stop[1]), device=device)
    best, best_state, best_epoch = np.inf, None, 0
    n_epochs = CNN_MAX_EPOCHS if stop is not None else int(epochs)
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = order_rng.permutation(len(xt))
        for start in range(0, len(order), CNN_BATCH):
            idx = torch.as_tensor(order[start:start + CNN_BATCH], device=device)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(augment(torch, xt[idx], gen)), yt[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f"CNN training loss is not finite in epoch {epoch}")
            loss.backward()
            opt.step()
        if stop is not None:
            model.eval()
            with torch.no_grad():
                val = float(torch.nn.functional.mse_loss(model(xs), ys))
            if np.isfinite(val) and val < best:
                best, best_epoch = val, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= CNN_PATIENCE:
                break
    if stop is not None:
        if best_state is None:
            raise RuntimeError("CNN training gave no finite stop-set loss in any epoch")
        model.load_state_dict(best_state)
        return model, scaler, best_epoch
    return model, scaler, n_epochs


def cnn_predict(model, scaler, x, device):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(scaler.x(x), device=device)).cpu().numpy()
    return scaler.y_back(z)


def cnn_cv(x, y, folds, lr, groups, split_seed, train_seed, device):
    """Returns (rmse per fold, epochs kept per fold). Early stopping uses an inner split of each training fold.

    The inner split of fold f depends on split_seed only, so runs with different train_seed (initial weights, batch
    order, augmentation) share the same fit and stop rows and differ only in training randomness.
    """
    scores, epochs = [], []
    for f, (tr, te) in enumerate(folds):
        fit, stop = inner_split(tr, None if groups is None else groups[tr], split_seed + f)
        model, scaler, ep = cnn_train(x[tr][fit], y[tr][fit], lr, train_seed * 1000 + f, device, stop=(x[tr][stop], y[tr][stop]))
        scores.append(rmse(y[te], cnn_predict(model, scaler, x[te], device)))
        epochs.append(ep)
    return np.array(scores), np.array(epochs)


# ---------------------------------------------------------------- tabular foundation model
def tabpfn_weights(folder, download=True):
    """Path of the pinned version-2 regressor weights, checked by SHA-256 (decision 002).

    A file already in `folder` is used as it is and only read. It is downloaded (by pinned repository commit) only
    if it is missing and download=True.
    """
    import os
    path = os.path.join(folder, TABPFN_FILE)
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(f"{path} is missing and downloading is switched off")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=TABPFN_REPO, filename=TABPFN_FILE, revision=TABPFN_COMMIT, local_dir=folder)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != TABPFN_SHA256:
        raise RuntimeError(f"foundation model weights have an unexpected hash: {h.hexdigest()}")
    return os.path.abspath(path)


def tabpfn_predict(x_context, y_context, x_new, weights, seed, device):
    """Predict x_new with the context (x_context, y_context). Nothing is trained; the weights file is fixed."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, model_path=weights, n_estimators=TABPFN_ESTIMATORS, device=device, random_state=seed)
    model.fit(np.asarray(x_context, dtype=np.float32), np.asarray(y_context, dtype=np.float32))
    pred = np.asarray(model.predict(np.asarray(x_new, dtype=np.float32)), dtype=np.float64).ravel()
    require_finite(pred, "foundation model predictions")
    return pred


def tabpfn_cv(x, y, folds, weights, seed, device):
    return np.array([rmse(y[te], tabpfn_predict(x[tr], y[tr], x[te], weights, seed, device)) for tr, te in folds])


# ---------------------------------------------------------------- choosing settings from cross-validation tables
def choose(rows, family):
    """Pick one row per family from the tuning table.

    rows: list of dicts with keys family, preprocessing, setting, cv_rmse (mean over folds) and cv_se.
    Rule: lowest mean cross-validated RMSE; ties go to the earlier preprocessing in PREPROCESSINGS, then to the
    earlier setting in the family's grid (the row order of the table).
    """
    mine = [r for r in rows if r["family"] == family]
    if not mine:
        raise ValueError(f"no tuning rows for {family}")
    require_finite([r["cv_rmse"] for r in mine], f"tuning scores of {family}")
    return min(mine, key=lambda r: (r["cv_rmse"], PREPROCESSINGS.index(r["preprocessing"]), r["order"]))
