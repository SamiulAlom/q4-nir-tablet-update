"""Update methods of paper-04 (analysis_plan.md section 5): what is done with n labelled new-condition tablets.

U1 slope-and-bias, U2 pieces (copies for refits and contexts, CNN fine-tuning) and T (Tikhonov-type PLS updating).
Nothing here reads a data or split file: the caller passes the rows and asserts which tablets a fit may see.
"""
import copy
import hashlib

import numpy as np

import p4_models as M

UPDATES_VERSION = "1"
U1_MIN_FOR_SLOPE = 8                     # below this, bias only (plan section 5)
COPIES_GRID = (1, 5, 20)                 # PLS and boosted-tree refits: copies of each update tablet
CNN_STEPS_GRID = (50, 200, 800)          # fine-tuning steps
CNN_FT_LR = 1e-4
CONTEXT_REPEAT_GRID = (1, 5, 20)         # foundation model: update tablets repeated in the context
LAMBDA_FACTOR_GRID = (0.1, 1.0, 10.0)    # T: lambda as a multiple of the mean squared norm of the centred update spectra
MIDDLE = 1                               # index of the middle value of every grid (main setting of Scenario I)


def seed_for(*parts):
    """A fixed seed from a tuple of labels and numbers (same on every machine and run)."""
    text = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:8], 16) % (2 ** 31 - 1)


def slope_bias(pred_upd, y_upd):
    """U1. Returns (intercept, slope) of the correction `intercept + slope * prediction`.

    Fewer than U1_MIN_FOR_SLOPE update tablets: bias only (slope 1). Otherwise the least-squares line from the
    model's predictions for the update tablets to their reference values. If those predictions are all equal no line
    exists, and the bias-only correction is used.
    """
    p, y = np.asarray(pred_upd, dtype=np.float64), np.asarray(y_upd, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1 or p.size < 1:
        raise ValueError("predictions and reference values must be two equal, non-empty vectors")
    M.require_finite(p, "update predictions")
    M.require_finite(y, "update reference values")
    if p.size < U1_MIN_FOR_SLOPE or np.ptp(p) == 0:
        return float(np.mean(y - p)), 1.0
    pc = p - p.mean()
    slope = float(pc @ (y - y.mean()) / (pc @ pc))
    return float(y.mean() - slope * p.mean()), slope


def pls_linear(model, n_features):
    """Regression vector b and intercept b0 of a fitted PLS model, so that prediction = x @ b + b0."""
    b = np.asarray(model.coef_, dtype=np.float64).ravel()
    if b.shape != (n_features,):
        raise ValueError(f"unexpected coefficient shape {np.asarray(model.coef_).shape}")
    b0 = float(M.pls_predict(model, np.zeros((1, n_features)))[0])
    return b, b0


def tikhonov(b_src, x_upd, y_upd, factor):
    """T. Re-estimate the regression vector from the update tablets with a ridge penalty on its distance from b_src.

    Minimises ||yc - xc b||^2 + lam ||b - b_src||^2, where xc, yc are the update spectra and values centred by their
    own means (so the intercept is re-estimated: b0 = mean(y) - mean(x) @ b) and
    lam = factor * mean squared norm of the centred update spectra. Returns (b, b0, lam).
    """
    x, y = np.asarray(x_upd, dtype=np.float64), np.asarray(y_upd, dtype=np.float64)
    b_src = np.asarray(b_src, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2 or b_src.shape != (x.shape[1],):
        raise ValueError("at least two update tablets and matching shapes are needed")
    xm, ym = x.mean(axis=0), float(y.mean())
    xc, yc = x - xm, y - ym
    lam = float(factor) * float(np.mean(np.sum(xc ** 2, axis=1)))
    if not (np.isfinite(lam) and lam > 0):
        raise ValueError(f"penalty is not positive: {lam}")
    # dual form: the update set is far smaller than the number of variables
    delta = xc.T @ np.linalg.solve(xc @ xc.T + lam * np.eye(x.shape[0]), yc - xc @ b_src)
    b = b_src + delta
    M.require_finite(b, "updated regression vector")
    return b, float(ym - xm @ b), lam


def stack_with_copies(x_src, y_src, x_upd, y_upd, copies):
    """Source rows in their order, then the block of update tablets (in the order given) repeated `copies` times."""
    copies = int(copies)
    if copies < 1:
        raise ValueError("copies must be at least 1")
    return np.vstack([x_src] + [x_upd] * copies), np.concatenate([y_src] + [y_upd] * copies)


def load_cnn(path, device):
    """Source CNN and its scaling from a checkpoint of notebook 02 (plain tensors, weights_only)."""
    torch = M._torch()
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if set(blob) != {"state", "x_mean", "x_sd", "y_mean", "y_sd"}:
        raise ValueError(f"unexpected checkpoint content in {path}")
    model = M.make_cnn(torch)
    model.load_state_dict(blob["state"])
    scaler = object.__new__(M.Scaler)          # the fitted statistics are restored, nothing is fitted again
    scaler.x_mean = blob["x_mean"].numpy().astype(np.float64)
    scaler.x_sd, scaler.y_mean, scaler.y_sd = float(blob["x_sd"]), float(blob["y_mean"]), float(blob["y_sd"])
    stats = np.array([scaler.x_sd, scaler.y_mean, scaler.y_sd])
    if not (scaler.x_mean.ndim == 1 and np.isfinite(scaler.x_mean).all() and np.isfinite(stats).all() and scaler.x_sd > 0 and scaler.y_sd > 0):
        raise ValueError(f"bad scaling statistics in {path}")
    return model.to(device).eval(), scaler


def cnn_finetune(model, scaler, x_src, y_src, x_upd, y_upd, x_eval, steps_at, lr, seed, device):
    """U2 for the CNN. Fine-tune a copy of `model` on all layers; return {step: predictions for x_eval}.

    Each step is one batch: all update tablets plus an equal number of source tablets drawn at random without
    replacement, with the training augmentation. The source scaling is kept. Predictions are taken at each step in
    steps_at on one trajectory, which equals separate runs with the same seed stopped at those steps.
    `model` itself is not changed.
    """
    torch = M._torch()
    steps_at = sorted(int(s) for s in steps_at)
    if not steps_at or steps_at[0] < 1:
        raise ValueError("steps must be positive")
    if len(x_upd) < 1 or len(x_upd) > len(x_src):
        raise ValueError("the update set must be non-empty and no larger than the source set")
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = copy.deepcopy(model).to(device)
    xs, ys = torch.tensor(scaler.x(x_src), device=device), torch.tensor(scaler.y(y_src), device=device)
    xu, yu = torch.tensor(scaler.x(x_upd), device=device), torch.tensor(scaler.y(y_upd), device=device)
    xe = torch.tensor(scaler.x(x_eval), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=M.CNN_WEIGHT_DECAY)
    out = {}
    for step in range(1, steps_at[-1] + 1):
        net.train()
        idx = torch.as_tensor(rng.choice(len(xs), size=len(xu), replace=False), device=device)
        xb, yb = torch.cat([xu, xs[idx]]), torch.cat([yu, ys[idx]])
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(M.augment(torch, xb, gen)), yb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"CNN fine-tuning loss is not finite at step {step}")
        loss.backward()
        opt.step()
        if step in steps_at:
            net.eval()
            with torch.no_grad():
                pred = scaler.y_back(net(xe).cpu().numpy())
            M.require_finite(pred, "fine-tuned CNN predictions")
            out[step] = pred
    return out


def choose_setting(grid, mean_by_direction):
    """Pseudo-transfer choice. mean_by_direction: list (one per direction) of mean RMSE per grid value.

    The directions weigh equally; the value with the lowest average wins; an exact tie goes to the middle value,
    then to the smaller. Returns (chosen value, average per grid value).
    """
    table = np.asarray(mean_by_direction, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != len(grid) or len(grid) != 3:
        raise ValueError("one row per direction and three grid values are needed")
    M.require_finite(table, "pseudo-transfer table")
    avg = table.mean(axis=0)
    best = np.flatnonzero(avg == avg.min()).tolist()
    pick = MIDDLE if MIDDLE in best else min(best)
    return grid[pick], avg
