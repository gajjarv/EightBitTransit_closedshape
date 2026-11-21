"""Utilities: rasterising ellipses, image-vs-truth metrics, light-curve file I/O."""
import numpy as np
from .forward import grid_coords


def pixelize_ellipse(N, M, x0, y0, ax, ay, ss=40):
    """Fractional coverage of an ellipse (semi-axes ax along motion, ay perpendicular) on an N x M grid."""
    xc, yc, w = grid_coords(N, M)
    sub = (np.arange(ss) + 0.5) / ss - 0.5
    g = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            X, Y = np.meshgrid(xc[j] + sub * w, yc[i] + sub * w)
            g[i, j] = np.mean(((X - x0) / ax) ** 2 + ((Y - y0) / ay) ** 2 <= 1)
    return g


def image_metrics(g, truth):
    """Mirror-invariant comparison of an opacity image g with a truth image (same N x M grid).
    Returns corr (Pearson), iou (threshold 0.5), aspect (x/y second-moment ratio of g), f_moment (signed)."""
    N, M = g.shape
    xc, yc, w = grid_coords(N, M)
    X, Y = np.meshgrid(xc, yc)

    def corr(a, b):
        return np.nan if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    tf = truth[::-1]
    c = max(corr(g, truth), corr(g, tf))
    gb = g >= 0.5
    iou = max((gb & (t >= 0.5)).sum() / max((gb | (t >= 0.5)).sum(), 1) for t in (truth, tf))
    s = g.sum()
    if s <= 0:
        return dict(corr=c, iou=float(iou), aspect=np.nan, f_moment=np.nan)
    xb, yb = (g * X).sum() / s, (g * Y).sum() / s
    Ixx = (g * (X - xb) ** 2).sum() / s + w ** 2 / 12
    Iyy = (g * (Y - yb) ** 2).sum() / s + w ** 2 / 12
    asp = float(np.sqrt(Ixx / Iyy))
    return dict(corr=c, iou=float(iou), aspect=asp, f_moment=(1 - 1 / asp) if asp >= 1 else -(1 - asp))


def load_lightcurve(path):
    """Whitespace/CSV text file with columns time [d], flux, [flux_err]. Lines starting with # are skipped."""
    d = np.loadtxt(path, comments="#", delimiter="," if str(path).endswith(".csv") else None)
    t, f = d[:, 0], d[:, 1]
    e = d[:, 2] if d.shape[1] > 2 else np.full_like(f, np.std(f[: max(5, len(f) // 10)]))
    return t, f, e
