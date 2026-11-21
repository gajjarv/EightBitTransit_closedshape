"""Pixel inversions: weighted bounded least squares (CLSQ) and the closed-shape pixel inversion."""
import numpy as np
from scipy import ndimage, sparse
from scipy.optimize import lsq_linear
from .forward import grid_coords


def gradient_matrix(N, M):
    """Finite-difference operator (horizontal + vertical neighbours) on an N x M grid, shape (n_edges, N*M)."""
    rows, cols, vals, r = [], [], [], 0
    for i in range(N):
        for j in range(M):
            k = i * M + j
            if j + 1 < M:
                rows += [r, r]; cols += [k, k + 1]; vals += [1, -1]; r += 1
            if i + 1 < N:
                rows += [r, r]; cols += [k, k + M]; vals += [1, -1]; r += 1
    return sparse.csr_matrix((vals, (rows, cols)), shape=(r, N * M)).toarray()


def clsq(A, flux, sig, N, M, lam=0.0, mu=0.0, centre=None, support=None, zero_cols=0):
    """Noise-weighted, bounded (0 <= opacity <= 1) least squares on the full (unfolded) grid.

    lam:     L2 gradient (smoothness) penalty weight, in units of the whitened residuals.
    mu:      compactness penalty weight; penalises opacity proportional to (distance from `centre` in pixels)^2.
    support: boolean (N, M) mask of pixels allowed to be non-zero.
    zero_cols: force the first `zero_cols` columns to zero (EightBitTransit-style ingress padding; default off).
    Returns the (N, M) opacity image.
    """
    flux = np.asarray(flux, float)
    sg = np.broadcast_to(np.asarray(sig, float), flux.shape)
    y = (1 - flux) / sg
    Aw = A / sg[:, None]
    cols = np.ones(N * M, bool) if support is None else np.asarray(support, bool).ravel().copy()
    if zero_cols:
        z = np.zeros((N, M), bool); z[:, :zero_cols] = True
        cols &= ~z.ravel()
    As = Aw[:, cols]
    if lam > 0:
        G = gradient_matrix(N, M)[:, cols]
        G = G[np.abs(G).sum(1) > 0]
        As = np.vstack([As, lam * G]); y = np.concatenate([y, np.zeros(len(G))])
    if mu > 0 and centre is not None:
        xc, yc, w = grid_coords(N, M)
        X, Y = np.meshgrid(xc, yc)
        dd = (np.hypot(X - centre[0], Y - centre[1]) / w).ravel()[cols]
        As = np.vstack([As, np.diag(np.sqrt(mu) * dd)]); y = np.concatenate([y, np.zeros(len(dd))])
    out = lsq_linear(As, y, bounds=(0, 1), lsmr_tol="auto", max_iter=3000)
    g = np.zeros(N * M)
    g[cols] = out.x
    return g.reshape(N, M)


def largest_blob(g, thr=0.3, prefer_upper=True):
    """Boolean mask of the connected component (4-connectivity) of g > thr*max(g) with the largest summed opacity,
    holes filled. Ties are broken towards the upper half of the grid (mirror-degeneracy convention)."""
    N, M = g.shape
    m = g > thr * g.max() if g.max() > 0 else np.zeros_like(g, bool)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    _, yc, _ = grid_coords(N, M)
    score = [g[lab == k].sum() + 1e-9 * prefer_upper * (yc[np.where(lab == k)[0]].mean() > 0) for k in range(1, n + 1)]
    return ndimage.binary_fill_holes(lab == 1 + int(np.argmax(score)))


def closed_pixel(A, flux, sig, N, M, lam=0.3, mu=10.0, centre=None, thr=0.3, n_iter=2, dilate=1):
    """Closed-shape pixel inversion ("P2"): smooth + compact CLSQ, keep one connected hole-free blob, dilate by
    `dilate` px, refit with every pixel outside that support forced to zero; repeat n_iter times.
    The result is non-zero only on a single connected region, which removes EightBitTransit's up/down mirror
    ambiguity and the arc/stretch smearing by construction. Defaults (lam=0.3, mu=10) were tuned on simulations
    (see README)."""
    g = clsq(A, flux, sig, N, M, lam=lam, mu=mu, centre=centre)
    for _ in range(n_iter):
        keep = largest_blob(g, thr)
        if not keep.any():
            break
        keep = ndimage.binary_dilation(keep, iterations=dilate)
        g = clsq(A, flux, sig, N, M, lam=lam, mu=mu, centre=centre, support=keep)
    return g
