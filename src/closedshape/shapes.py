"""Parametric closed outlines: sphere (circle), axis-aligned ellipse, and ellipse x Fourier deformation (star-convex).

Parameter vector c = [log r_area, e, a_1, b_1, ..., a_K, b_K]:
    r(theta) = r_ell(theta; r_area, e) * exp(sum_k a_k cos k theta + b_k sin k theta)
r_ell is the exact ellipse with semi-axes r_area*exp(+e/2) along the motion and r_area*exp(-e/2) perpendicular,
so c[:2] alone is the ellipse model. Signed flattening f = sign(e) (1 - exp(-|e|)); f > 0 = elongated along motion.
"""
import numpy as np
from scipy.optimize import least_squares
from .forward import blocked_flux, grid_coords, rebin


def f_to_e(f):
    return float(np.sign(f) * (-np.log(1 - min(abs(f), 0.97))))


def e_to_f(e):
    return float(np.sign(e) * (1 - np.exp(-abs(e))))


class ShapeModel:
    """Light curve of a closed star-convex occulter centred at (0, b) relative to the orbit's reference point.

    The outline is rasterised on a fixed n x n point grid of half-width R with a sigmoid edge, and the total weight
    is renormalised to the exact enclosed area (smooth in the parameters, so gradient-based fitting works).
    t may be supersampled; nsub sub-exposures per data point are averaged.
    """

    def __init__(self, t, orbit, I, Ftot, R, n=70, nsub=1):
        self.t, self.orbit, self.I, self.Ftot, self.R, self.nsub = np.asarray(t, float), orbit, I, Ftot, R, nsub
        g = (np.arange(n) + 0.5) / n * 2 * R - R
        U, W = np.meshgrid(g, g)
        self.U, self.W = U.ravel(), W.ravel()
        self.h = 2 * R / n
        self.rho = np.hypot(self.U, self.W)
        self.th = np.arctan2(self.W, self.U)
        self._thq = np.linspace(-np.pi, np.pi, 721)[:-1]

    def radius(self, c, th):
        c = np.asarray(c, float)
        ra, e = np.exp(c[0]), c[1]
        A, B = ra * np.exp(e / 2), ra * np.exp(-e / 2)
        r = A * B / np.sqrt((B * np.cos(th)) ** 2 + (A * np.sin(th)) ** 2)
        K = (len(c) - 2) // 2
        if K:
            r = r * np.exp(sum(c[2 * k] * np.cos(k * th) + c[2 * k + 1] * np.sin(k * th) for k in range(1, K + 1)))
        return r

    def area(self, c):
        return np.sum(self.radius(c, self._thq) ** 2 / 2) * (self._thq[1] - self._thq[0])

    def rmax(self, c):
        return self.radius(c, self._thq).max()

    def weights(self, c):
        r = self.radius(c, self.th)
        wt = self.h ** 2 / (1 + np.exp(-(r - self.rho) / (0.35 * self.h)))
        return wt * self.area(c) / wt.sum()

    def flux(self, c, b):
        wt = self.weights(c)
        k = wt > 1e-7 * self.h ** 2
        lc = 1 - blocked_flux(self.t, self.U[k], self.W[k], wt[k], self.orbit, self.I, self.Ftot, b=b, cell=self.h)
        return rebin(lc, self.nsub)

    def moments(self, c):
        """Area, area-equivalent radius, aspect ratio (x/y second moments), and signed flattening of the outline."""
        th = np.linspace(-np.pi, np.pi, 2001)[:-1]
        r = self.radius(c, th); d = th[1] - th[0]
        A = np.sum(r ** 2 / 2) * d
        xb = np.sum(r ** 3 / 3 * np.cos(th)) * d / A
        yb = np.sum(r ** 3 / 3 * np.sin(th)) * d / A
        Ixx = np.sum(r ** 4 / 4 * np.cos(th) ** 2) * d / A - xb ** 2
        Iyy = np.sum(r ** 4 / 4 * np.sin(th) ** 2) * d / A - yb ** 2
        fs = (1 - np.sqrt(Iyy / Ixx)) if Ixx >= Iyy else -(1 - np.sqrt(Ixx / Iyy))
        return dict(area=float(A), r_area=float(np.sqrt(A / np.pi)), aspect=float(np.sqrt(Ixx / Iyy)), f_signed=float(fs))

    def raster(self, c, b, N, M, ss=8):
        """Fractional coverage of the outline (centred at y = b) on an EightBitTransit-style N x M grid."""
        xc, yc, w = grid_coords(N, M)
        sub = ((np.arange(ss) + 0.5) / ss - 0.5) * w
        g = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                U, W = np.meshgrid(xc[j] + sub, yc[i] + sub - b)
                g[i, j] = np.mean(self.radius(c, np.arctan2(W, U)) >= np.hypot(U, W))
        return g


def _fit(fun, x0, lo, hi):
    x0 = np.clip(np.asarray(x0, float), np.asarray(lo) + 1e-6, np.asarray(hi) - 1e-6)
    r = least_squares(fun, x0, bounds=(lo, hi), diff_step=1e-3, x_scale="jac", max_nfev=300)
    return r, np.linalg.pinv(r.jac.T @ r.jac)


def fit_sphere(model, flux, sig, b0s=(0.05, 0.3, 0.55)):
    """Circle of radius r at impact parameter b. Returns dict(r, b, chi2, cov)."""
    flux = np.asarray(flux, float)
    lr0 = np.log(np.sqrt(max(1 - flux.min(), 1e-6)))
    best = None
    for b0 in b0s:
        fun = lambda p: (model.flux([p[0], 0.0], p[1]) - flux) / sig
        r, cov = _fit(fun, [lr0, b0], [-7, 0.0], [np.log(0.85 * model.R), 1.2])
        if best is None or r.cost < best[0].cost:
            best = (r, cov)
    r, cov = best
    return dict(r=float(np.exp(r.x[0])), b=float(r.x[1]), chi2=float(2 * r.cost), cov=cov, c=np.array([r.x[0], 0.0]))


def fit_ellipse(model, flux, sig, sphere=None):
    """Axis-aligned ellipse (r_area, f, b). Returns dict(r_area, f, sig_f, e, b, chi2, cov, c)."""
    sphere = sphere or fit_sphere(model, flux, sig)
    best = None
    e_max = 2 * np.log(0.9 * model.R / sphere["r"])
    for f0 in (0.3, -0.3, 0.6):
        fun = lambda p: (model.flux(p[:2], p[2]) - flux) / sig
        r, cov = _fit(fun, [np.log(sphere["r"]), f_to_e(f0), sphere["b"]], [-7, -e_max, 0.0],
                      [np.log(0.85 * model.R), e_max, 1.2])
        if best is None or r.cost < best[0].cost:
            best = (r, cov)
    r, cov = best
    e, se = r.x[1], np.sqrt(abs(cov[1, 1]))
    return dict(r_area=float(np.exp(r.x[0])), e=float(e), f=e_to_f(e), sig_f=float(np.exp(-abs(e)) * se),
                b=float(r.x[2]), chi2=float(2 * r.cost), cov=cov, c=np.array(r.x[:2]))


def fit_fourier(model, flux, sig, ellipse=None, K=3, prior=0.3):
    """Closed star-convex outline: ellipse x exp(K Fourier harmonics), weak N(0, prior) prior on the harmonics.
    Returns dict(c, b, chi2 (data only), f (signed, from moments), sig_f, aspect, r_area, area, cov)."""
    ellipse = ellipse or fit_ellipse(model, flux, sig)
    x0 = np.concatenate([[np.log(ellipse["r_area"]), ellipse["e"]], np.zeros(2 * K), [ellipse["b"]]])

    def fun(p):
        c = p[:-1]
        if model.rmax(c) >= model.R - 2 * model.h:
            return np.full(len(flux) + 2 * K, 1e3)
        return np.concatenate([(model.flux(c, p[-1]) - flux) / sig, c[2:] / prior])

    lo = np.concatenate([[-7.0, -4.0], -np.ones(2 * K), [0.0]])
    hi = np.concatenate([[np.log(0.85 * model.R), 4.0], np.ones(2 * K), [1.2]])
    r, cov = _fit(fun, x0, lo, hi)
    c = r.x[:-1]
    m = model.moments(c)
    chi2 = float(np.sum(((model.flux(c, r.x[-1]) - flux) / sig) ** 2))
    gr = np.zeros(len(c))
    for k in range(len(c)):
        dc = np.zeros(len(c)); dc[k] = 1e-4
        gr[k] = (model.moments(c + dc)["f_signed"] - model.moments(c - dc)["f_signed"]) / 2e-4
    return dict(c=c, b=float(r.x[-1]), chi2=chi2, f=m["f_signed"], sig_f=float(np.sqrt(abs(gr @ cov[:-1, :-1] @ gr))),
                aspect=m["aspect"], r_area=m["r_area"], area=m["area"], cov=cov)
