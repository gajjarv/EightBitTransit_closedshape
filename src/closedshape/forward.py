"""Forward model: a rigid, non-rotating occulter (set of weighted sample points or pixels) crossing a limb-darkened star.

Sky frame: x along the direction of motion, y perpendicular, unit = stellar radius R*.
Each occulter point has body-frame offsets (u, w) relative to a reference point that follows the orbit:
  curved (Keplerian circular) orbit:  x = a sin(phi) + u,  y = b_ref cos(phi) + w,  phi = 2 pi (t - t_ref) / P
  straight line (EightBitTransit):    x = v (t - t_ref) + u, y = b_ref + w
Blocked flux = sum_p weight_p I(mu_p) / Ftot. Each point is treated as an h x h cell: at the stellar limb the covered
fraction is used and I(mu) is averaged across the covered strip (removes the limb-crossing error of point sampling).
"""
import numpy as np


class Orbit:
    """Path of the occulter reference point.

    Orbit(P=..., a=..., b=0, t_ref=0)  -> curved circular orbit (a in R*, P in days)
    Orbit(v=..., b=0, t_ref=0)          -> straight line at constant speed v (R*/day), as in EightBitTransit
    """

    def __init__(self, P=None, a=None, v=None, b=0.0, t_ref=0.0):
        if a is not None and P is None and v is not None:
            P = 2 * np.pi * a / v
        self.curved = a is not None and P is not None
        if not self.curved and v is None:
            raise ValueError("give (P, a) for a curved orbit, or v for a straight path")
        self.P, self.a, self.b, self.t_ref = P, a, float(b), float(t_ref)
        self.v = 2 * np.pi * a / P if self.curved else float(v)

    def centre(self, t, b=None):
        """Reference-point sky position (x, y) and an 'in front of the star' mask, each shape (T, 1)."""
        b = self.b if b is None else b
        t = np.asarray(t, float)[:, None] - self.t_ref
        if self.curved:
            phi = 2 * np.pi * t / self.P
            return self.a * np.sin(phi), b * np.cos(phi), np.cos(phi) > 0
        return self.v * t, np.full_like(t, b), np.ones_like(t, bool)

    def __repr__(self):
        return (f"Orbit(P={self.P}, a={self.a}, b={self.b}, t_ref={self.t_ref})" if self.curved
                else f"Orbit(v={self.v}, b={self.b}, t_ref={self.t_ref}) [straight]")


def blocked_flux(t, u, w, wt, orbit, I, Ftot, b=None, cell=None):
    """Fraction of stellar flux blocked at times t by points (u, w) with weights wt (areas x opacity).

    cell: size h of the square cell each point represents (enables partial-coverage limb treatment).
    """
    x0, y0, front = orbit.centre(t, b)
    x = x0 + np.asarray(u)[None, :]
    y = y0 + np.asarray(w)[None, :]
    rho = np.sqrt(x ** 2 + y ** 2)
    val = I(np.sqrt(np.clip(1 - rho ** 2, 0, None)))
    if cell:
        d = 1 - rho
        val = val * (d >= cell / 2)
        part = np.abs(d) < cell / 2
        if part.any():
            dp = d[part]
            frac = dp / cell + 0.5
            top = dp + cell / 2
            Im = 0.0
            for k in range(4):
                rk = 1 - top * (k + 0.5) / 4
                Im = Im + I(np.sqrt(np.clip(1 - rk ** 2, 0, None))) / 4
            val[part] = frac * Im
    else:
        val = val * (rho < 1)
    val = val * front
    return (val * np.asarray(wt)[None, :]).sum(1) / Ftot


def grid_coords(N, M):
    """Pixel-centre coordinates of an N x M grid in EightBitTransit's convention at t = t_ref.

    Pixel width w = 2/N (N rows span the stellar diameter); row i centre y = 1 - w/2 - i w;
    column j centre x = (j + 1 - jmid) w with jmid = 1 + (M-1)/2.
    """
    w = 2.0 / N
    jmid = 1 + (M - 1) / 2.0
    yc = 1 - w / 2 - np.arange(N) * w
    xc = (np.arange(M) + 1 - jmid) * w
    return xc, yc, w


def pixel_design(t, N, M, orbit, I, Ftot, ss=6, b_ref=0.0):
    """Design matrix A (T, N*M): blocked flux per unit-opacity pixel, so 1 - A @ opacity.ravel() is the light curve.

    Pixels are supersampled ss x ss with partial limb coverage. The grid is a rigid body whose point at height
    y = b_ref follows the orbit (only matters for curved orbits; b_ref = 0 reproduces a grid centred on the star's
    equator, b_ref = fitted impact parameter is slightly more accurate for an off-centre planet).
    """
    xc, yc, w = grid_coords(N, M)
    sub = ((np.arange(ss) + 0.5) / ss - 0.5) * w
    A = np.zeros((len(t), N * M))
    wt = np.full(ss * ss, (w / ss) ** 2)
    for i in range(N):
        for j in range(M):
            U, W = np.meshgrid(xc[j] + sub, yc[i] - b_ref + sub)
            A[:, i * M + j] = blocked_flux(t, U.ravel(), W.ravel(), wt, orbit, I, Ftot, b=b_ref, cell=w / ss)
    return A


def supersample_times(t, exptime=None, n=1):
    """Return (t_fine, nsub): times expanded to n sub-exposures across exptime (days); average with
    `rebin(model, nsub)`."""
    t = np.asarray(t, float)
    if not exptime or n <= 1:
        return t, 1
    off = ((np.arange(n) + 0.5) / n - 0.5) * exptime
    return (t[:, None] + off[None, :]).ravel(), n


def rebin(y, nsub):
    return y if nsub == 1 else np.asarray(y).reshape(-1, nsub, *np.shape(y)[1:]).mean(1)
