"""EightBitTransit-style front end."""
import numpy as np
from .limbdark import make_intensity
from .forward import Orbit, pixel_design, supersample_times, rebin
from .pixel import clsq, closed_pixel
from .shapes import ShapeModel, fit_sphere, fit_ellipse, fit_fourier


class TransitingImage:
    """Shadow-imaging object, modelled on EightBitTransit's ``TransitingImage``.

    Parameters (EightBitTransit names where they exist)
    ----------
    opacitymat : (N, M) array, optional   opacity grid (0..1). Either this or N (and M) must be given.
    N, M       : grid shape if no opacitymat (M defaults to N). Pixel width = 2/N stellar radii.
    LDlaw, LDCs: 'uniform' | 'linear' | 'quadratic' | 'nonlinear' (as EightBitTransit) or 'poly'.
    v          : sky speed in R*/day. With v alone the path is a straight line (EightBitTransit behaviour).
    a, P       : a/R* and period [d]. If given (or a with v) the occulter follows the curved circular orbit.
    b          : impact parameter used for the orbit's reference point (grid rows are absolute heights at
                 mid-transit; parametric fits fit b themselves).
    t_ref      : mid-transit time [d].
    t_arr      : times [d].
    fluxes, errors : observed normalised light curve and 1-sigma errors (needed for inversion / fits).
    exptime, supersample : exposure time [d] and number of sub-exposures for integration smearing.
    ss         : sub-pixel supersampling per axis for the pixel design matrix.
    """

    def __init__(self, opacitymat=None, N=None, M=None, LDlaw="uniform", LDCs=None, v=None, a=None, P=None,
                 b=0.0, t_ref=0.0, t_arr=None, fluxes=None, errors=None, exptime=None, supersample=1, ss=6):
        if opacitymat is not None:
            self.opacitymat = np.asarray(opacitymat, float)
            self.N, self.M = self.opacitymat.shape
        elif N is not None:
            self.N, self.M = int(N), int(M if M is not None else N)
            self.opacitymat = None
        else:
            raise ValueError("give opacitymat or N (and M)")
        if t_arr is None:
            raise ValueError("t_arr is required (as in EightBitTransit)")
        self.LDlaw, self.LDCs = LDlaw, LDCs
        self.I, self.Ftot = make_intensity(LDlaw, LDCs)
        self.orbit = Orbit(P=P, a=a, v=v, b=b, t_ref=t_ref)
        self.v, self.t_ref = self.orbit.v, t_ref
        self.t_arr = np.asarray(t_arr, float)
        self.fluxes = None if fluxes is None else np.asarray(fluxes, float)
        self.errors = None if errors is None else np.broadcast_to(np.asarray(errors, float), self.t_arr.shape).copy()
        self.exptime, self.supersample, self.ss = exptime, int(supersample), ss
        self._A, self._A_bref = None, None
        self.fits = {}

    # ---------------- forward model ----------------
    def design(self, b_ref=None):
        """(T, N*M) design matrix on the observed times (cached)."""
        b_ref = self.orbit.b if b_ref is None else b_ref
        if self._A is None or self._A_bref != b_ref:
            tf, nsub = supersample_times(self.t_arr, self.exptime, self.supersample)
            self._A = rebin(pixel_design(tf, self.N, self.M, self.orbit, self.I, self.Ftot, self.ss, b_ref), nsub)
            self._A_bref = b_ref
        return self._A

    def gen_LC(self, t_arr=None):
        """Light curve of the current opacitymat. Returns (LC, t_arr) like EightBitTransit's gen_LC.
        Unlike EightBitTransit, all requested times are returned (no trimming to the grid-overlap window)."""
        if self.opacitymat is None:
            raise ValueError("no opacitymat yet; call invertLC() or pass opacitymat")
        if t_arr is not None and not np.array_equal(np.asarray(t_arr, float), self.t_arr):
            self.t_arr = np.asarray(t_arr, float); self._A = None
        return 1 - self.design() @ self.opacitymat.ravel(), self.t_arr

    # ---------------- inversions ----------------
    def _data(self, fluxes, errors):
        f = self.fluxes if fluxes is None else np.asarray(fluxes, float)
        e = self.errors if errors is None else np.broadcast_to(np.asarray(errors, float), f.shape)
        if f is None:
            raise ValueError("no fluxes given")
        if e is None:
            e = np.full_like(f, 1.0)
        return f, e

    def shape_model(self, fluxes=None, R=None, n=70):
        f, _ = self._data(fluxes, None)
        R = R or float(np.clip(2.4 * np.sqrt(max(1 - f.min(), 1e-6)), 0.05, 1.2))
        tf, nsub = supersample_times(self.t_arr, self.exptime, self.supersample)
        return ShapeModel(tf, self.orbit, self.I, self.Ftot, R, n=n, nsub=nsub)

    def fit(self, kind="ellipse", fluxes=None, errors=None, K=3, prior=0.3, R=None, n=70):
        """Parametric closed-outline fit: kind = 'sphere' | 'ellipse' | 'fourier'. Returns a dict and stores it in
        self.fits[kind]; the corresponding rasterised opacity image is in result['opacitymat']."""
        f, e = self._data(fluxes, errors)
        m = self.shape_model(f, R, n)
        sph = self.fits.get("sphere") or fit_sphere(m, f, e)
        self.fits["sphere"] = sph
        if kind == "sphere":
            res = sph
        else:
            ell = self.fits.get("ellipse") or fit_ellipse(m, f, e, sph)
            self.fits["ellipse"] = ell
            res = ell if kind == "ellipse" else fit_fourier(m, f, e, ell, K=K, prior=prior)
        res["dchi2_vs_sphere"] = sph["chi2"] - res["chi2"]
        res["model_LC"] = m.flux(res["c"], res["b"])
        res["opacitymat"] = m.raster(res["c"], res["b"], self.N, self.M)
        self.fits[kind] = res
        return res

    def invertLC(self, fluxes=None, errors=None, method="closed", lam=0.3, mu=10.0, **kw):
        """Invert the light curve to an N x M opacity image and store it in self.opacitymat.

        method: 'closed'  - closed-shape pixel inversion (single connected hole-free blob; recommended)
                'clsq'    - plain weighted bounded least squares on the full grid (closest to EightBitTransit CLSQ)
                'sphere' | 'ellipse' | 'fourier' - parametric outline, rasterised onto the grid
        """
        f, e = self._data(fluxes, errors)
        if method in ("sphere", "ellipse", "fourier"):
            self.opacitymat = self.fit(method, f, e, **kw)["opacitymat"]
        elif method == "clsq":
            self.opacitymat = clsq(self.design(), f, e, self.N, self.M, **kw)
        elif method == "closed":
            sph = self.fits.get("sphere") or self.fit("sphere", f, e)
            A = self.design(b_ref=sph["b"] if self.orbit.curved else self.orbit.b)
            self.opacitymat = closed_pixel(A, f, e, self.N, self.M, lam=lam, mu=mu, centre=(0.0, sph["b"]), **kw)
        else:
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        return self.opacitymat

    # ---------------- plotting ----------------
    def plot_grid(self, save=False, filename=None, ax=None):
        import matplotlib.pyplot as plt
        ax = ax or plt.figure(figsize=(4, 4 * self.N / self.M)).gca()
        ax.imshow(self.opacitymat, cmap="Greys", vmin=0, vmax=1, interpolation="none",
                  extent=[-self.M / self.N - 0, self.M / self.N, -1, 1])
        ax.set_xlabel("x [R*] (direction of motion)"); ax.set_ylabel("y [R*]")
        if save:
            ax.figure.savefig(filename or "opacitymat.png", dpi=120, bbox_inches="tight")
        return ax

    def plot_LC(self, save=False, filename=None, ax=None):
        import matplotlib.pyplot as plt
        ax = ax or plt.figure(figsize=(6, 3.5)).gca()
        t = (self.t_arr - self.t_ref) * 24
        if self.fluxes is not None:
            ax.errorbar(t, (self.fluxes - 1) * 1e6, None if self.errors is None else self.errors * 1e6,
                        fmt=".", ms=2, color="0.6", lw=0.5, label="data")
        if self.opacitymat is not None:
            ax.plot(t, (self.gen_LC()[0] - 1) * 1e6, label=f"image ({getattr(self, 'method', 'given')})")
        for k, r in self.fits.items():
            if "model_LC" in r:
                ax.plot(t, (r["model_LC"] - 1) * 1e6, lw=0.8, label=f"{k} fit")
        ax.set_xlabel("hours from t_ref"); ax.set_ylabel("ppm"); ax.legend(fontsize=7)
        if save:
            ax.figure.savefig(filename or "lightcurve.png", dpi=120, bbox_inches="tight")
        return ax


def invertLC(N, M, v, t_ref, t_arr, obsLC, obsLCerr, method="closed", LDlaw="uniform", LDCs=(), a=None, P=None,
             b=0.0, exptime=None, supersample=1, **kwargs):
    """Functional entry point with EightBitTransit's ``inversion.invertLC`` argument order.

    EightBitTransit methods are 'SART', 'arcAvg', 'arcPars', 'arcComb'; here method is 'closed' (default), 'clsq',
    'sphere', 'ellipse' or 'fourier'. Pass a (and P, or v) to use the curved orbit; with v only the path is straight.
    Returns (opacitymat, TransitingImage) rather than EightBitTransit's multi-stage outputs.
    """
    ti = TransitingImage(N=N, M=M, LDlaw=LDlaw, LDCs=list(LDCs), v=v, a=a, P=P, b=b, t_ref=t_ref, t_arr=t_arr,
                         fluxes=obsLC, errors=obsLCerr, exptime=exptime, supersample=supersample)
    return ti.invertLC(method=method, **kwargs), ti
