"""Command line: closedshape LCFILE [options]  -> opacity image, fits (JSON) and plots in --out."""
import argparse, json, os
import numpy as np
from .image import TransitingImage
from .utils import load_lightcurve


def main(argv=None):
    p = argparse.ArgumentParser(prog="closedshape", description="Closed-shape shadow imaging of a transit light curve "
                                "(EightBitTransit-style). Input: text/CSV with columns time[d] flux [err].")
    p.add_argument("lcfile")
    p.add_argument("--N", type=int, default=20, help="grid rows (pixel width 2/N R*)")
    p.add_argument("--M", type=int, default=None, help="grid columns (default N)")
    p.add_argument("--a", type=float, default=None, help="a/R* (enables curved orbit; needs --P or --v)")
    p.add_argument("--P", type=float, default=None, help="period [d]")
    p.add_argument("--v", type=float, default=None, help="sky speed [R*/d]; alone = straight path like EightBitTransit")
    p.add_argument("--b", type=float, default=0.0, help="impact parameter for the grid reference (default 0)")
    p.add_argument("--t-ref", type=float, default=0.0)
    p.add_argument("--ld-law", default="uniform", choices=["uniform", "linear", "quadratic", "nonlinear", "poly"])
    p.add_argument("--ld", type=float, nargs="*", default=[], help="LD coefficients")
    p.add_argument("--exptime", type=float, default=None, help="exposure time in seconds (supersampled x5)")
    p.add_argument("--method", default="closed", choices=["closed", "clsq", "sphere", "ellipse", "fourier", "all"])
    p.add_argument("--normalize", action="store_true", help="divide by median of the outer 20%% of points")
    p.add_argument("--out", default="closedshape_out")
    a = p.parse_args(argv)
    t, f, e = load_lightcurve(a.lcfile)
    if a.normalize:
        oot = np.abs(t - a.t_ref) > np.quantile(np.abs(t - a.t_ref), 0.8)
        med = np.median(f[oot]); f, e = f / med, e / med
    ti = TransitingImage(N=a.N, M=a.M, LDlaw=a.ld_law, LDCs=a.ld, v=a.v, a=a.a, P=a.P, b=a.b, t_ref=a.t_ref, t_arr=t,
                         fluxes=f, errors=e, exptime=None if a.exptime is None else a.exptime / 86400.,
                         supersample=5 if a.exptime else 1)
    os.makedirs(a.out, exist_ok=True)
    methods = ["sphere", "ellipse", "fourier", "closed"] if a.method == "all" else [a.method]
    summary = {"orbit": repr(ti.orbit), "N": ti.N, "M": ti.M, "n_points": len(t)}
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for m in methods:
        g = ti.invertLC(method=m)
        lc = ti.gen_LC()[0] if m in ("closed", "clsq") else ti.fits[m]["model_LC"]
        s = {"rms_ppm": float(np.std(lc - f) * 1e6), "chi2": float(np.sum(((lc - f) / e) ** 2))}
        if m in ti.fits:
            r = ti.fits[m]
            s.update({k: float(r[k]) for k in ("r", "r_area", "f", "sig_f", "b", "chi2", "dchi2_vs_sphere", "aspect") if k in r})
        summary[m] = s
        np.savetxt(os.path.join(a.out, f"opacitymat_{m}.txt"), g, fmt="%.4f")
        ti.plot_grid(save=True, filename=os.path.join(a.out, f"opacitymat_{m}.png")); plt.close("all")
    ti.plot_LC(save=True, filename=os.path.join(a.out, "lightcurve.png")); plt.close("all")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
