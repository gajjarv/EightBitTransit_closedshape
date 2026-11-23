"""Quickstart: EightBitTransit-style workflow with closedshape (runtime ~30-60 s).

1) Injection/recovery: an oblate occulter (f = 0.5) observed TESS-like; compare plain CLSQ, the closed-pixel
   inversion and the parametric closed-outline fits.
2) WASP-12b-style fit: a spherical planet with WASP-12b's parameters, 288 x 60-s points, 285 ppm noise,
   120-s exposures; how well can f be constrained at fixed a/R*?
   Pass --file path/to/fold.txt (columns t_from_mid[d] flux err) to fit a real fold instead.
"""
import sys, os, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import closedshape as cs

ap = argparse.ArgumentParser(); ap.add_argument("--file", default=None); args = ap.parse_args()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(42)

# ---------------- 1. injection / recovery ----------------
P, a, b, ra, f_true, sig = 1.0, 3.0, 0.35, 0.15, 0.5, 300e-6
LD = [0.15, 1.2, -1.1, 0.36]                       # nonlinear (Claret) coefficients, TESS-like
I, F = cs.make_intensity("nonlinear", LD)
t = np.arange(-0.075, 0.075, 60 / 86400.)
truth_model = cs.ShapeModel(t, cs.Orbit(P=P, a=a), I, F, R=2.4 * ra, n=100)
flux = truth_model.flux([np.log(ra), cs.f_to_e(f_true)], b) + rng.normal(0, sig, len(t))
N, M = 20, 15
truth_img = cs.pixelize_ellipse(N, M, 0.0, b, ra / np.sqrt(1 - f_true), ra * np.sqrt(1 - f_true))

# EightBitTransit-style: TransitingImage(..., LDlaw, LDCs, v / a,P, t_ref, t_arr) then invert
ti = cs.TransitingImage(N=N, M=M, LDlaw="nonlinear", LDCs=LD, a=a, P=P, t_ref=0.0, t_arr=t, fluxes=flux, errors=sig)
imgs = {}
for method in ["clsq", "closed", "closed-regularized", "ellipse", "fourier"]:
    imgs[method] = ti.invertLC(method=method).copy()
    mt = cs.image_metrics(imgs[method], truth_img)
    print(f"[injection] {method:8s} corr={mt['corr']:.2f} IoU={mt['iou']:.2f} f_moment={mt['f_moment']:+.2f}")
for k in ["ellipse", "fourier"]:
    r = ti.fits[k]
    print(f"[injection] {k:8s} f = {r['f']:+.3f} +/- {r['sig_f']:.3f} (true {f_true}), b = {r['b']:.3f}, "
          f"dchi2 vs sphere = {r['dchi2_vs_sphere']:.1f}")
LABELS = {"clsq": "EightBitTransit CLSQ"}  # our re-implementation of EBT's CLSQ
fig, axs = plt.subplots(1, 6, figsize=(14, 3))
for ax, (k, g) in zip(axs, [("truth", truth_img)] + list(imgs.items())):
    ax.imshow(g, cmap="Greys", vmin=0, vmax=1, interpolation="none"); ax.set_title(LABELS.get(k, k)); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(out, "injection_recovery.png"), dpi=110)

# ---------------- 2. WASP-12b-style ----------------
P12, a12, c12 = 1.0914185, 3.036, [0.15, 1.2, -1.1, 0.36]
if args.file:
    t2, f2, e2 = cs.load_lightcurve(args.file)
    oot = np.abs(t2) > 0.075; f2 = f2 / np.median(f2[oot])
else:
    t2 = (np.arange(288) - 143.5) * 60 / 86400.
    I2, F2 = cs.make_intensity("nonlinear", c12)
    tf, nsub = cs.supersample_times(t2, 120 / 86400., 5)
    m = cs.ShapeModel(tf, cs.Orbit(P=P12, a=a12), I2, F2, R=0.3, n=80, nsub=nsub)
    f2 = m.flux([np.log(0.1162), 0.0], 0.357) + rng.normal(0, 285e-6, len(t2)); e2 = np.full_like(t2, 285e-6)
w12 = cs.TransitingImage(N=20, M=15, LDlaw="nonlinear", LDCs=c12, a=a12, P=P12, t_arr=t2, fluxes=f2, errors=e2,
                         exptime=120 / 86400., supersample=5)
sph = w12.fit("sphere"); ell = w12.fit("ellipse")
print(f"[WASP-12b{'' if args.file else '-style sim'}] sphere r={sph['r']:.4f} b={sph['b']:.3f} chi2={sph['chi2']:.1f}/{len(t2)}; "
      f"ellipse f = {ell['f']:+.3f} +/- {ell['sig_f']:.3f}, dchi2 = {ell['dchi2_vs_sphere']:.2f}")
w12.invertLC(method="closed"); w12.plot_LC(save=True, filename=os.path.join(out, "wasp12b_style_lc.png"))
w12.plot_grid(save=True, filename=os.path.join(out, "wasp12b_style_closed_image.png"))
print("figures in", out)
