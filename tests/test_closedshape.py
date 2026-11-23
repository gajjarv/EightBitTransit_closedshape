import numpy as np
import pytest
from scipy import ndimage
import closedshape as cs


def uniform_overlap(z, p):
    """Fraction of a uniform unit disk covered by a disk of radius p at separation z."""
    z = np.asarray(z, float); out = np.zeros_like(z)
    full = z <= 1 - p; out[full] = p ** 2
    part = (z > 1 - p) & (z < 1 + p)
    zz = z[part]
    k0 = np.arccos(np.clip((p ** 2 + zz ** 2 - 1) / (2 * p * zz), -1, 1))
    k1 = np.arccos(np.clip((1 - p ** 2 + zz ** 2) / (2 * zz), -1, 1))
    out[part] = (p ** 2 * k0 + k1 - 0.5 * np.sqrt(np.clip(4 * zz ** 2 - (1 + zz ** 2 - p ** 2) ** 2, 0, None))) / np.pi
    return out


def test_sphere_uniform_vs_analytic_straight():
    I, F = cs.make_intensity("uniform")
    v = 20.0; t = np.linspace(-0.07, 0.07, 301)
    orb = cs.Orbit(v=v)
    m = cs.ShapeModel(t, orb, I, F, R=0.25, n=70)
    z = np.hypot(v * t, 0.3)
    ref = 1 - uniform_overlap(z, 0.1)
    err70 = np.max(np.abs(m.flux([np.log(0.1), 0.0], 0.3) - ref))
    m2 = cs.ShapeModel(t, orb, I, F, R=0.25, n=140)
    err140 = np.max(np.abs(m2.flux([np.log(0.1), 0.0], 0.3) - ref))
    assert err70 < 50e-6 and err140 < 0.6 * err70   # ~3e-3 of the depth, converging with sampling


def test_sphere_nonlinear_vs_batman_curved():
    batman = pytest.importorskip("batman")
    c = [0.15, 1.2, -1.1, 0.36]
    t = np.linspace(-0.07, 0.07, 201)
    p = batman.TransitParams(); p.t0 = 0; p.per = 1.0; p.rp = 0.1; p.a = 3.0; p.inc = np.degrees(np.arccos(0.35 / 3))
    p.ecc = 0; p.w = 90; p.limb_dark = "nonlinear"; p.u = c
    ref = batman.TransitModel(p, t).light_curve(p)
    I, F = cs.make_intensity("nonlinear", c)
    m = cs.ShapeModel(t, cs.Orbit(P=1.0, a=3.0), I, F, R=0.24, n=70)
    assert np.max(np.abs(m.flux([np.log(0.1), 0.0], 0.35) - ref)) < 40e-6


def test_curved_differs_from_straight():
    I, F = cs.make_intensity("quadratic", [0.4, 0.2])
    t = np.linspace(-0.07, 0.07, 201)
    a = cs.ShapeModel(t, cs.Orbit(P=1.0, a=3.0), I, F, R=0.25).flux([np.log(0.1), 0], 0.35)
    s = cs.ShapeModel(t, cs.Orbit(v=2 * np.pi * 3.0), I, F, R=0.25).flux([np.log(0.1), 0], 0.35)
    assert np.max(np.abs(a - s)) > 300e-6   # the curvature EightBitTransit ignores is ~1e3 ppm at a/R*=3


def _sim(f=0.5, ra=0.2, b=0.35, sig=200e-6, seed=0, n=160):
    I, F = cs.make_intensity("quadratic", [0.4, 0.2])
    t = np.linspace(-0.075, 0.075, n)
    m = cs.ShapeModel(t, cs.Orbit(P=1.0, a=3.0), I, F, R=2.4 * ra, n=80)
    y = m.flux([np.log(ra), cs.f_to_e(f)], b)
    return t, y + np.random.default_rng(seed).normal(0, sig, n) * (sig > 0), I, F


def test_ellipse_recovers_f_noiseless():
    t, y, I, F = _sim(sig=0)
    m = cs.ShapeModel(t, cs.Orbit(P=1.0, a=3.0), I, F, R=0.48, n=70)
    ell = cs.fit_ellipse(m, y, 200e-6)
    assert abs(ell["f"] - 0.5) < 0.03 and abs(ell["b"] - 0.35) < 0.02
    assert ell["chi2"] < 5 and cs.fit_sphere(m, y, 200e-6)["chi2"] > 100


def test_closed_connected_only():
    """Default 'closed': opacity only inside ONE connected region; no smoothness/compactness terms accepted."""
    t, y, I, F = _sim()
    ti = cs.TransitingImage(N=14, M=12, LDlaw="quadratic", LDCs=[0.4, 0.2], a=3.0, P=1.0, t_arr=t, fluxes=y, errors=200e-6)
    g = ti.invertLC(method="closed")
    assert ndimage.label(ti.support)[1] == 1           # the allowed region is a single connected patch
    assert np.all(g[~ti.support] == 0)                  # nothing outside it
    assert g.min() >= 0 and g.max() <= 1
    assert np.std(ti.gen_LC()[0] - y) < 400e-6
    assert "sphere" not in ti.fits                      # no sphere fit / centring in the default method
    import pytest
    with pytest.raises(ValueError):
        ti.invertLC(method="closed", lam=0.3)


def test_closed_regularized_single_blob():
    """Opt-in 'closed-regularized' (the 0.1.0 'closed'): one blob, holes filled, smooth + compact."""
    t, y, I, F = _sim()
    ti = cs.TransitingImage(N=14, M=12, LDlaw="quadratic", LDCs=[0.4, 0.2], a=3.0, P=1.0, t_arr=t, fluxes=y, errors=200e-6)
    g = ti.invertLC(method="closed-regularized")
    lab, n = ndimage.label(g > 1e-3)
    assert n == 1
    assert not ndimage.binary_fill_holes(g > 1e-3)[~(g > 1e-3)].any()   # no holes
    assert np.std(ti.gen_LC()[0] - y) < 400e-6


def test_closed_matches_regularized_with_zero_penalties():
    """closed_pixel == closed_pixel_regularized(lam=0, mu=0) apart from hole filling (fill_holes=True)."""
    t, y, I, F = _sim()
    ti = cs.TransitingImage(N=12, M=10, LDlaw="quadratic", LDCs=[0.4, 0.2], a=3.0, P=1.0, t_arr=t, fluxes=y, errors=200e-6)
    A = ti.design()
    g1 = cs.closed_pixel(A, y, 200e-6, 12, 10, fill_holes=True)
    g2 = cs.closed_pixel_regularized(A, y, 200e-6, 12, 10, lam=0.0, mu=0.0)
    assert np.allclose(g1, g2, atol=1e-6)


def test_transitingimage_ebt_style_roundtrip():
    """opacitymat -> gen_LC -> invertLC('clsq') reproduces the light curve (EightBitTransit-style workflow)."""
    N = M = 10
    truth = cs.pixelize_ellipse(N, M, 0.0, 0.0, 0.3, 0.3)
    t = np.linspace(-0.06, 0.06, 150)
    ti = cs.TransitingImage(opacitymat=truth, LDlaw="nonlinear", LDCs=[0.15, 1.2, -1.1, 0.36], v=20.0, t_ref=0.0, t_arr=t)
    lc, tt = ti.gen_LC(t)
    assert lc.shape == t.shape and lc.min() < 0.95
    ti2 = cs.TransitingImage(N=N, M=M, LDlaw="nonlinear", LDCs=[0.15, 1.2, -1.1, 0.36], v=20.0, t_arr=t, fluxes=lc, errors=1e-4)
    ti2.invertLC(method="clsq")
    assert np.std(ti2.gen_LC()[0] - lc) < 20e-6


def test_functional_invertLC_ebt_signature():
    t, y, I, F = _sim()
    g, ti = cs.invertLC(12, 12, 2 * np.pi * 3.0, 0.0, t, y, np.full_like(y, 200e-6), "closed",
                        LDlaw="quadratic", LDCs=[0.4, 0.2], a=3.0)
    assert g.shape == (12, 12) and ti.orbit.curved and abs(ti.orbit.P - 1.0) < 1e-9
