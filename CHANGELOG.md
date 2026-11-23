# Changelog

## 0.2.0 (2026-11-23)
- `--method closed` is now **connected-only**: plain bounded, noise-weighted least squares restricted to one connected
  region of pixels (largest patch above 0.3 × max, grown by 1 pixel, refit, twice). There is no smoothness term, no
  compactness term and no sphere-fit centring. Hole filling is off by default (`fill_holes=True` / `--fill-holes`).
- The 0.1.0 behaviour (smoothness `lam=0.3`, compactness `mu=10` centred on the sphere-fit b, holes filled) moves to
  the opt-in `--method closed-regularized` (`closed_pixel_regularized`; `--smooth`, `--compact`). For the same inputs
  it gives exactly the 0.1.0 `closed` result.
- `TransitingImage.invertLC(method="closed")` rejects `lam`/`mu` and stores the allowed region in `ti.support`.
- `--method all` now also runs `closed-regularized`.

## 0.1.0 (2026-11-20)
- First release: curved-orbit forward model with exact limb darkening and partial limb coverage; weighted CLSQ;
  closed-pixel inversion; sphere / ellipse / Fourier closed-outline fits; EightBitTransit-style `TransitingImage`
  and `invertLC`; `closedshape` command-line tool; tests and quickstart example.
