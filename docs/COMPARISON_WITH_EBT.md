# How `closedshape` relates to EightBitTransit (EBT)

This page explains, in plain language, how this package differs from the original shadow-imaging code
[EightBitTransit](https://github.com/esandford/EightBitTransit) (Sandford & Kipping 2019). Every statement about EBT
refers to a function in upstream EBT (commit `51cb4c0`, files `EightBitTransit/*.pyx` and `examples/*.ipynb`). Every
statement about this package refers to a function in `src/closedshape/` at version 0.2.0. In 0.2.0 `closed` became
connected-only; the 0.1.0 version, commit `208d202`, is now called `closed-regularized`.

**Short version:**
- `closed` is a pixel inversion, like EBT's. It uses the same grid layout and the same bounded `[0, 1]`
  least-squares solver as EBT's constrained-least-squares (CLSQ) example. Compared with EBT CLSQ it changes:
  1. the forward model (curved orbit, limb darkening evaluated per sub-pixel, exposure smearing, noise weighting);
  2. no up/down folding: the full grid is solved;
  3. **one physical rule:** opacity may be non-zero only inside one connected region of pixels. It has no smoothness
     term, no compactness term, no sphere-fit centring and (by default) no hole filling.
- `closed-regularized` is an **opt-in add-on**, and it is the 0.1.0 `closed`. It is `closed` plus a smoothness
  penalty, a pull towards a centre (set by a sphere fit) and hole filling.
- `closed-compact` (0.2.1) is a **shortcut** for `closed-regularized` with the smoothness weight set to 0. It is `closed`
  plus the pull towards the sphere-fit centre and hole filling. It uses the same code, with no new algorithm.
- `clsq` in this package is the **EBT-like baseline**. It is plain bounded least squares on the full grid, using our
  forward model, with no folding and no regularisation.
- `sphere`, `ellipse` and `fourier` are **parametric outlines we added**. They are not part of EBT, and this page
  only mentions them where `closed-regularized` depends on them (it uses a sphere fit to choose its centre).
- **No EBT source code is included or modified in this package.** Everything in `src/closedshape/` is new code that
  copies EBT's grid and interface conventions. The only EBT code we patched is a local copy of the EBT GPU fork,
  which we used to make the baseline comparison figures. Those patches are listed in
  [`ebt_gpu_baseline_patches.diff`](ebt_gpu_baseline_patches.diff) and are not part of the package.

---

## 1. What each EBT inversion method does

All EBT methods use the same grid. It has N rows and M columns of square pixels of width `w = 2/N` stellar radii.
They use the same forward model (section 3) and a straight-line path at constant speed `v`. The row in the middle of
the grid (the "midplane") moves along the transit chord. A pixel above the chord and its mirror image below the chord
block the same starlight at every moment, because the star is symmetric and the path is straight. So a light curve
cannot tell "above" from "below". Each EBT method handles this **flip (mirror) degeneracy** by forcing an up/down
symmetric answer. They differ in how they do that.

### SART (simultaneous algebraic reconstruction): `inversion.pyx: simultaneous_ART`, called from `invertLC(method="SART")`
- **How opacity is spread:** an iterative tomography solver. At each of `n_iter` iterations, every pixel's opacity is
  nudged in proportion to how much that pixel overlaps the star at the times where the model is wrong. The solver
  uses the normal equations (`Asquare = A.T @ A`, `RHS = A.T @ (1 - obsLC)`). The data errors `obsLCerr` are only used
  to report χ², not to weight the fit. Opacities are not bounded during the iterations.
- **Midplane / symmetry:** before solving, `invertLC` keeps only the top half of the design matrix
  (`halfAreas = raveledareas[:, 0:(Nhalf*M)]`, comment "take only half of area matrix to avoid dealing with flip
  degeneracy"). After solving, it copies the top half onto the bottom half and divides the whole grid by 2
  (`raveledtau = raveledtau/2.`). Each mirror pair therefore shares the opacity equally. For odd N the middle row is
  also divided by 2, even though it has no mirror partner. This is what the code does; we did not check whether it
  was intended.
- **After solving:** `wedgeRearrange` and `wedgeNegativeEdge` move opacities that are above 1 or below 0 into
  neighbouring rows. They use blocks of pixels with the "same transit duration" (the *wedge degeneracy*). The result is
  rounded to 2 decimals.

### Arc averaging: `makeArcBasisAverage`, `invertLC(method="arcAvg")`
- **How opacity is spread:** at every time step it computes the change in flux, `delta_fluxes`. It finds the pixels
  that are crossing the stellar limb at that moment (`delta_areas != 0`): entering the star (ingress) when flux drops,
  leaving it (egress) when flux rises. It then **spreads that flux change evenly along the arc** of limb pixels. Each
  pixel gets `avg_opacity * area / sines`, where `sines` is a fixed geometric factor that depends on the row. This
  gives one candidate grid per time step. The final image is the **mean of all candidate grids**
  (`np.mean(avgBasis, axis=0)`).
- **Midplane / symmetry:** the image is then averaged with its mirror image (`(arcAvg + arcAvg[::-1,:])/2.`).
- **After solving:** `renormBasis` rescales the whole image so that the model's deepest point equals the observed
  transit depth, then clips the opacities to [0, 1].

### dF/dt parsimony: `makeArcBasisParsimony`, `invertLC(method="arcPars")`
- **How opacity is spread:** this is also done arc by arc, at each time step. It is greedy: it keeps switching on the
  limb pixel with the largest change in overlap area, at opacity 1 or 0.5, until the summed change matches the
  observed change in flux, and it never goes past it ("most parsimonious" = as few pixels as possible). The final image
  is again the mean over time steps.
- **Midplane / symmetry:** each pixel is switched on **together with its mirror pixel**
  (`max_area_mirror_i = N - max_area_i - 1`, with `mirrorfac = 2`). The mean image is then mirror-averaged again.
- **After solving:** `renormBasis` is applied twice.

### Arc combinatorics: `makeArcBasisCombinatoric`, `invertLC(method="arcComb")`
- **How opacity is spread:** at each time step it tries **every combination** of 0.5-opacity "units" on the top-half
  limb pixels (`itertools.combinations`). It keeps the combination whose model flux changes (`trial_delta_fluxes`)
  best match the observed flux changes. The score is summed over the **whole** light curve: the code computes a time
  window around each step (`delta_slice_mask`) but never uses it. The final image is the mean over time steps.
- **Midplane / symmetry:** every unit is placed on the top-half pixel **and** on its mirror (`northern_i` /
  `southern_i`). The image is mirror-averaged again and `renormBasis` is applied twice. Trial grids are passed through
  `foldOpacities`, which moves opacity from the bottom half onto the top half, capped at 1 per pixel.
- Note: inside this function the local `v` is overwritten with `v = 4./tEvent`, the duration of the event. It only
  feeds into the unused window.

### Constrained least squares (CLSQ): `examples/constrained-least-squares-examples.ipynb`, `examples/examples.ipynb`
- CLSQ is **not one of the methods in `invertLC`**. It lives in the example notebooks, with helper functions
  `symmetrize_design_matrix` and `unfold_opacity_map` in `inversion.pyx` (credited there to Kirill Tchernyshyov).
- **How opacity is spread:** a single bounded least-squares solve,
  `lsq_linear(sym_design_matrix, 1. - overlappingObsLC, bounds=[0., 1.])`. It finds the opacities between 0 and 1 that
  best reproduce the light curve. The fit is **unweighted**: errors are not used.
- **Midplane / symmetry:** `symmetrize_design_matrix` adds each top row's column to its mirror row's column, so a
  mirror pair is **one unknown**. For odd N the middle row counts once. `unfold_opacity_map` copies the answer to both
  halves, so the image is exactly up/down symmetric.
- **After solving:** the notebook sets the first few columns to zero by hand, with a number that depends on `v`
  (`clsq[:,0:6]`, `0:12`, `0:20`). The comment says these edge pixels "can sometimes artificially blow up".

### What every EBT method has in common
- A straight path at constant speed `v` (`cGridFunctions.pyx: positions`, `x = xref + (t - tref)*v`).
- Only times when the grid overlaps the star are used (`positions` returns `overlappingTimes`).
- The forward model in `TransitingImage.gen_LC`. For any limb darkening other than uniform, it calls `LDfluxsmall`,
  a small-planet approximation that treats each pixel as a tiny disk of the same area.
- No exposure-time smearing.
- No uncertainties on the image.
- The result is always up/down symmetric.

---

## 2. Side-by-side table

| | EBT SART | EBT arcAvg / arcPars / arcComb | EBT CLSQ (notebook) | **closedshape `clsq`** (baseline) | **closedshape `closed`** (default) | `closed-regularized` (opt-in add-on) |
|---|---|---|---|---|---|---|
| **Shared with EBT** | – | – | – | Grid layout (`grid_coords` = EBT `positions`), opacity in [0, 1], `lsq_linear` solver, `TransitingImage`/`gen_LC`/`invertLC` names, LD law names | Same as `clsq` | Same as `clsq` |
| **Unknowns** | N/2 × M (top half) | N × M, but filled in mirror pairs | ⌈N/2⌉ × M mirror pairs | N × M (full grid) | N × M; only pixels inside one connected region may be non-zero | Same as `closed` |
| **Constraints / priors** | None (wedge clean-up afterwards) | Flux change matched arc by arc; parsimony = fewest pixels; combinatorics = best 0.5-unit combination; mean over time steps | Bounds [0, 1] only | Bounds [0, 1] only | Bounds [0, 1] + **one connected region**. Nothing else | `closed` + **smoothness** (`lam=0.3`) + **compactness** (`mu=10`, pull towards a centre) + holes filled |
| **Symmetry / folding** | Top half solved, mirrored, halved | Mirror pairs set together, then mirror-averaged | Folded design matrix → mirror-symmetric | **None**; the full grid is solved | **None imposed.** If the mirror copies are separate patches, the one with more opacity is kept (an exact tie goes to the upper half); if they touch at the chord, both stay | **None imposed.** The centre is at the sphere fit's b ≥ 0, so the blob goes in the **upper half by convention** |
| **Connectedness** | Not enforced | Not enforced | Not enforced | Not enforced | Enforced on the **allowed region**: the largest 4-connected patch of pixels above 0.3 × max, grown by 1 pixel, then refit (twice). Holes are allowed unless `fill_holes=True`. Pixels inside the region can still come out at 0 | Same, with holes filled |
| **Opacity bounds** | Unbounded while solving; out-of-range values moved by wedge steps; rounded | Built from 0, 0.5, 1 units, then rescaled and clipped to [0, 1] | [0, 1] | [0, 1] | [0, 1] | [0, 1] |
| **Forward model** | EBT `gen_LC`: exact pixel–disk overlap area (`pixeloverlaparea`) × small-planet LD (`LDfluxsmall`) | same | same (`calculateLCdecrements`) | **New, not reused from EBT:** `forward.pixel_design`, with each pixel split into 6×6 sub-pixels, exact I(μ) per sub-pixel, and partial coverage at the stellar limb | same as `clsq` | same as `clsq` (for curved orbits the grid's reference height is the sphere-fit b) |
| **Orbit / velocity** | Straight line, speed `v` | Straight line, `v` | Straight line, `v` | Straight line with `--v` (same as EBT), **or** curved circular orbit with `--a --P` (`forward.Orbit`) | same as `clsq` | same as `clsq` |
| **Limb darkening** | uniform / linear / quadratic / nonlinear, all turned into 4-parameter nonlinear for `LDfluxsmall` | same | same | Same laws plus `poly`; I(μ) evaluated directly (`limbdark.make_intensity`) | same | same |
| **Exposure time** | Not modelled | Not modelled | Not modelled | `exptime` + `supersample` (`forward.supersample_times`, `rebin`) | same | same |
| **Use of data errors** | Only for χ² reporting | Only for χ² reporting | Not used | Residuals divided by σ (noise-weighted) | Noise-weighted | Noise-weighted |
| **Optimiser** | Fixed number of SART iterations | Greedy (pars) or exhaustive per-step search (comb); mean over steps | `scipy.optimize.lsq_linear` | `lsq_linear(..., lsmr_tol="auto", max_iter=3000)` | The same `lsq_linear`, run 3 times (initial + 2 refits inside the region) | The same, plus one parametric sphere fit (`scipy.optimize.least_squares`) to set the centre |
| **Uncertainties** | None | None | None | None | None | None |
| **Inputs required** | N, M, v, t_ref, t, flux, err, LD; `n_iter`, `initstate`, `filename` | N, M, v, t_ref, t, flux, err, LD | same | N, M, (v) or (a, P), t_ref, t, flux, err, LD; optional `exptime`, `ss` | Same as `clsq`; knobs `thr`, `n_iter`, `dilate`, `fill_holes` have defaults | Same, plus `lam` (`--smooth`), `mu` (`--compact`) |
| **Edge handling** | – | – | First columns zeroed by hand, depending on v | No zeroing by default (`zero_cols=0`) | No zeroing; edge pixels can be used if they are attached to the main patch | Compactness keeps opacity away from the grid edges |

---

## 3. Every assumption `closed` adds (compared with EBT CLSQ)

**Physical choices** (statements about the object or the orbit; wrong if the object isn't like this):
1. **One connected object.** Opacity is allowed only inside one connected patch of pixels (`pixel.closed_pixel`,
   `pixel.largest_blob`). Rings with gaps, detached tails, debris clouds or several bodies break this assumption. A
   ring or annulus *is* allowed (holes are not filled unless you pass `fill_holes=True` / `--fill-holes`, which adds a
   second assumption: "no holes").
2. **Rigid, non-rotating silhouette.** Its shape on the sky stays the same during the transit (`forward.blocked_flux`).
   EBT assumes this too.
3. **Path.** Either the straight line at speed `v` (the same as EBT) or, if you give a/R\* and P, a circular orbit
   (`Orbit`: `x = a sin φ`, `y = b cos φ`). The curved option is new, and it requires you to know a/R\* and P.
4. **Known limb darkening, period and a/R\*.** These are taken as given. EBT also takes LD as given.

**Not assumed by `closed`** (0.1.0 did assume these, and they are now only in `closed-regularized`):
- no smoothness (`lam`), no compactness (`mu`), no pull towards mid-transit, no sphere fit, no hole filling.
- `closed` rejects `lam`/`mu` (`TransitingImage.invertLC` raises an error and points you to `closed-regularized`).

**Numerical choices** (ways of solving the problem, with no physical meaning on their own):
1. `thr = 0.3`: pixels above 30 % of the brightest pixel form the candidate patch.
2. 4-connectivity (pixels touching side by side, not only at corners); the patch is grown by `dilate = 1` pixel and
   refit `n_iter = 2` times.
3. Mirror tie-break: when the two mirror copies are separate patches with *exactly* equal opacity, the upper one is
   kept (`largest_blob(prefer_upper=True)`). This is a convention.
4. Forward-model resolution: 6 × 6 sub-pixels per pixel (`ss=6`); 4-point averaging across the limb strip;
   exposure smearing with `supersample` sub-exposures (the CLI uses 5 when `--exptime` is given).
5. `lsq_linear` with `lsmr_tol="auto"` and `max_iter=3000`.
6. All requested times are used, rather than trimming to EBT's overlap window.

**Extra assumptions in the opt-in `closed-regularized`:**
- Physical: **compact and centred** on (x = 0, y = b_sphere), i.e. crossing at `t_ref` at the height a sphere fit
  prefers. **Smooth**. **No holes**. Up/down side set by the sphere fit's b ≥ 0 (upper half).
- Numerical: `lam = 0.3`, `mu = 10` (tuned on TESS-like simulations). Sphere-fit start values b0 ∈ {0.05, 0.3, 0.55},
  a 70 × 70 outline sampling, and box size `R = clip(2.4·√depth, 0.05, 1.2)`.

**Removed relative to EBT** (EBT assumptions that `closed` does not make):
- No forced up/down symmetry (EBT's folding and averaging).
- No straight-path requirement (optional).
- No small-planet limb-darkening approximation.
- No hand-zeroed edge columns.

---

## 4. Exactly what we changed

### 4a. Files in this package (all new code; no EBT file is copied)

| File / function | Status | Relation to EBT |
|---|---|---|
| `forward.py: grid_coords` | **New, same convention** | Reproduces EBT `cGridFunctions.positions` at `t = t_ref`: `w = 2/N`, row centre `y = 1 - w/2 - i*w`, column centre `x = (j + 1 - jmid)*w` |
| `forward.py: Orbit` | **New** | EBT has only `x = xref + (t - tref)*v`. `Orbit(v=...)` gives exactly that; `Orbit(P=, a=)` adds the curved path |
| `forward.py: blocked_flux`, `pixel_design` | **New (replaces EBT's forward model)** | Replaces `pixeloverlaparea` + `LDfluxsmall` + `calculateLCdecrements`. Sub-pixel I(μ), partial coverage at the limb, works with a curved orbit |
| `forward.py: supersample_times`, `rebin` | **New** | EBT has no exposure smearing |
| `limbdark.py: make_intensity` | **New, same law names** | EBT turns linear/quadratic into nonlinear coefficients and uses analytic annulus integrals; we evaluate I(μ) directly and integrate the total flux numerically |
| `pixel.py: clsq` | **New, EBT-like** | The same `lsq_linear(bounds=(0, 1))` as the EBT CLSQ notebook, but on the **full** grid (no `symmetrize_design_matrix`/`unfold_opacity_map`), **noise-weighted**, with optional penalties (off by default) |
| `pixel.py: largest_blob`, `closed_pixel` | **New** | No EBT counterpart. The connected-region rule (`closed`) |
| `pixel.py: gradient_matrix`, `closed_pixel_regularized` | **New, opt-in add-on** | No EBT counterpart. Smoothness + compactness + hole filling (`closed-regularized`; the 0.1.0 `closed`) |
| `shapes.py` (sphere / ellipse / fourier) | **New, not in EBT** | Parametric outlines. `closed-regularized` uses `fit_sphere` to pick its centre; `closed` does not use it |
| `image.py: TransitingImage`, `invertLC` | **New, EBT-style interface** | Same names and argument order as EBT's `TransitingImage` and `inversion.invertLC`, but the methods are `closed`/`clsq`/`closed-regularized`/`sphere`/`ellipse`/`fourier`. `gen_LC` returns all times, not only the overlap times |
| `cli.py`, `utils.py` | **New** | No EBT counterpart |
| SART, arcAvg, arcPars, arcComb, wedge functions, `foldOpacities`, `bruteForceSearch` | **Not included** | Use EBT itself for these |

### 4b. The key code differences, side by side

**Symmetry: EBT CLSQ folds the grid; we don't.**
```python
# EBT, examples/examples.ipynb (CLSQ cell)
sym_design_matrix = symmetrize_design_matrix(LCdecs).reshape([sym_N * M, len(overlapTimes)]).T
out_bounded = lsq_linear(sym_design_matrix, 1.-overlappingObsLC, bounds=[0., 1.])
clsq = unfold_opacity_map(out_bounded['x'].reshape([sym_N, M]), N)      # copied to both halves
clsq[:,0:12] = 0.                                                        # (for v = 1.6) edge columns zeroed by hand

# closedshape, src/closedshape/pixel.py: clsq
y  = (1 - flux) / sg                     # noise-weighted
Aw = A / sg[:, None]                     # full N*M design matrix, no folding
out = lsq_linear(As, y, bounds=(0, 1), lsmr_tol="auto", max_iter=3000)
```

**Connected-region rule: new in `closed_pixel` (default `closed`).**
```python
# closedshape, src/closedshape/pixel.py: closed_pixel   (no lam, no mu, no centre)
g = clsq(A, flux, sig, N, M)                                        # plain CLSQ, full grid
for _ in range(n_iter):
    k = largest_blob(g, thr, fill_holes=fill_holes)                 # one 4-connected patch (holes kept by default)
    keep = ndimage.binary_dilation(k, iterations=dilate)
    g = clsq(A, flux, sig, N, M, support=keep)                      # refit, zero outside the patch

# closedshape, src/closedshape/image.py: TransitingImage.invertLC
elif method == "closed":
    if "lam" in kw or "mu" in kw:
        raise ValueError("'closed' has no smoothness/compactness terms; use method='closed-regularized'")
    self.opacitymat, self.support = closed_pixel(self.design(), f, e, self.N, self.M, return_support=True, **kw)
```

**Opt-in add-on: `closed_pixel_regularized` (`closed-regularized`, the 0.1.0 `closed`).**
```python
# closedshape, src/closedshape/pixel.py: closed_pixel_regularized
g = clsq(A, flux, sig, N, M, lam=lam, mu=mu, centre=centre)          # + smoothness + compactness
...  largest_blob(g, thr, fill_holes=True) ... clsq(..., lam=lam, mu=mu, centre=centre, support=keep)
# image.py: centre = (0.0, sph["b"]) from a sphere fit
```

**Forward model: EBT's small-planet LD vs per-sub-pixel I(μ).**
```python
# EBT, cGridFunctions.pyx: LDfluxsmall  (per pixel; r = radius of a disk with the pixel's area)
r = w/(pi**0.5)
I0[i] = pi*(Ar_occ[i]/Ar_ann)*(Fl_ann/Ftot)      # mean intensity of the annulus [S-r, S+r] around the pixel

# closedshape, forward.py: pixel_design / blocked_flux  (each pixel split into ss x ss sub-pixels)
val = I(np.sqrt(np.clip(1 - rho ** 2, 0, None)))  # intensity at every sub-pixel
# sub-pixels that straddle the limb: covered fraction x I(mu) averaged over the covered strip
```

**Path: EBT's straight line vs optional circular orbit.**
```python
# EBT, cGridFunctions.pyx: positions
pos[k,:,j-1,0] = xref[j-1] + (t[k] - tref)*v           # y fixed for every row

# closedshape, forward.py: Orbit.centre
if self.curved:  return a*sin(phi), b*cos(phi), cos(phi) > 0
return self.v * t, b                                   # straight path = EBT
```

### 4c. EBT code we patched (baseline only, not in this package)
For the "EightBitTransit CLSQ" baseline in the benchmark and example figures, we ran the EBT GPU fork
(github.com/d-giles/EightBitTransitGPU, commit `f15fadd`) on NumPy 2. That needed small fixes in
`EightBitTransit/inversion.py`; the full diff is in [`ebt_gpu_baseline_patches.diff`](ebt_gpu_baseline_patches.diff):
- removed invalid Cython-style `np.ndarray(np.int64_t, ...)` declarations;
- `np.in1d` → `np.isin`; `.astype(np.int)` → `.astype(int)`;
- removed a stray `RHS` line in `simultaneous_ART`;
- fixed `times` → `t_arr` in the `arcComb` branch of `invertLC`;
- added a guard `nOpacityUnits = max(0, min(nOpacityUnits, nLimbPixelSpaces))` in `makeArcBasisCombinatoric`. This
  one line changes behaviour on noisy data, where unpatched EBT raises an error instead.

None of these patches touch the forward model or the CLSQ logic. The baseline CLSQ solve itself followed the EBT
notebook: folded design matrix, `lsq_linear` with bounds [0, 1], unfolded, first two columns zeroed. It lives in the
analysis scripts that accompany the paper, not in this package.

---

## 5. Which method should I compare with what?
- To see the effect of **only the better forward model**: compare EBT CLSQ with closedshape `clsq` using `--v` (a
  straight path). The remaining differences are no folding, noise weighting, the LD evaluation and exposure smearing.
- To see the effect of **the connected-region rule alone**: compare closedshape `clsq` with closedshape `closed`.
  To see what smoothness and compactness add on top, compare `closed` with `closed-regularized`. Run all of them with the
  same settings.
- `sphere`/`ellipse`/`fourier` answer a different question ("which simple outline fits?"). They are not shadow-image
  inversions in the EBT sense.
