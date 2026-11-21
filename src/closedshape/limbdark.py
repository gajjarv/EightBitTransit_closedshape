"""Stellar limb-darkening profiles I(mu) normalised to I(1)=1, and the total disk flux."""
import numpy as np

LAWS = ("uniform", "linear", "quadratic", "nonlinear", "poly")


def make_intensity(law="uniform", coeffs=None):
    """Return (I, Ftot): I(mu) callable and Ftot = integral of I over the unit stellar disk.

    law: 'uniform' | 'linear' [u] | 'quadratic' [u1, u2] | 'nonlinear' [c1..c4] (Claret 2000) |
         'poly' [u1..uK], I = 1 - sum_k u_k (1-mu)^k (squishyplanet convention).
    The names and coefficient conventions of the first four match EightBitTransit's ``LDlaw``/``LDCs``.
    """
    law = law.lower()
    c = np.atleast_1d(np.asarray(coeffs if coeffs is not None else [], float))
    if law == "uniform":
        I = lambda mu: np.ones_like(np.asarray(mu, float))
    elif law == "linear":
        I = lambda mu: 1 - c[0] * (1 - mu)
    elif law == "quadratic":
        I = lambda mu: 1 - c[0] * (1 - mu) - c[1] * (1 - mu) ** 2
    elif law == "nonlinear":
        c1, c2, c3, c4 = c
        I = lambda mu: 1 - c1 * (1 - mu ** 0.5) - c2 * (1 - mu) - c3 * (1 - mu ** 1.5) - c4 * (1 - mu ** 2)
    elif law == "poly":
        I = lambda mu: 1 - sum(c[k] * (1 - mu) ** (k + 1) for k in range(len(c)))
    else:
        raise ValueError(f"unknown LD law {law!r}; choose from {LAWS}")
    r = np.linspace(0, 1, 20001)
    Ftot = np.trapezoid(2 * np.pi * r * I(np.sqrt(1 - r ** 2)), r)
    return I, Ftot
