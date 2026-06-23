"""Linear-Gaussian structural causal model helpers for the B01 confounding bundle.

Three standardized variables T (treatment), Z (confounder), Y (outcome) with
Z -> T, Z -> Y, T -> Y. The backdoor-adjusted effect of T on Y is the partial
regression coefficient of Y on T controlling for Z. We compute it two
independent ways - a closed form and a seeded simulation - so the dual-
verification gate can reject any item whose answer key the two disagree on.
"""

from __future__ import annotations

import math
import random


def partial_regression_coef(r_ty: float, r_tz: float, r_zy: float) -> float:
    """Standardized partial regression coefficient of Y on T, controlling Z."""
    return (r_ty - r_tz * r_zy) / (1 - r_tz**2)


def _cholesky3(c: list[list[float]]) -> list[list[float]]:
    L = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(c[i][i] - s, 0.0))
            else:
                L[i][j] = (c[i][j] - s) / L[j][j] if L[j][j] else 0.0
    return L


def is_psd(r_ty: float, r_tz: float, r_zy: float) -> bool:
    c = [[1.0, r_tz, r_ty], [r_tz, 1.0, r_zy], [r_ty, r_zy, 1.0]]
    L = _cholesky3(c)
    return all(L[i][i] > 1e-6 for i in range(3))


def simulate_partial_coef(
    r_ty: float, r_tz: float, r_zy: float, n: int = 20000, seed: int = 0
) -> float:
    """Recover the partial coefficient by sampling the SCM and running OLS."""
    # Variable order [T, Z, Y].
    c = [[1.0, r_tz, r_ty], [r_tz, 1.0, r_zy], [r_ty, r_zy, 1.0]]
    L = _cholesky3(c)
    rng = random.Random(seed)
    s_tt = s_tz = s_zz = s_ty = s_zy = 0.0
    for _ in range(n):
        z0, z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        t = L[0][0] * z0
        z = L[1][0] * z0 + L[1][1] * z1
        y = L[2][0] * z0 + L[2][1] * z1 + L[2][2] * z2
        s_tt += t * t
        s_tz += t * z
        s_zz += z * z
        s_ty += t * y
        s_zy += z * y
    # Solve the 2x2 normal equations for the coefficient on T.
    det = s_tt * s_zz - s_tz * s_tz
    return (s_zz * s_ty - s_tz * s_zy) / det


def draw_scenario(rng: random.Random) -> dict:
    """Draw correlations (rounded to 2dp) for a non-trivially confounded item."""
    for _ in range(1000):
        r_tz = round(rng.uniform(0.3, 0.7) * rng.choice([-1, 1]), 2)
        r_zy = round(rng.uniform(0.3, 0.7) * rng.choice([-1, 1]), 2)
        b = round(rng.uniform(0.2, 0.5) * rng.choice([-1, 1]), 2)
        r_ty = round(b * (1 - r_tz**2) + r_tz * r_zy, 2)
        if is_psd(r_ty, r_tz, r_zy) and abs(r_ty - b) > 0.05 and abs(r_ty) < 0.95:
            return {"r_tz": r_tz, "r_zy": r_zy, "r_ty": r_ty}
    raise RuntimeError("could not draw a valid scenario")
