"""MUSIC baseline utilities (NumPy-only).

This module was extracted from `train_test.py` so it can be reused independently.

Implemented features (multi-source friendly):
- ULA steering vector (sin(theta) convention)
- MUSIC pseudo-spectrum scanning
- Peak picking via local maxima + minimum separation
- Permutation-invariant RMSE evaluation with pi / 2pi wrap

Notes
-----
1) Angle units:
   - `music_rmse_deg()` expects the DoA label file (`doas*.npy`) to be **radians**,
     consistent with `f_gendata.py`.
   - Internally, estimates/metrics are computed in **degrees**.

2) Periodicity:
   - wrap_period='pi' corresponds to 180° ambiguity: theta ≡ theta + 180°.
   - wrap_period='2pi' corresponds to standard 360° wrapping.

3) Multi-source pairing:
   - Metrics use permutation matching (brute force). This is fine for small M (e.g., 1..4).
"""

from __future__ import annotations

from itertools import permutations
import argparse
from typing import Tuple
from pathlib import Path

import numpy as np


def steering_vector_ula(N: int, theta_deg: np.ndarray, d_over_lambda: float = 0.5) -> np.ndarray:
    """ULA steering vector for a plane wave.

    Convention used here (common in array processing):
        a_n(theta) = exp(-j * 2*pi * d/lambda * n * sin(theta))
    where n = 0..N-1 and theta is in degrees.

    Returns
    -------
    A : np.ndarray
        (N, G) complex steering matrix over grid.
    """

    theta = np.deg2rad(theta_deg).reshape(1, -1)  # (1,G)
    n = np.arange(N, dtype=np.float64).reshape(-1, 1)  # (N,1)
    phase = -1j * 2.0 * np.pi * d_over_lambda * n * np.sin(theta)
    return np.exp(phase)


def wrap_deg_180(x: np.ndarray) -> np.ndarray:
    """Wrap degrees to (-180, 180]."""
    return (x + 180.0) % 360.0 - 180.0


def wrap_deg_90(x: np.ndarray) -> np.ndarray:
    """Wrap degrees to (-90, 90] (pi-periodic / 180° periodic error)."""
    return (x + 90.0) % 180.0 - 90.0


def match_rmse_deg(doa_hat_deg: np.ndarray, doa_gt_deg: np.ndarray, wrap_period: str = "pi") -> float:
    """Permutation-invariant RMSE between two angle sets (deg)."""

    doa_hat_deg = np.asarray(doa_hat_deg, dtype=np.float64).reshape(-1)
    doa_gt_deg = np.asarray(doa_gt_deg, dtype=np.float64).reshape(-1)
    K = int(min(doa_hat_deg.size, doa_gt_deg.size))
    if K <= 0:
        return 0.0

    doa_hat_deg = doa_hat_deg[:K]
    doa_gt_deg = doa_gt_deg[:K]

    wrap_period = str(wrap_period or "pi").lower().strip()
    if wrap_period == "pi":
        wrap = wrap_deg_90
    elif wrap_period == "2pi":
        wrap = wrap_deg_180
    else:
        raise ValueError("wrap_period must be 'pi' or '2pi'")

    best = np.inf
    for perm in permutations(range(K)):
        diff = wrap(doa_hat_deg[list(perm)] - doa_gt_deg)
        mse = float(np.mean(diff**2))
        if mse < best:
            best = mse

    return float(np.sqrt(best))


def music_estimate_doa(
    R: np.ndarray,
    num_sources: int,
    grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
    min_sep_deg: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate DoA(s) from covariance using MUSIC.

    Parameters
    ----------
    R:
        (N,N) complex covariance.
    num_sources:
        number of sources (M).
    grid_deg:
        (G,) scanning grid in degrees.
    d_over_lambda:
        sensor spacing / wavelength.
    min_sep_deg:
        minimum separation between selected peaks on the grid (deg).

    Returns
    -------
    doa_hat_deg:
        (M,) estimated angles in degrees.
    P:
        (G,) MUSIC pseudo-spectrum.
    """

    R = np.asarray(R)
    # Hermitian ensure
    R = 0.5 * (R + R.conj().T)

    # EVD (ascending eigenvalues)
    w, V = np.linalg.eigh(R)
    idx = np.argsort(w)
    V = V[:, idx]

    N = int(R.shape[0])
    M = int(num_sources)
    En = V[:, : max(0, N - M)]  # (N, N-M)

    # scan
    A = steering_vector_ula(N, grid_deg, d_over_lambda=d_over_lambda)  # (N,G)
    EHa = En.conj().T @ A  # (N-M,G)
    denom = np.sum(np.abs(EHa) ** 2, axis=0) + 1e-12
    P = 1.0 / denom

    # peak picking: local maxima + minimum separation
    G = int(P.shape[0])
    if G < 3:
        peak_idx = np.argsort(P)[::-1][:M]
        return np.sort(grid_deg[peak_idx]), P

    loc = np.where((P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:]))[0] + 1
    if loc.size == 0:
        loc = np.argsort(P)[::-1]

    cand = loc[np.argsort(P[loc])[::-1]]

    min_sep_deg = float(max(0.0, min_sep_deg))
    selected: list[int] = []
    for idx0 in cand:
        if len(selected) >= M:
            break
        if not selected:
            selected.append(int(idx0))
            continue
        if np.min(np.abs(grid_deg[int(idx0)] - grid_deg[np.array(selected, dtype=int)])) >= min_sep_deg:
            selected.append(int(idx0))

    if len(selected) < M:
        extra = [int(i) for i in np.argsort(P)[::-1] if int(i) not in selected]
        selected += extra[: max(0, M - len(selected))]

    peak_idx = np.array(selected[:M], dtype=int)
    doa_hat = np.sort(grid_deg[peak_idx])
    return doa_hat, P


def music_rmse_deg(
    cov_file: str,
    doa_file: str,
    num_sources: int,
    max_sets: int | None = 200,
    grid_min: float = -90.0,
    grid_max: float = 90.0,
    grid_num: int = 361,
    d_over_lambda: float = 0.5,
    wrap_period: str = "pi",
    min_sep_deg: float = 5.0,
) -> float:
    """Compute MUSIC DoA RMSE (deg) on generated covariances.

    Expects f_gendata.py outputs:
      covariances*.npy: (num_sets, T, N, N)
      doas*.npy:        (num_sets, M)  [radians]

    We average covariances over T snapshots to get one covariance per set.

    Metric aggregation
    ------------------
    This function returns the **mean of per-set RMSEs** (i.e., compute RMSE per set first,
    then average across sets). This matches the common "per-sample RMSE, then mean" style
    used by `PeriodicPermutationDoALoss(mode='rmse')` in torch (as used in eval_visualize.py).

    Note this is slightly different from "global RMSE" aggregation: sqrt(mean(MSE)).

    Returns
    -------
    rmse_deg : float
        Dataset-level mean RMSE in degrees.
    """

    covs = np.load(cov_file)
    doas_rad = np.load(doa_file)

    if covs.ndim != 4:
        raise ValueError(f"Expect covs (num_sets,T,N,N), got shape={covs.shape}")

    num_sets, T, N, N2 = covs.shape
    if N != N2:
        raise ValueError(f"Cov must be square, got {covs.shape}")

    if doas_rad.shape[0] != num_sets:
        m = min(int(doas_rad.shape[0]), int(num_sets))
        covs = covs[:m]
        doas_rad = doas_rad[:m]
        num_sets = m

    if max_sets is not None:
        num_sets = min(num_sets, int(max_sets))
        covs = covs[:num_sets]
        doas_rad = doas_rad[:num_sets]

    grid = np.linspace(float(grid_min), float(grid_max), int(grid_num), dtype=np.float64)

    rmses = []
    for i in range(num_sets):
        R = covs[i].mean(axis=0)  # (N,N)
        doa_hat, _ = music_estimate_doa(
            R,
            num_sources=int(num_sources),
            grid_deg=grid,
            d_over_lambda=float(d_over_lambda),
            min_sep_deg=float(min_sep_deg),
        )
        doa_gt = np.rad2deg(doas_rad[i, : int(num_sources)])
        rmse_i = match_rmse_deg(doa_hat, doa_gt, wrap_period=wrap_period)
        rmses.append(float(rmse_i))

    # mean of per-set RMSE (matches eval_visualize / PeriodicPermutationDoALoss aggregation)
    return float(np.mean(rmses))


def _infer_num_sources_from_doa_file(doa_file: str | Path) -> int:
    doas = np.load(str(doa_file))
    if doas.ndim != 2:
        raise ValueError(f"Expect doas as (num_sets,M), got {doas.shape}")
    return int(doas.shape[1])
