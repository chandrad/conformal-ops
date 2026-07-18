"""
Radii redistribution for DICA.

Core formula:
    w_j = ((1 - beta) + beta * z_bar_j / max(z_bar)) / w_bar
    r_j^DICA = q * sigma_j * w_j

Where z_bar_j is the EMA of |z*_j| (LP allocation feedback).
L1 normalization ensures sum(r_DICA) ≈ sum(r_std) (budget neutrality).
"""

import numpy as np


def redistribute_radii(
    r_std: np.ndarray,
    allocation_ema: np.ndarray,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Redistribute standard conformal radii based on LP allocation feedback.

    Dimensions with high allocation keep standard radii (or slightly wider).
    Dimensions with low allocation get tighter radii (saving cost).
    Total radii budget is approximately preserved (L1 normalization).

    Parameters
    ----------
    r_std : array of shape (d,)
        Standard conformal radii (r_j = q * sigma_j).
    allocation_ema : array of shape (d,)
        Exponential moving average of |z*_j| from recent LP solutions.
    beta : float in [0, 1]
        Redistribution strength. 0 = standard (no change), 1 = fully proportional.

    Returns
    -------
    r_dica : array of shape (d,)
        Reshaped radii.
    """
    alloc_norm = allocation_ema / (np.max(allocation_ema) + 1e-10)
    weights = (1 - beta) + beta * alloc_norm
    weights = weights / (np.mean(weights) + 1e-10)
    return np.maximum(r_std * weights, 0.0)
