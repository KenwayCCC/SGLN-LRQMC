"""Typed configuration for an SGLN-LRQMC experiment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(slots=True)
class SGLNLRQMCConfig:
    """All inputs and hyperparameters required for one reconstruction run."""

    image_path: Path
    observed_mask: np.ndarray
    lam: float
    alpha: float
    gamma: float
    eta: float
    mu: float
    rho: float
    mu_max: float
    iterations: int
    tolerance: float
    seed: int
    equivariant_pnp: bool
    evaluate_ssim: bool
    ema_weight: float = 0.0
    truncation: int = 0
    local_denoiser: Literal['ffdnet', 'drunet'] = 'ffdnet'
    cbm3d_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.iterations < 1:
            raise ValueError('iterations must be positive')
        if self.mu <= 0 or self.mu_max <= 0:
            raise ValueError('mu and mu_max must be positive')
        if self.rho <= 0:
            raise ValueError('rho must be positive')
        if not 0 <= self.ema_weight < 1:
            raise ValueError('ema_weight must lie in [0, 1)')
        if self.local_denoiser not in {'ffdnet', 'drunet'}:
            raise ValueError("local_denoiser must be either 'ffdnet' or 'drunet'")
