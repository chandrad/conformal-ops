"""conformal-ops: Online Conformal Prediction for Optimization."""

from conformal_ops.core.online_conformal import OnlineConformal
from conformal_ops.dica.dica import DICA
from conformal_ops.baselines.methods import UCA, CPO, EWMA, ACRO, Nominal, FixedMargin
from conformal_ops.diagnostics.free_coverage import (
    FreeCoverageDiagnostic,
    FreeCoverageReport,
)

__version__ = "0.2.0"
__all__ = [
    "OnlineConformal",
    "DICA",
    "UCA",
    "CPO",
    "EWMA",
    "ACRO",
    "Nominal",
    "FixedMargin",
    "FreeCoverageDiagnostic",
    "FreeCoverageReport",
]
