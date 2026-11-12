"""rlprim: fast building blocks for RL training loops.

C++ core, NumPy in and out.
"""

from ._core import (
    PrioritizedSampler,
    SumTree,
    discounted_returns,
    gae,
    nstep_returns,
    retrace,
    vtrace,
)

__version__ = "0.1.0"

__all__ = [
    "PrioritizedSampler",
    "SumTree",
    "discounted_returns",
    "gae",
    "nstep_returns",
    "retrace",
    "vtrace",
    "__version__",
]
