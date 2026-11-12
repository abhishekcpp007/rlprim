"""PyTorch front end: the same functions, tensors in and tensors out.

    from rlprim import torch as rlt
    adv, ret = rlt.gae(rewards, values, dones, last_value)

Results come back on the device of the first tensor argument. The math runs on
the CPU, so CUDA and MPS tensors are copied over and back. Gradients don't
flow through these functions, which is what you want for targets.
"""

import functools

import torch as _torch

from . import _core

__all__ = ["discounted_returns", "gae", "nstep_returns", "retrace", "vtrace"]


def _to_numpy(x):
    if isinstance(x, _torch.Tensor):
        x = x.detach()
        if x.dtype in (_torch.float16, _torch.bfloat16):
            x = x.float()
        return x.cpu().numpy()
    return x


def _wrap(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        device = next(
            (a.device for a in (*args, *kwargs.values()) if isinstance(a, _torch.Tensor)), None
        )
        out = fn(*map(_to_numpy, args), **{k: _to_numpy(v) for k, v in kwargs.items()})

        def back(a):
            t = _torch.from_numpy(a)
            return t if device is None else t.to(device)

        return tuple(back(o) for o in out) if isinstance(out, tuple) else back(out)

    return inner


gae = _wrap(_core.gae)
discounted_returns = _wrap(_core.discounted_returns)
nstep_returns = _wrap(_core.nstep_returns)
vtrace = _wrap(_core.vtrace)
retrace = _wrap(_core.retrace)
