import numpy as np
import pytest

import rlprim

torch = pytest.importorskip("torch")
from rlprim import torch as rlt  # noqa: E402


def data(dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    r = torch.randn(40, 3, generator=g, dtype=dtype)
    v = torch.randn(40, 3, generator=g, dtype=dtype)
    d = torch.rand(40, 3, generator=g) < 0.1
    last = torch.randn(3, generator=g, dtype=dtype)
    lr = 0.5 * torch.randn(40, 3, generator=g, dtype=dtype)
    return r, v, d, last, lr


def test_every_function_matches_the_numpy_api():
    r, v, d, last, lr = data()
    n = [x.numpy() for x in (r, v, d, last, lr)]
    pairs = [
        (rlt.gae(r, v, d, last), rlprim.gae(n[0], n[1], n[2], n[3])),
        (rlt.vtrace(lr, r, v, d, last), rlprim.vtrace(n[4], n[0], n[1], n[2], n[3])),
        ((rlt.discounted_returns(r, d, last),), (rlprim.discounted_returns(n[0], n[2], n[3]),)),
        ((rlt.nstep_returns(r, v, d, last, n=4),), (rlprim.nstep_returns(n[0], n[1], n[2], n[3], n=4),)),
        ((rlt.retrace(r, v, v, lr, d, last),), (rlprim.retrace(n[0], n[1], n[1], n[4], n[2], n[3]),)),
    ]
    for got, want in pairs:
        for g, w in zip(got, want):
            assert isinstance(g, torch.Tensor) and g.dtype == torch.float32
            np.testing.assert_array_equal(g.numpy(), w)


def test_float64_and_keywords():
    r, v, d, last, _ = data(torch.float64)
    adv, ret = rlt.gae(rewards=r, values=v, dones=d, last_value=last, gamma=0.9, lam=0.8)
    assert adv.dtype == torch.float64
    want, _ = rlprim.gae(r.numpy(), v.numpy(), d.numpy(), last.numpy(), gamma=0.9, lam=0.8)
    np.testing.assert_array_equal(adv.numpy(), want)


def test_tensors_that_require_grad_and_half_precision():
    r, v, d, last, _ = data()
    v_grad = v.clone().requires_grad_(True)
    adv, _ = rlt.gae(r, v_grad, d, last)
    assert not adv.requires_grad
    np.testing.assert_array_equal(adv.numpy(), rlt.gae(r, v, d, last)[0].numpy())
    out = rlt.discounted_returns(r.half(), d, last)
    assert torch.isfinite(out).all()


def test_non_contiguous_tensor():
    r, v, d, last, _ = data()
    got, _ = rlt.gae(r[::2], v[::2], d[::2], last)
    want, _ = rlprim.gae(r.numpy()[::2], v.numpy()[::2], d.numpy()[::2], last.numpy())
    np.testing.assert_array_equal(got.numpy(), want)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_result_returns_to_the_input_device():
    r, v, d, last, _ = data()
    adv, ret = rlt.gae(r.to("mps"), v.to("mps"), d.to("mps"), last.to("mps"))
    assert adv.device.type == "mps" and ret.device.type == "mps"
    np.testing.assert_allclose(adv.cpu().numpy(), rlt.gae(r, v, d, last)[0].numpy(), rtol=1e-6)
