import numpy as np
import pytest

import rlprim


class NumpyRunningMeanStd:
    """The Gymnasium / Stable-Baselines3 implementation, kept here as the reference."""

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        bm, bv, bc = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + np.square(delta) * self.count * bc / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot


@pytest.mark.parametrize("shape", [(1,), (7,), (3, 4), (2, 3, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_reference_over_many_batches(shape, dtype):
    rng = np.random.default_rng(len(shape))
    ours, ref = rlprim.RunningNorm(shape), NumpyRunningMeanStd(shape)
    seen = []
    for batch in [1, 5, 64, 2, 300]:
        x = (rng.normal(size=(batch, *shape)) * 3 + 2).astype(dtype)
        ours.update(x)
        ref.update(x.astype(np.float64))
        seen.append(x.astype(np.float64))
    np.testing.assert_allclose(ours.mean, ref.mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ours.var, ref.var, rtol=1e-12, atol=1e-12)
    assert ours.count == pytest.approx(ref.count)
    assert ours.mean.shape == shape
    everything = np.concatenate(seen)
    np.testing.assert_allclose(ours.mean, everything.mean(axis=0), atol=1e-4)
    np.testing.assert_allclose(ours.var, everything.var(axis=0), rtol=1e-4)


def test_normalize_matches_numpy_and_keeps_dtype():
    rng = np.random.default_rng(0)
    norm = rlprim.RunningNorm(6)
    norm.update(rng.normal(size=(500, 6)) * 4 + 1)
    for dtype in (np.float32, np.float64):
        x = (rng.normal(size=(40, 6)) * 20).astype(dtype)
        got = norm.normalize(x, clip=5.0)
        want = np.clip((x - norm.mean) / np.sqrt(norm.var + 1e-8), -5.0, 5.0)
        assert got.dtype == dtype and got.shape == x.shape
        np.testing.assert_allclose(got, want, rtol=1e-5 if dtype == np.float32 else 1e-12, atol=1e-6)
        assert np.abs(got).max() <= 5.0
    single = norm.normalize(x[0])
    assert single.shape == (6,)
    np.testing.assert_array_equal(single, norm.normalize(x)[0])


def test_single_item_update_and_int_shape():
    norm = rlprim.RunningNorm(3)
    norm.update(np.array([1.0, 2.0, 3.0]))
    norm.update(np.array([[3.0, 2.0, 1.0]]))
    assert norm.count == pytest.approx(2.0001)
    np.testing.assert_allclose(norm.mean, [2.0, 2.0, 2.0], atol=1e-3)


def test_fresh_normalizer_is_identity():
    x = np.array([[0.5, -2.0]])
    np.testing.assert_allclose(rlprim.RunningNorm(2).normalize(x), x, rtol=1e-7)


def test_empty_batch_is_a_no_op():
    norm = rlprim.RunningNorm(4)
    norm.update(np.zeros((0, 4)))
    assert norm.count == pytest.approx(1e-4)


def test_errors():
    norm = rlprim.RunningNorm((2, 3))
    with pytest.raises(ValueError):
        norm.update(np.zeros((5, 3, 2)))
    with pytest.raises(ValueError):
        norm.normalize(np.zeros(6))
    with pytest.raises(ValueError):
        norm.normalize(np.zeros((2, 3)), clip=0.0)
    with pytest.raises(ValueError):
        rlprim.RunningNorm(0)
    with pytest.raises(ValueError):
        rlprim.RunningNorm(3, epsilon=0.0)
