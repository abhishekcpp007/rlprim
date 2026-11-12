import numpy as np
import pytest

import rlprim


@pytest.mark.parametrize("capacity", [1, 2, 3, 8, 13, 100, 1000])
def test_sum_tree_matches_cumsum(capacity):
    rng = np.random.default_rng(capacity)
    tree = rlprim.SumTree(capacity)
    weights = np.zeros(capacity)
    for _ in range(5):
        k = int(rng.integers(1, capacity + 1))
        idx = rng.integers(0, capacity, size=k)
        # whole numbers, with plenty of zeros, so the brute force is exact
        val = rng.integers(0, 6, size=k).astype(float)
        tree.set(idx, val)
        for i, x in zip(idx, val):
            weights[i] = x
        assert tree.total == weights.sum()
        np.testing.assert_array_equal(tree.get(np.arange(capacity)), weights)
        if weights.sum() == 0:
            continue
        u = rng.random(200) * weights.sum()
        want = np.searchsorted(np.cumsum(weights), u, side="right")
        got = tree.find(u)
        np.testing.assert_array_equal(got, want)
        assert (weights[got] > 0).all()


def test_sum_tree_find_never_lands_on_zero_weight():
    tree = rlprim.SumTree(10)
    tree.set([2, 5], [0.1, 0.2])
    total = tree.total
    u = np.array([0.0, np.nextafter(total, 0), total, total * 2])
    assert set(tree.find(u).tolist()) <= {2, 5}


def test_sum_tree_no_drift_after_many_updates():
    rng = np.random.default_rng(0)
    tree = rlprim.SumTree(257)
    weights = np.zeros(257)
    for _ in range(200):
        idx = rng.integers(0, 257, size=64)
        val = rng.random(64) * 1e6
        tree.set(idx, val)
        for i, x in zip(idx, val):
            weights[i] = x
    assert tree.total == pytest.approx(weights.sum(), rel=1e-12)


def test_sum_tree_errors():
    tree = rlprim.SumTree(4)
    with pytest.raises(IndexError):
        tree.set([4], [1.0])
    with pytest.raises(IndexError):
        tree.get([-1])
    with pytest.raises(ValueError):
        tree.set([0], [-1.0])
    with pytest.raises(ValueError):
        tree.set([0], [float("nan")])
    with pytest.raises(ValueError):
        tree.find([0.0])
    with pytest.raises(ValueError):
        rlprim.SumTree(0)


def test_sampler_frequencies_follow_priority_to_the_alpha():
    capacity, alpha = 64, 0.7
    rng = np.random.default_rng(1)
    pri = rng.random(capacity) * 5 + 0.1
    s = rlprim.PrioritizedSampler(capacity, alpha=alpha, seed=123)
    s.update(np.arange(capacity), pri)
    counts = np.zeros(capacity)
    draws = 0
    for _ in range(4000):
        idx, _ = s.sample(64)
        np.add.at(counts, idx, 1)
        draws += 64
    want = pri**alpha / (pri**alpha).sum()
    got = counts / draws
    # 256k draws: standard error per slot is about 2.5e-4
    np.testing.assert_allclose(got, want, atol=2e-3)


def test_sampler_weights():
    capacity, alpha, beta = 50, 0.6, 0.5
    rng = np.random.default_rng(2)
    pri = rng.random(capacity) + 0.05
    s = rlprim.PrioritizedSampler(capacity, alpha=alpha, seed=7)
    s.update(np.arange(capacity), pri)
    idx, w = s.sample(256, beta=beta)
    scaled = pri**alpha
    prob = scaled / scaled.sum()
    want = (capacity * prob[idx]) ** -beta
    want /= ((capacity * prob.min()) ** -beta)
    np.testing.assert_allclose(w, want, rtol=1e-12)
    assert w.max() <= 1.0 + 1e-12 and w.min() > 0


def test_sampler_only_returns_filled_slots():
    s = rlprim.PrioritizedSampler(1000, seed=3)
    filled = np.array([3, 500, 999])
    s.update(filled, [1.0, 2.0, 0.5])
    assert len(s) == 3
    for _ in range(200):
        idx, _ = s.sample(32)
        assert set(idx.tolist()) <= set(filled.tolist())
    s.remove([500])
    assert len(s) == 2
    for _ in range(200):
        idx, _ = s.sample(32)
        assert set(idx.tolist()) <= {3, 999}
    s.remove([500])  # removing twice is a no-op
    assert len(s) == 2


def test_sampler_stratified_covers_every_equal_slot():
    s = rlprim.PrioritizedSampler(16, seed=4)
    s.update(np.arange(16), 1.0)
    idx, w = s.sample(16)
    np.testing.assert_array_equal(idx, np.arange(16))
    np.testing.assert_allclose(w, 1.0)


def test_sampler_seed_is_reproducible():
    def run(seed):
        s = rlprim.PrioritizedSampler(300, seed=seed)
        s.update(np.arange(300), np.linspace(0.1, 3, 300))
        return np.concatenate([s.sample(32)[0] for _ in range(10)])

    np.testing.assert_array_equal(run(11), run(11))
    assert not np.array_equal(run(11), run(12))


def test_sampler_bookkeeping():
    s = rlprim.PrioritizedSampler(8, alpha=0.5)
    assert len(s) == 0 and s.max_priority == 1.0 and s.capacity == 8
    s.update([0, 1], [4.0, 9.0])
    assert s.max_priority == 9.0
    np.testing.assert_allclose(s.priorities([0, 1, 2]), [2.0, 3.0, 0.0])
    assert s.total == pytest.approx(5.0)
    s.update([0], [16.0])  # overwrite keeps the count
    assert len(s) == 2 and s.total == pytest.approx(7.0)


def test_sampler_errors():
    s = rlprim.PrioritizedSampler(4)
    with pytest.raises(ValueError):
        s.sample(2)
    with pytest.raises(IndexError):
        s.update([4], [1.0])
    with pytest.raises(ValueError):
        s.update([0], [0.0])
    with pytest.raises(ValueError):
        s.update([0], [float("inf")])
    with pytest.raises(ValueError):
        s.update([0, 1], [1.0, 2.0, 3.0])
    s.update([0], [1.0])
    with pytest.raises(ValueError):
        s.sample(0)
    with pytest.raises(ValueError):
        rlprim.PrioritizedSampler(4, alpha=-1.0)
    # a failed update must not half-apply
    with pytest.raises(ValueError):
        s.update([1, 2], [1.0, -1.0])
    assert len(s) == 1


@pytest.mark.parametrize("count", [1, 15, 16, 17, 64, 1000])
def test_sum_tree_find_handles_any_batch_size(count):
    rng = np.random.default_rng(count)
    weights = rng.integers(0, 4, size=777).astype(float)
    weights[5] = 1.0
    tree = rlprim.SumTree(777)
    tree.set(np.arange(777), weights)
    u = rng.random(count) * weights.sum()
    np.testing.assert_array_equal(tree.find(u), np.searchsorted(np.cumsum(weights), u, side="right"))
