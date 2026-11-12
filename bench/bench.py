"""Benchmarks against the code people usually write by hand.

Run:  python bench/bench.py
Prints a markdown table. Each number is the best of several repeats.
"""

import platform
import time

import numpy as np

import rlprim


def best(fn, repeats=7, min_time=0.2):
    """Best per-call time in seconds."""
    fn()
    result = float("inf")
    for _ in range(repeats):
        calls, start = 0, time.perf_counter()
        while True:
            fn()
            calls += 1
            elapsed = time.perf_counter() - start
            if elapsed >= min_time / repeats:
                break
        result = min(result, elapsed / calls)
    return result


def fmt(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


# ---- baselines -------------------------------------------------------------

def numpy_gae(r, v, d, last, gamma=0.99, lam=0.95):
    """The backward loop from CleanRL-style PPO, vectorized over envs."""
    steps = r.shape[0]
    adv = np.zeros_like(r)
    lastgae = 0.0
    for t in reversed(range(steps)):
        nv = last if t == steps - 1 else v[t + 1]
        alive = 1.0 - d[t]
        delta = r[t] + gamma * nv * alive - v[t]
        adv[t] = lastgae = delta + gamma * lam * alive * lastgae
    return adv, adv + v


def numpy_nstep(r, v, d, last, n, gamma=0.99):
    """n passes of whole-array numpy ops, the usual vectorized trick."""
    steps = r.shape[0]
    vals = np.concatenate([v, last[None]], axis=0)
    out = np.zeros_like(r)
    alive = np.ones_like(r)
    coef = 1.0
    boot_idx = np.minimum(np.arange(steps) + n, steps)
    for k in range(n):
        idx = np.minimum(np.arange(steps) + k, steps - 1)
        valid = (np.arange(steps) + k < steps)[:, None]
        out += coef * r[idx] * alive * valid
        ended_here = d[idx] * alive * valid
        alive = alive * (1.0 - ended_here)
        coef *= gamma
    # bootstrap wherever the window never hit a done
    k_eff = boot_idx - np.arange(steps)
    out += (gamma ** k_eff)[:, None] * vals[boot_idx] * alive
    return out


class PySumTree:
    """The pure-Python sum tree found in most PER tutorials."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

    def update(self, idx, p):
        i = idx + self.capacity - 1
        change = p - self.tree[i]
        self.tree[i] = p
        while i != 0:
            i = (i - 1) // 2
            self.tree[i] += change

    def get(self, s):
        i = 0
        while True:
            left = 2 * i + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                i = left
            else:
                s -= self.tree[left]
                i = left + 1
        return i - self.capacity + 1

    def sample(self, batch, rng):
        seg = self.tree[0] / batch
        return [self.get((k + rng.random()) * seg) for k in range(batch)]


# ---- runs ------------------------------------------------------------------

def bench_gae(rows):
    rng = np.random.default_rng(0)
    for steps, envs in [(2048, 1), (2048, 8), (128, 64), (128, 4096)]:
        r = rng.normal(size=(steps, envs)).astype(np.float32)
        v = rng.normal(size=(steps, envs)).astype(np.float32)
        d = (rng.random((steps, envs)) < 0.01)
        df = d.astype(np.float32)
        last = rng.normal(size=envs).astype(np.float32)
        a, _ = rlprim.gae(r, v, d, last)
        b, _ = numpy_gae(r, v, df, last)
        assert np.allclose(a, b, atol=1e-3)
        base = best(lambda: numpy_gae(r, v, df, last))
        ours = best(lambda: rlprim.gae(r, v, d, last))
        rows.append((f"GAE, {steps} steps x {envs} envs", "NumPy loop over steps", base, ours))


def bench_nstep(rows):
    rng = np.random.default_rng(1)
    steps, envs = 2048, 8
    r = rng.normal(size=(steps, envs))
    v = rng.normal(size=(steps, envs))
    d = (rng.random((steps, envs)) < 0.01)
    df = d.astype(np.float64)
    last = rng.normal(size=envs)
    for n in [3, 20]:
        assert np.allclose(rlprim.nstep_returns(r, v, d, last, n=n), numpy_nstep(r, v, df, last, n))
        base = best(lambda: numpy_nstep(r, v, df, last, n))
        ours = best(lambda: rlprim.nstep_returns(r, v, d, last, n=n))
        rows.append((f"{n}-step returns, {steps} steps x {envs} envs", "NumPy, n array passes", base, ours))


def bench_sampler(rows):
    rng = np.random.default_rng(2)
    batch = 256
    for capacity in [100_000, 1_000_000]:
        pri = rng.random(capacity) + 0.01
        ours = rlprim.PrioritizedSampler(capacity, alpha=1.0, seed=0)
        ours.update(np.arange(capacity), pri)

        # full numpy alternative: O(capacity) per batch
        prob = pri / pri.sum()
        t_choice = best(lambda: rng.choice(capacity, size=batch, p=prob), repeats=3)
        t_ours = best(lambda: ours.sample(batch))
        rows.append((f"PER sample, batch {batch}, {capacity:,} slots", "rng.choice(p=...)", t_choice, t_ours))

        if capacity == 100_000:
            py = PySumTree(capacity)
            for i in range(capacity):
                py.update(i, pri[i])
            t_py = best(lambda: py.sample(batch, rng), repeats=3)
            rows.append((f"PER sample, batch {batch}, {capacity:,} slots", "pure-Python sum tree", t_py, t_ours))

            idx = rng.integers(0, capacity, size=batch)
            new = rng.random(batch) + 0.01

            def py_update():
                for i, p in zip(idx.tolist(), new.tolist()):
                    py.update(i, p)

            t_py_up = best(py_update, repeats=3)
            t_ours_up = best(lambda: ours.update(idx, new))
            rows.append((f"PER update, batch {batch}, {capacity:,} slots", "pure-Python sum tree", t_py_up, t_ours_up))


def main():
    rows = []
    bench_gae(rows)
    bench_nstep(rows)
    bench_sampler(rows)
    print(f"{platform.machine()}, {platform.system()}, Python {platform.python_version()}, NumPy {np.__version__}\n")
    print("| Task | Baseline | Baseline time | rlprim | Speedup |")
    print("|---|---|---:|---:|---:|")
    for task, name, base, ours in rows:
        print(f"| {task} | {name} | {fmt(base)} | {fmt(ours)} | {base / ours:.1f}x |")


if __name__ == "__main__":
    main()
