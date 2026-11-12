"""Sampling cost by buffer size. Run: python bench/bench_sample.py"""
import time

import numpy as np

import rlprim


def best(fn, repeats=9, calls=300):
    fn()
    result = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        result = min(result, (time.perf_counter() - start) / calls)
    return result


rng = np.random.default_rng(0)
for capacity in [100_000, 1_000_000, 10_000_000]:
    s = rlprim.PrioritizedSampler(capacity, alpha=1.0, seed=0)
    s.update(np.arange(capacity), rng.random(capacity) + 0.01)
    for batch in [256, 4096]:
        t = best(lambda: s.sample(batch))
        print(f"{capacity:>10,} slots, batch {batch:>4}: {t * 1e6:8.1f} us  ({t / batch * 1e9:.0f} ns/sample)")
