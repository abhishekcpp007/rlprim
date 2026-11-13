# rlprim

Fast building blocks for RL training loops. The core is header-only C++17, and the Python package takes and returns plain NumPy arrays.

Most RL codebases carry their own copy of the same few functions: a GAE loop, an n-step return helper, a sum tree for prioritized replay. They're usually written in Python and they're usually slow. rlprim gives you those pieces as a small library you can drop in without adopting a framework.

## What's in 0.1

| Name | What it does |
|---|---|
| `gae` | Generalized Advantage Estimation over a `(steps, envs)` rollout |
| `discounted_returns` | Reward-to-go with episode cuts and a bootstrap value |
| `nstep_returns` | n-step bootstrapped returns in O(steps), whatever n is |
| `vtrace` | V-trace value targets and policy-gradient advantages (IMPALA) |
| `retrace` | Retrace(lambda) Q targets for off-policy learning |
| `SumTree` | Sum tree with O(log n) update and prefix lookup |
| `PrioritizedSampler` | Proportional prioritized replay sampling with importance weights |
| `RunningNorm` | Running mean and variance for observation or reward normalization |

## Install

```bash
pip install git+https://github.com/abhishekcpp007/rlprim   # needs a C++17 compiler and CMake
pip install rlprim                                          # once wheels are on PyPI
```

Needs Python 3.10 or newer.

## Python

```python
import numpy as np
import rlprim

# rollout arrays are (steps,) or (steps, envs)
adv, ret = rlprim.gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95)
targets = rlprim.nstep_returns(rewards, values, dones, last_value, n=5, gamma=0.99)

# off-policy: log_rhos = log pi(a|s) - log mu(a|s)
vs, pg_adv = rlprim.vtrace(log_rhos, rewards, values, dones, last_value)
q_targets = rlprim.retrace(rewards, q_values, values, log_rhos, dones, last_value)

# prioritized replay: the sampler tracks slot indices, your buffer keeps the data
sampler = rlprim.PrioritizedSampler(capacity=1_000_000, alpha=0.6, seed=0)
sampler.update(new_slots, sampler.max_priority)       # fresh transitions
idx, weights = sampler.sample(batch_size=256, beta=0.4)
sampler.update(idx, np.abs(td_errors) + 1e-6)         # after the learner step

# observation normalization, same math as Gymnasium's RunningMeanStd
norm = rlprim.RunningNorm(obs_shape)
norm.update(obs_batch)
obs_batch = norm.normalize(obs_batch, clip=10.0)
```

`dones[t]` means the episode ended at step t. `last_value` is the value of the state after the final step, either a scalar or one entry per env. float32 rewards give float32 results and everything else gives float64.

## PyTorch

The same functions with tensors in and tensors out. PyTorch isn't a dependency, so this module only loads if you already have it.

```python
from rlprim import torch as rlt

adv, ret = rlt.gae(rewards, values, dones, last_value)        # tensors
vs, pg_adv = rlt.vtrace(log_rhos, rewards, values, dones, last_value)
```

Results land on the device of the first tensor you pass. The math runs on the CPU, so CUDA and MPS tensors get copied over and back. Gradients don't flow through, which is what you want for targets.

## C++

Copy `include/rlprim` into your project, or add the repo with CMake and link `rlprim::rlprim`.

```cpp
#include <rlprim/returns.hpp>
#include <rlprim/sampler.hpp>

rlprim::gae<float>(rewards, values, dones, last_values, steps, envs,
                   0.99f, 0.95f, advantages, returns);

rlprim::PrioritizedSampler sampler(1'000'000, /*alpha=*/0.6, /*seed=*/0);
sampler.update(slot, priority);
sampler.sample(256, /*beta=*/0.4, indices, weights);
```

## Benchmarks

Measured on an Apple Silicon Mac (arm64, Python 3.12, NumPy 2.5), best of 7 runs. Reproduce with `python bench/bench.py`.

| Task | Baseline | Baseline time | rlprim | Speedup |
|---|---|---:|---:|---:|
| GAE, 2048 steps x 1 env | NumPy loop over steps | 7.31 ms | 8.1 us | 905x |
| GAE, 2048 steps x 8 envs | NumPy loop over steps | 7.67 ms | 11.7 us | 654x |
| GAE, 128 steps x 64 envs | NumPy loop over steps | 460.9 us | 2.8 us | 167x |
| GAE, 128 steps x 4096 envs | NumPy loop over steps | 1.03 ms | 137.1 us | 7.5x |
| 3-step returns, 2048 x 8 | NumPy, n array passes | 343.3 us | 23.8 us | 14x |
| 20-step returns, 2048 x 8 | NumPy, n array passes | 1.92 ms | 23.8 us | 80x |
| 3-step returns, 128 x 4096 | NumPy, n array passes | 5.38 ms | 642.5 us | 8.4x |
| V-trace, 2048 x 8 | NumPy loop over steps | 2.43 ms | 18.7 us | 129x |
| Retrace, 2048 x 8 | NumPy loop over steps | 3.99 ms | 19.7 us | 202x |
| V-trace, 128 x 4096 | NumPy loop over steps | 2.35 ms | 531.9 us | 4.4x |
| Retrace, 128 x 4096 | NumPy loop over steps | 1.78 ms | 519.0 us | 3.4x |
| Normalizer update + apply, 8 x 64 | NumPy RunningMeanStd | 20.5 us | 1.0 us | 22x |
| Normalizer update + apply, 4096 x 64 | NumPy RunningMeanStd | 714.9 us | 113.6 us | 6.3x |
| PER sample, batch 256, 100k slots | pure-Python sum tree | 718.7 us | 19.9 us | 36x |
| PER update, batch 256, 100k slots | pure-Python sum tree | 568.6 us | 19.3 us | 29x |
| PER sample, batch 256, 100k slots | `rng.choice(p=...)` | 847.6 us | 19.9 us | 43x |
| PER sample, batch 256, 1M slots | `rng.choice(p=...)` | 7.86 ms | 24.4 us | 323x |

Read these honestly. Every baseline is the per-step NumPy loop from typical training code, so the gap is biggest when you have few envs and Python overhead dominates. With thousands of envs NumPy is already doing real work per step and the gap shrinks to single digits. The V-trace and Retrace rows use float32. With float64 they run about 3x slower than that, because the exp inside them vectorizes less well. No GPU baseline is included yet. Timings move by 10 to 30 percent between runs on a laptop.

### Sampling cost by buffer size

Lookups walk the tree in groups of 16 and prefetch each one's next node, so the cache misses of a big tree overlap. Same machine, `python bench/bench_sample.py`, time per `sample()` call:

| Slots | Batch | Time per call | Per sample |
|---:|---:|---:|---:|
| 100,000 | 256 | 19.2 us | 75 ns |
| 100,000 | 4096 | 229.9 us | 56 ns |
| 1,000,000 | 256 | 25.0 us | 97 ns |
| 1,000,000 | 4096 | 305.3 us | 75 ns |
| 10,000,000 | 256 | 33.2 us | 130 ns |
| 10,000,000 | 4096 | 438.2 us | 107 ns |

Before the grouped walk the 10M-slot rows took 87.6 us and 1151.6 us.

## How it's tested

Every function is compared against a slow brute-force version on random inputs, including rollouts where every step is a done and rollouts with none. The sum tree is checked against `cumsum` plus `searchsorted`, and the normalizer against the Gymnasium formula. Sampling frequencies are checked against `priority ** alpha`. A separate C++ smoke test builds the headers with no Python involved.

```bash
pip install ".[test]" && pytest
cmake -S . -B build -DRLPRIM_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build
```

## Design notes

- The sum tree recomputes each parent from its two children on update. Trees that push a delta up to the root pile up rounding error over many updates, and this one doesn't.
- `find` never returns an empty slot, even when rounding pushes the lookup past the last filled leaf.
- `nstep_returns` keeps the discounted return-to-go for the last n + 1 rows and gets each window as a difference of two of them. The window length is the same across a row, so the inner loop has no per-element lookups and it vectorizes. The running values are kept in double so float32 rollouts don't lose digits in the subtraction.
- V-trace and Retrace use their own `exp`, written so the compiler can vectorize the loop around it. The C++ test checks it against `std::exp`: the worst relative error is 2.2e-16 for double and 7.1e-8 for float.
- The GIL is released inside the return estimators.
- V-trace and Retrace take log importance ratios, the form your policy network already gives you.

## Roadmap

- Segment-tree min/max queries exposed in Python
- Vectorized frame stacking
- A GPU baseline in the benchmarks
- Faster float64 V-trace and Retrace
- Prebuilt wheels on PyPI

## License

MIT
