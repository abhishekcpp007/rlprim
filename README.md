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

## Install

```bash
pip install git+https://github.com/abhishekcpp007/rlprim   # needs a C++17 compiler and CMake
pip install rlprim                                          # once wheels are on PyPI
```

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
```

`dones[t]` means the episode ended at step t. `last_value` is the value of the state after the final step, either a scalar or one entry per env. float32 rewards give float32 results and everything else gives float64.

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
| GAE, 2048 steps x 1 env | NumPy loop over steps | 7.69 ms | 8.4 us | 920x |
| GAE, 2048 steps x 8 envs | NumPy loop over steps | 9.03 ms | 13.0 us | 692x |
| GAE, 128 steps x 64 envs | NumPy loop over steps | 492.4 us | 3.1 us | 161x |
| GAE, 128 steps x 4096 envs | NumPy loop over steps | 1.32 ms | 188.4 us | 7.0x |
| 3-step returns, 2048 x 8 | NumPy, n array passes | 386.7 us | 44.1 us | 8.8x |
| 20-step returns, 2048 x 8 | NumPy, n array passes | 2.28 ms | 40.0 us | 57x |
| PER sample, batch 256, 100k slots | pure-Python sum tree | 1.02 ms | 21.9 us | 47x |
| PER update, batch 256, 100k slots | pure-Python sum tree | 985.7 us | 21.9 us | 45x |
| PER sample, batch 256, 100k slots | `rng.choice(p=...)` | 1.07 ms | 21.9 us | 49x |
| PER sample, batch 256, 1M slots | `rng.choice(p=...)` | 9.29 ms | 31.8 us | 292x |

Read these honestly. The GAE baseline is the per-step NumPy loop from typical PPO code, so the gap is biggest when you have few envs and Python overhead dominates. With thousands of envs NumPy is already doing real work per step and the gap shrinks to single digits. No GPU baseline is included yet, and V-trace and Retrace have no benchmark yet. Timings move by 10 to 30 percent between runs on a laptop.

### Sampling cost by buffer size

Lookups walk the tree in groups of 16 and prefetch each one's next node, so the cache misses of a big tree overlap. Same machine, `python bench/bench_sample.py`, time per `sample()` call:

| Slots | Batch | One lookup at a time | Grouped with prefetch |
|---:|---:|---:|---:|
| 100,000 | 256 | 29.6 us | 20.9 us |
| 100,000 | 4096 | 336.4 us | 245.6 us |
| 1,000,000 | 256 | 58.0 us | 26.8 us |
| 1,000,000 | 4096 | 695.5 us | 363.2 us |
| 10,000,000 | 256 | 87.6 us | 38.3 us |
| 10,000,000 | 4096 | 1151.6 us | 549.9 us |

## How it's tested

Every function is compared against a slow brute-force version on random inputs, including rollouts where every step is a done and rollouts with none. The sum tree is checked against `cumsum` plus `searchsorted`. Sampling frequencies are checked against `priority ** alpha`. A separate C++ smoke test builds the headers with no Python involved.

```bash
pip install ".[test]" && pytest
cmake -S . -B build -DRLPRIM_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build
```

## Design notes

- The sum tree recomputes each parent from its two children on update. Trees that push a delta up to the root pile up rounding error over many updates, and this one doesn't.
- `find` never returns an empty slot, even when rounding pushes the lookup past the last filled leaf.
- `nstep_returns` slides one window backward per env. It keeps the running sum in double so float32 rollouts don't pick up error from the subtraction.
- The GIL is released inside the return estimators.
- V-trace and Retrace take log importance ratios, the form your policy network already gives you.

## Roadmap

- Segment-tree min/max queries exposed in Python
- Vectorized frame stacking and observation normalization
- Benchmarks for V-trace and Retrace, plus a GPU baseline
- Prebuilt wheels on PyPI

## License

MIT
