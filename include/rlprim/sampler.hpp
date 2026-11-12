// rlprim: prioritized replay sampler (Schaul et al. 2016, proportional variant).
// Header-only, C++17, no dependencies.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "sum_tree.hpp"

namespace rlprim {

// xoshiro256++ seeded through splitmix64. Small, fast, good enough for sampling.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) {
    for (auto& word : s_) {
      seed += 0x9e3779b97f4a7c15ULL;
      std::uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      word = z ^ (z >> 31);
    }
  }

  std::uint64_t next() noexcept {
    const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  // Uniform double in [0, 1).
  double uniform() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

 private:
  static std::uint64_t rotl(std::uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
  std::uint64_t s_[4];
};

// Tracks priorities for the slots of a replay buffer and samples slot indices
// with probability proportional to priority^alpha. The transitions themselves
// stay wherever you keep them. This class only deals in indices.
class PrioritizedSampler {
 public:
  PrioritizedSampler(std::size_t capacity, double alpha = 0.6, std::uint64_t seed = 0)
      : sum_(capacity), alpha_(alpha), rng_(seed) {
    if (!(alpha >= 0.0)) throw std::invalid_argument("alpha must be >= 0");
    base_ = 1;
    while (base_ < capacity) base_ <<= 1;
    min_.assign(2 * base_, kInf);
  }

  std::size_t capacity() const noexcept { return sum_.capacity(); }
  std::size_t size() const noexcept { return size_; }
  double alpha() const noexcept { return alpha_; }
  double total() const noexcept { return sum_.total(); }
  // Largest raw priority seen so far. New transitions usually get this one.
  double max_priority() const noexcept { return max_priority_; }
  // Stored priority^alpha of a slot, 0 when the slot is empty.
  double scaled_priority(std::size_t i) const noexcept { return sum_.get(i); }

  // Set the raw priority of slot i. Caller guarantees i < capacity and
  // priority finite and > 0.
  void update(std::size_t i, double priority) noexcept {
    const double scaled = std::pow(priority, alpha_);
    if (sum_.get(i) == 0.0) ++size_;
    if (priority > max_priority_) max_priority_ = priority;
    sum_.set(i, scaled);
    set_min(i, scaled);
  }

  // Empty slot i so it can't be sampled.
  void remove(std::size_t i) noexcept {
    if (sum_.get(i) == 0.0) return;
    --size_;
    sum_.set(i, 0.0);
    set_min(i, kInf);
  }

  // Stratified sample of `batch` indices plus importance weights.
  // Weights are (N * P(i))^-beta divided by the largest possible weight in
  // the buffer, which works out to (p_min / p_i)^beta. Caller guarantees
  // size() > 0.
  void sample(std::size_t batch, double beta, std::int64_t* indices, double* weights) noexcept {
    const double total = sum_.total();
    const double segment = total / static_cast<double>(batch);
    const double p_min = min_[1];
    constexpr std::size_t kGroup = 64;
    double u[kGroup];
    for (std::size_t start = 0; start < batch; start += kGroup) {
      const std::size_t m = (batch - start < kGroup) ? batch - start : kGroup;
      for (std::size_t j = 0; j < m; ++j) {
        u[j] = (static_cast<double>(start + j) + rng_.uniform()) * segment;
      }
      sum_.find_many(u, m, indices + start);
      for (std::size_t j = 0; j < m; ++j) {
        const double p = sum_.get(static_cast<std::size_t>(indices[start + j]));
        weights[start + j] = std::pow(p_min / p, beta);
      }
    }
  }

 private:
  static constexpr double kInf = std::numeric_limits<double>::infinity();

  void set_min(std::size_t i, double value) noexcept {
    std::size_t node = base_ + i;
    min_[node] = value;
    for (node >>= 1; node >= 1; node >>= 1) {
      const double a = min_[2 * node];
      const double b = min_[2 * node + 1];
      min_[node] = a < b ? a : b;
    }
  }

  SumTree sum_;
  std::vector<double> min_;
  std::size_t base_ = 1;
  std::size_t size_ = 0;
  double alpha_;
  double max_priority_ = 1.0;
  Rng rng_;
};

}  // namespace rlprim
