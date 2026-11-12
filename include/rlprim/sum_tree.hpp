// rlprim: sum tree for O(log n) proportional sampling.
// Header-only, C++17, no dependencies.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define RLPRIM_PREFETCH(addr) __builtin_prefetch(addr)
#else
#define RLPRIM_PREFETCH(addr) ((void)0)
#endif

namespace rlprim {

// Flat binary tree over `capacity` non-negative weights.
// Leaves live at [base, base + capacity). Node i has children 2i and 2i+1.
// Parents are recomputed from their children on every update, so the root
// never drifts the way a "propagate the delta" tree does.
class SumTree {
 public:
  explicit SumTree(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0) throw std::invalid_argument("capacity must be > 0");
    base_ = 1;
    while (base_ < capacity) base_ <<= 1;
    tree_.assign(2 * base_, 0.0);
  }

  std::size_t capacity() const noexcept { return capacity_; }
  double total() const noexcept { return tree_[1]; }
  double get(std::size_t i) const noexcept { return tree_[base_ + i]; }

  // Set weight i. Caller guarantees i < capacity and value >= 0.
  void set(std::size_t i, double value) noexcept {
    std::size_t node = base_ + i;
    tree_[node] = value;
    for (node >>= 1; node >= 1; node >>= 1) {
      tree_[node] = tree_[2 * node] + tree_[2 * node + 1];
    }
  }

  // Index of the leaf that holds prefix mass `u`, for u in [0, total()).
  // Only ever lands on a leaf with weight > 0, even when rounding pushes
  // `u` past the last one. Caller guarantees total() > 0.
  std::size_t find(double u) const noexcept {
    std::size_t node = 1;
    while (node < base_) {
      const std::size_t left = 2 * node;
      const double lw = tree_[left];
      if (u >= lw && tree_[left + 1] > 0.0) {
        u -= lw;
        node = left + 1;
      } else {
        node = left;
      }
    }
    return node - base_;
  }

  // Same as find() for n lookups at once. Walks a small group of lookups down
  // the tree level by level and prefetches each one's next node, so the cache
  // misses of a big tree overlap. Results match find() exactly.
  void find_many(const double* u, std::size_t n, std::int64_t* out) const noexcept {
    constexpr std::size_t kGroup = 16;
    std::size_t node[kGroup];
    double rem[kGroup];
    for (std::size_t start = 0; start < n; start += kGroup) {
      const std::size_t m = (n - start < kGroup) ? n - start : kGroup;
      for (std::size_t j = 0; j < m; ++j) {
        node[j] = 1;
        rem[j] = u[start + j];
      }
      const double* tree = tree_.data();
      for (std::size_t width = 1; width < base_; width <<= 1) {
        const bool more = 2 * width < base_;  // is there another level below?
        for (std::size_t j = 0; j < m; ++j) {
          const std::size_t left = 2 * node[j];
          const double lw = tree[left];
          const bool right = rem[j] >= lw && tree[left + 1] > 0.0;
          if (right) rem[j] -= lw;
          node[j] = left + (right ? 1 : 0);
          if (more) RLPRIM_PREFETCH(tree + 2 * node[j]);
        }
      }
      for (std::size_t j = 0; j < m; ++j) out[start + j] = static_cast<std::int64_t>(node[j] - base_);
    }
  }

 private:
  std::size_t capacity_;
  std::size_t base_;
  std::vector<double> tree_;
};

}  // namespace rlprim
