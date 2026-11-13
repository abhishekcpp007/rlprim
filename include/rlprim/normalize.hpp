// rlprim: running mean and variance for observation or reward normalization.
// Header-only, C++17, no dependencies.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace rlprim {

// Tracks a per-feature mean and variance over everything it has been shown,
// one batch at a time, and rescales new data with them. Uses the same update
// rule and the same starting state (mean 0, variance 1, tiny count) as the
// RunningMeanStd found in Gymnasium and Stable-Baselines3, so the two agree.
class RunningNorm {
 public:
  explicit RunningNorm(std::size_t dim, double epsilon = 1e-4)
      : dim_(dim), count_(epsilon), mean_(dim, 0.0), var_(dim, 1.0), batch_mean_(dim), batch_sq_(dim) {
    if (dim == 0) throw std::invalid_argument("dim must be > 0");
    if (!(epsilon > 0.0)) throw std::invalid_argument("epsilon must be > 0");
  }

  std::size_t dim() const noexcept { return dim_; }
  double count() const noexcept { return count_; }
  const std::vector<double>& mean() const noexcept { return mean_; }
  const std::vector<double>& var() const noexcept { return var_; }

  // Fold in a batch laid out row-major as [rows][dim].
  template <class T>
  void update(const T* x, std::size_t rows) noexcept {
    if (rows == 0) return;
    double* bm = batch_mean_.data();
    double* bs = batch_sq_.data();
    for (std::size_t j = 0; j < dim_; ++j) bm[j] = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
      const T* row = x + i * dim_;
      for (std::size_t j = 0; j < dim_; ++j) bm[j] += static_cast<double>(row[j]);
    }
    const double n = static_cast<double>(rows);
    for (std::size_t j = 0; j < dim_; ++j) {
      bm[j] /= n;
      bs[j] = 0.0;
    }
    for (std::size_t i = 0; i < rows; ++i) {
      const T* row = x + i * dim_;
      for (std::size_t j = 0; j < dim_; ++j) {
        const double diff = static_cast<double>(row[j]) - bm[j];
        bs[j] += diff * diff;
      }
    }
    const double total = count_ + n;
    const double cross = count_ * n / total;
    for (std::size_t j = 0; j < dim_; ++j) {
      const double delta = bm[j] - mean_[j];
      const double m2 = var_[j] * count_ + bs[j] + delta * delta * cross;
      mean_[j] += delta * n / total;
      var_[j] = m2 / total;
    }
    count_ = total;
  }

  // out = clip((x - mean) / sqrt(var + eps), -clip, clip). x and out may be the same buffer.
  template <class T>
  void normalize(const T* x, std::size_t rows, T* out, double clip, double eps = 1e-8) const {
    std::vector<T> mu(dim_), inv(dim_);
    for (std::size_t j = 0; j < dim_; ++j) {
      mu[j] = static_cast<T>(mean_[j]);
      inv[j] = static_cast<T>(1.0 / std::sqrt(var_[j] + eps));
    }
    const T hi = static_cast<T>(clip);
    const T lo = -hi;
    for (std::size_t i = 0; i < rows; ++i) {
      const T* row = x + i * dim_;
      T* o = out + i * dim_;
      for (std::size_t j = 0; j < dim_; ++j) {
        const T z = (row[j] - mu[j]) * inv[j];
        o[j] = z < lo ? lo : (z > hi ? hi : z);
      }
    }
  }

 private:
  std::size_t dim_;
  double count_;
  std::vector<double> mean_, var_, batch_mean_, batch_sq_;
};

}  // namespace rlprim
