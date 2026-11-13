// rlprim: advantage and return estimators for rollouts.
// Header-only, C++17, no dependencies.
//
// Every function takes a rollout laid out row-major as [steps][envs].
// dones[t][e] != 0 means the episode of env e ended at step t, so nothing
// after step t leaks into the estimate for step t.
// last_values[e] is the value of the state that follows the final step.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#define RLPRIM_RESTRICT __restrict
#else
#define RLPRIM_RESTRICT __restrict__
#endif

namespace rlprim {

namespace detail {

// exp() written so a compiler can vectorize the loop around it, which it can't
// do with a call into libm. Range reduction x = k ln2 + r, a Taylor polynomial
// for e^r, then 2^k built straight from the exponent bits. Matches std::exp to
// a few ulp (the smoke test checks it). The input is clamped so the result
// stays a normal number: exp(-700) is as good as zero for an importance ratio.
inline double exp_fast(double x) noexcept {
  constexpr double kLo = -700.0, kHi = 700.0;
  constexpr double kInvLn2 = 1.44269504088896338700e+00;
  constexpr double kLn2Hi = 6.93147180369123816490e-01;
  constexpr double kLn2Lo = 1.90821492927058770002e-10;
  constexpr double kRound = 6755399441055744.0;  // 1.5 * 2^52: adding it rounds to an integer
  x = x < kLo ? kLo : (x > kHi ? kHi : x);
  const double shifted = x * kInvLn2 + kRound;
  const double k = shifted - kRound;
  const double r = (x - k * kLn2Hi) - k * kLn2Lo;
  double p = 1.0 / 6227020800.0;
  p = p * r + 1.0 / 479001600.0;
  p = p * r + 1.0 / 39916800.0;
  p = p * r + 1.0 / 3628800.0;
  p = p * r + 1.0 / 362880.0;
  p = p * r + 1.0 / 40320.0;
  p = p * r + 1.0 / 5040.0;
  p = p * r + 1.0 / 720.0;
  p = p * r + 1.0 / 120.0;
  p = p * r + 1.0 / 24.0;
  p = p * r + 1.0 / 6.0;
  p = p * r + 0.5;
  p = p * r + 1.0;
  p = p * r + 1.0;
  // the low bits of `shifted` hold k as an integer
  std::uint64_t bits;
  std::memcpy(&bits, &shifted, sizeof bits);
  bits = (bits + 1023u) << 52;
  double scale;
  std::memcpy(&scale, &bits, sizeof scale);
  return p * scale;
}

inline float exp_fast(float x) noexcept {
  constexpr float kLo = -87.0f, kHi = 88.0f;
  constexpr float kInvLn2 = 1.44269504088896341f;
  constexpr float kLn2Hi = 0.693359375f;
  constexpr float kLn2Lo = -2.12194440e-4f;
  constexpr float kRound = 12582912.0f;  // 1.5 * 2^23
  x = x < kLo ? kLo : (x > kHi ? kHi : x);
  const float shifted = x * kInvLn2 + kRound;
  const float k = shifted - kRound;
  const float r = (x - k * kLn2Hi) - k * kLn2Lo;
  float p = 1.0f / 5040.0f;
  p = p * r + 1.0f / 720.0f;
  p = p * r + 1.0f / 120.0f;
  p = p * r + 1.0f / 24.0f;
  p = p * r + 1.0f / 6.0f;
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  p = p * r + 1.0f;
  std::uint32_t bits;
  std::memcpy(&bits, &shifted, sizeof bits);
  bits = (bits + 127u) << 23;
  float scale;
  std::memcpy(&scale, &bits, sizeof scale);
  return p * scale;
}

}  // namespace detail

// Generalized Advantage Estimation (Schulman et al. 2016).
// Writes advantages and returns (advantage + value). One backward pass.
template <class T>
void gae(const T* RLPRIM_RESTRICT rewards, const T* RLPRIM_RESTRICT values,
         const std::uint8_t* RLPRIM_RESTRICT dones, const T* RLPRIM_RESTRICT last_values,
         std::size_t steps, std::size_t envs, T gamma, T lam,
         T* RLPRIM_RESTRICT advantages, T* RLPRIM_RESTRICT returns) {
  const T gl = gamma * lam;
  for (std::size_t t = steps; t-- > 0;) {
    const bool last = (t + 1 == steps);
    const T* next_v = last ? last_values : values + (t + 1) * envs;
    const T* next_a = last ? nullptr : advantages + (t + 1) * envs;
    const T* r = rewards + t * envs;
    const T* v = values + t * envs;
    const std::uint8_t* d = dones + t * envs;
    T* a = advantages + t * envs;
    T* g = returns + t * envs;
    if (last) {
      for (std::size_t e = 0; e < envs; ++e) {
        const T alive = d[e] ? T(0) : T(1);
        const T adv = r[e] + gamma * next_v[e] * alive - v[e];
        a[e] = adv;
        g[e] = adv + v[e];
      }
    } else {
      for (std::size_t e = 0; e < envs; ++e) {
        const T alive = d[e] ? T(0) : T(1);
        const T delta = r[e] + gamma * next_v[e] * alive - v[e];
        const T adv = delta + gl * alive * next_a[e];
        a[e] = adv;
        g[e] = adv + v[e];
      }
    }
  }
}

// Discounted reward-to-go: G_t = r_t + gamma * G_{t+1}, cut at episode ends
// and bootstrapped from last_values at the end of the rollout.
template <class T>
void discounted_returns(const T* RLPRIM_RESTRICT rewards, const std::uint8_t* RLPRIM_RESTRICT dones,
                        const T* RLPRIM_RESTRICT last_values, std::size_t steps, std::size_t envs,
                        T gamma, T* RLPRIM_RESTRICT out) {
  for (std::size_t t = steps; t-- > 0;) {
    const T* next = (t + 1 == steps) ? last_values : out + (t + 1) * envs;
    const T* r = rewards + t * envs;
    const std::uint8_t* d = dones + t * envs;
    T* g = out + t * envs;
    for (std::size_t e = 0; e < envs; ++e) {
      g[e] = r[e] + (d[e] ? T(0) : gamma * next[e]);
    }
  }
}

// n-step returns:
//   G_t = r_t + gamma r_{t+1} + ... + gamma^{k-1} r_{t+k-1} + gamma^k V(s_{t+k})
// where k is n, shortened by an episode end (no bootstrap then) or by the end
// of the rollout (bootstrap from last_values).
//
// Cost is O(steps * envs) whatever n is. One backward pass keeps the plain
// discounted return-to-go R_t for the last n + 1 rows, and uses
//   sum of the window = R_t - gamma^k R_{t+k}
// For a given row k is the same for every env, so the inner loop touches whole
// rows with no per-element lookups and the compiler can vectorize it. R is
// kept in double so float32 rollouts don't lose digits in the subtraction.
template <class T>
void nstep_returns(const T* RLPRIM_RESTRICT rewards, const T* RLPRIM_RESTRICT values,
                   const std::uint8_t* RLPRIM_RESTRICT dones,
                   const T* RLPRIM_RESTRICT last_values, std::size_t steps, std::size_t envs,
                   std::size_t n, T gamma, T* RLPRIM_RESTRICT out) {
  if (n == 0 || steps == 0 || envs == 0) return;
  if (n > steps) n = steps;
  const double g = static_cast<double>(gamma);
  std::vector<double> gpow(n + 1);
  gpow[0] = 1.0;
  for (std::size_t k = 1; k <= n; ++k) gpow[k] = gpow[k - 1] * g;

  const std::size_t ring_rows = n + 1;
  std::vector<double> ring(ring_rows * envs);  // R for rows t .. t + n
  const std::vector<double> zeros(envs, 0.0);  // R past the end of the rollout
  // steps until the next episode end, counting that step. Saturates when there is none.
  constexpr std::uint32_t kNever = 0x7fffffffu;
  std::vector<std::uint32_t> dist_buf(envs, kNever);
  std::uint32_t* RLPRIM_RESTRICT dist = dist_buf.data();

  for (std::size_t t = steps; t-- > 0;) {
    const std::size_t k = (steps - t < n) ? steps - t : n;  // window length without a done
    const bool at_end = (t + k == steps);
    const double gk = gpow[k];
    const auto k32 = static_cast<std::uint32_t>(k);
    const T* RLPRIM_RESTRICT r = rewards + t * envs;
    const std::uint8_t* RLPRIM_RESTRICT d = dones + t * envs;
    const T* RLPRIM_RESTRICT boot = at_end ? last_values : values + (t + k) * envs;
    double* RLPRIM_RESTRICT cur = ring.data() + (t % ring_rows) * envs;
    const double* RLPRIM_RESTRICT next =
        (t + 1 == steps) ? zeros.data() : ring.data() + ((t + 1) % ring_rows) * envs;
    const double* RLPRIM_RESTRICT far =
        at_end ? zeros.data() : ring.data() + ((t + k) % ring_rows) * envs;
    T* RLPRIM_RESTRICT o = out + t * envs;

    if (k == 1) {
      // next and far are the same row here, so read it once
      for (std::size_t e = 0; e < envs; ++e) {
        const bool done = d[e] != 0;
        const double nx = next[e];
        const double ret = static_cast<double>(r[e]) + (done ? 0.0 : g * nx);
        dist[e] = done ? 1u : dist[e] + (dist[e] < kNever ? 1u : 0u);
        cur[e] = ret;
        o[e] = static_cast<T>(ret + (done ? 0.0 : gk * (static_cast<double>(boot[e]) - nx)));
      }
      continue;
    }
    for (std::size_t e = 0; e < envs; ++e) {
      const bool done = d[e] != 0;
      const double ret = static_cast<double>(r[e]) + (done ? 0.0 : g * next[e]);
      const std::uint32_t ds = done ? 1u : dist[e] + (dist[e] < kNever ? 1u : 0u);
      dist[e] = ds;
      cur[e] = ret;
      const double tail = gk * (static_cast<double>(boot[e]) - far[e]);
      o[e] = static_cast<T>(ret + (ds <= k32 ? 0.0 : tail));
    }
  }
}

// V-trace targets (Espeholt et al. 2018, IMPALA) for off-policy actor-critic.
// log_rhos[t][e] = log pi(a_t|s_t) - log mu(a_t|s_t), target over behaviour.
// Writes the value targets `vs` and the policy-gradient advantages
//   pg_adv_t = min(clip_pg_rho, rho_t) * (r_t + gamma * vs_{t+1} - V_t).
template <class T>
void vtrace(const T* RLPRIM_RESTRICT log_rhos, const T* RLPRIM_RESTRICT rewards,
            const T* RLPRIM_RESTRICT values, const std::uint8_t* RLPRIM_RESTRICT dones,
            const T* RLPRIM_RESTRICT last_values, std::size_t steps, std::size_t envs, T gamma,
            T lam, T clip_rho, T clip_c, T clip_pg_rho, T* RLPRIM_RESTRICT vs,
            T* RLPRIM_RESTRICT pg_advantages) {
  for (std::size_t t = steps; t-- > 0;) {
    const bool last = (t + 1 == steps);
    // every pointer below is a different row, so they can't overlap
    const T* RLPRIM_RESTRICT next_v = last ? last_values : values + (t + 1) * envs;
    const T* RLPRIM_RESTRICT next_vs = last ? last_values : vs + (t + 1) * envs;
    const T* RLPRIM_RESTRICT lr = log_rhos + t * envs;
    const T* RLPRIM_RESTRICT r = rewards + t * envs;
    const T* RLPRIM_RESTRICT v = values + t * envs;
    const std::uint8_t* RLPRIM_RESTRICT d = dones + t * envs;
    T* RLPRIM_RESTRICT out_vs = vs + t * envs;
    T* RLPRIM_RESTRICT out_pg = pg_advantages + t * envs;
    for (std::size_t e = 0; e < envs; ++e) {
      const T rho_raw = detail::exp_fast(lr[e]);
      const T rho = rho_raw < clip_rho ? rho_raw : clip_rho;
      const T c = lam * (rho_raw < clip_c ? rho_raw : clip_c);
      const T rho_pg = rho_raw < clip_pg_rho ? rho_raw : clip_pg_rho;
      const T disc = d[e] ? T(0) : gamma;
      const T nv = next_v[e];
      const T nvs = next_vs[e];
      const T base = r[e] - v[e];
      out_vs[e] = v[e] + rho * (base + disc * nv) + disc * c * (nvs - nv);
      out_pg[e] = rho_pg * (base + disc * nvs);
    }
  }
}

// Retrace(lambda) targets (Munos et al. 2016) for off-policy Q-learning.
//   G_t = r_t + gamma * (V_{t+1} + c_{t+1} * (G_{t+1} - Q_{t+1}))
//   c_t = lam * min(1, exp(log_rhos[t]))
// q_values[t] = Q(s_t, a_t) for the action taken, values[t] = E_pi Q(s_t, .),
// last_values = E_pi Q of the state after the final step.
template <class T>
void retrace(const T* RLPRIM_RESTRICT rewards, const T* RLPRIM_RESTRICT q_values,
             const T* RLPRIM_RESTRICT values, const T* RLPRIM_RESTRICT log_rhos,
             const std::uint8_t* RLPRIM_RESTRICT dones, const T* RLPRIM_RESTRICT last_values,
             std::size_t steps, std::size_t envs, T gamma, T lam, T* RLPRIM_RESTRICT out) {
  for (std::size_t t = steps; t-- > 0;) {
    const std::size_t row = t * envs;
    if (t + 1 == steps) {
      for (std::size_t e = 0; e < envs; ++e) {
        out[row + e] = rewards[row + e] + (dones[row + e] ? T(0) : gamma * last_values[e]);
      }
      continue;
    }
    const T* RLPRIM_RESTRICT r = rewards + row;
    const std::uint8_t* RLPRIM_RESTRICT d = dones + row;
    const T* RLPRIM_RESTRICT nv = values + row + envs;
    const T* RLPRIM_RESTRICT nq = q_values + row + envs;
    const T* RLPRIM_RESTRICT nlr = log_rhos + row + envs;
    const T* RLPRIM_RESTRICT nout = out + row + envs;
    T* RLPRIM_RESTRICT o = out + row;
    for (std::size_t e = 0; e < envs; ++e) {
      const T rho = detail::exp_fast(nlr[e]);
      const T c = lam * (rho < T(1) ? rho : T(1));
      const T disc = d[e] ? T(0) : gamma;
      o[e] = r[e] + disc * (nv[e] + c * (nout[e] - nq[e]));
    }
  }
}

}  // namespace rlprim
