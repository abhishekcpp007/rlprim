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
#include <vector>

#if defined(_MSC_VER)
#define RLPRIM_RESTRICT __restrict
#else
#define RLPRIM_RESTRICT __restrict__
#endif

namespace rlprim {

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
// Runs in O(steps * envs), not O(steps * envs * n): each env's window sum
// slides backward and drops one reward when it grows past n. The running sum is kept in
// double so the subtraction doesn't drift for float32 inputs.
template <class T>
void nstep_returns(const T* rewards, const T* values, const std::uint8_t* dones,
                   const T* last_values, std::size_t steps, std::size_t envs, std::size_t n,
                   T gamma, T* out) {
  if (n == 0 || steps == 0) return;
  const double g = static_cast<double>(gamma);
  std::vector<double> gpow(n + 1);
  gpow[0] = 1.0;
  for (std::size_t k = 1; k <= n; ++k) gpow[k] = gpow[k - 1] * g;

  // Per-env window state, walked row by row so memory access stays sequential.
  // The inner loop is branch-free: each of the three cases (episode end,
  // window full, window still growing) is picked with selects.
  std::vector<double> window_buf(envs, 0.0);     // discounted reward sum over [t, end)
  std::vector<std::size_t> end_buf(envs, steps);  // exclusive end of the window
  std::vector<double> open_buf(envs, 1.0);        // 0 when the window stops at an episode end
  double* RLPRIM_RESTRICT window = window_buf.data();
  std::size_t* RLPRIM_RESTRICT end = end_buf.data();
  double* RLPRIM_RESTRICT open = open_buf.data();
  const double* RLPRIM_RESTRICT gp = gpow.data();
  const double gdrop = gpow[n - 1];

  for (std::size_t t = steps; t-- > 0;) {
    const T* r = rewards + t * envs;
    const std::uint8_t* d = dones + t * envs;
    // when t + n is past the rollout no window can be full, so any valid row works
    const T* dropped = (t + n < steps) ? rewards + (t + n) * envs : r;
    T* o = out + t * envs;
    for (std::size_t e = 0; e < envs; ++e) {
      const bool done = d[e] != 0;
      const std::size_t en = end[e];
      const bool full = !done && (en - t > n);
      const double re = static_cast<double>(r[e]);
      const double sub = full ? gdrop * static_cast<double>(dropped[e]) : 0.0;
      const double w = re + (done ? 0.0 : g) * (window[e] - sub);
      const std::size_t en2 = done ? t + 1 : (full ? t + n : en);
      const double op = done ? 0.0 : (full ? 1.0 : open[e]);
      window[e] = w;
      end[e] = en2;
      open[e] = op;
      const double boot = (en2 == steps) ? static_cast<double>(last_values[e])
                                         : static_cast<double>(values[en2 * envs + e]);
      o[e] = static_cast<T>(w + op * gp[en2 - t] * boot);
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
    const T* next_v = last ? last_values : values + (t + 1) * envs;
    const T* next_vs = last ? last_values : vs + (t + 1) * envs;
    const std::size_t row = t * envs;
    for (std::size_t e = 0; e < envs; ++e) {
      const T rho_raw = std::exp(log_rhos[row + e]);
      const T rho = rho_raw < clip_rho ? rho_raw : clip_rho;
      const T c = lam * (rho_raw < clip_c ? rho_raw : clip_c);
      const T rho_pg = rho_raw < clip_pg_rho ? rho_raw : clip_pg_rho;
      const T disc = dones[row + e] ? T(0) : gamma;
      const T r = rewards[row + e];
      const T v = values[row + e];
      const T delta = rho * (r + disc * next_v[e] - v);
      vs[row + e] = v + delta + disc * c * (next_vs[e] - next_v[e]);
      pg_advantages[row + e] = rho_pg * (r + disc * next_vs[e] - v);
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
    const std::size_t nxt = row + envs;
    for (std::size_t e = 0; e < envs; ++e) {
      const T rho = std::exp(log_rhos[nxt + e]);
      const T c = lam * (rho < T(1) ? rho : T(1));
      const T disc = dones[row + e] ? T(0) : gamma;
      out[row + e] = rewards[row + e] + disc * (values[nxt + e] + c * (out[nxt + e] - q_values[nxt + e]));
    }
  }
}

}  // namespace rlprim
