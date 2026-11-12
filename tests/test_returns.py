"""Every estimator is checked against a slow, obviously-right version on random rollouts."""

import numpy as np
import pytest

import rlprim


def naive_gae(r, v, d, last, gamma, lam):
    steps, envs = r.shape
    adv = np.zeros((steps, envs))
    for e in range(envs):
        for t in range(steps):
            total, coef = 0.0, 1.0
            for k in range(t, steps):
                nv = last[e] if k + 1 == steps else v[k + 1, e]
                alive = 0.0 if d[k, e] else 1.0
                total += coef * (r[k, e] + gamma * nv * alive - v[k, e])
                if d[k, e]:
                    break
                coef *= gamma * lam
            adv[t, e] = total
    return adv, adv + v


def naive_discounted(r, d, last, gamma):
    steps, envs = r.shape
    out = np.zeros((steps, envs))
    for e in range(envs):
        for t in range(steps):
            total, coef, cut = 0.0, 1.0, False
            for k in range(t, steps):
                total += coef * r[k, e]
                coef *= gamma
                if d[k, e]:
                    cut = True
                    break
            out[t, e] = total if cut else total + coef * last[e]
    return out


def naive_nstep(r, v, d, last, n, gamma):
    steps, envs = r.shape
    out = np.zeros((steps, envs))
    for e in range(envs):
        for t in range(steps):
            total, coef, cut = 0.0, 1.0, False
            k = t
            while k < min(t + n, steps):
                total += coef * r[k, e]
                coef *= gamma
                k += 1
                if d[k - 1, e]:
                    cut = True
                    break
            if not cut:
                total += coef * (last[e] if k == steps else v[k, e])
            out[t, e] = total
    return out


def rollout(rng, steps, envs, p_done):
    r = rng.normal(size=(steps, envs))
    v = rng.normal(size=(steps, envs))
    d = rng.random((steps, envs)) < p_done
    last = rng.normal(size=envs)
    return r, v, d, last


CASES = [(1, 1, 0.0), (7, 1, 0.3), (40, 3, 0.1), (64, 5, 0.0), (33, 4, 0.5), (50, 2, 1.0)]


@pytest.mark.parametrize("steps,envs,p_done", CASES)
@pytest.mark.parametrize("gamma,lam", [(0.99, 0.95), (1.0, 1.0), (0.9, 0.0)])
def test_gae_matches_naive(steps, envs, p_done, gamma, lam):
    rng = np.random.default_rng(steps * 100 + envs)
    r, v, d, last = rollout(rng, steps, envs, p_done)
    adv, ret = rlprim.gae(r, v, d, last, gamma=gamma, lam=lam)
    want_adv, want_ret = naive_gae(r, v, d, last, gamma, lam)
    np.testing.assert_allclose(adv, want_adv, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ret, want_ret, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("steps,envs,p_done", CASES)
@pytest.mark.parametrize("gamma", [0.99, 1.0, 0.5])
def test_discounted_returns_match_naive(steps, envs, p_done, gamma):
    rng = np.random.default_rng(steps * 7 + envs)
    r, _, d, last = rollout(rng, steps, envs, p_done)
    got = rlprim.discounted_returns(r, d, last, gamma=gamma)
    np.testing.assert_allclose(got, naive_discounted(r, d, last, gamma), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("steps,envs,p_done", CASES)
@pytest.mark.parametrize("n", [1, 2, 3, 5, 17, 200])
@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_nstep_matches_naive(steps, envs, p_done, n, gamma):
    rng = np.random.default_rng(steps * 13 + envs + n)
    r, v, d, last = rollout(rng, steps, envs, p_done)
    got = rlprim.nstep_returns(r, v, d, last, n=n, gamma=gamma)
    np.testing.assert_allclose(got, naive_nstep(r, v, d, last, n, gamma), rtol=1e-9, atol=1e-9)


def test_nstep_long_rollout_does_not_drift():
    rng = np.random.default_rng(0)
    r, v, d, last = rollout(rng, 20000, 1, 0.0)
    r, v = r.astype(np.float32), v.astype(np.float32)
    got = rlprim.nstep_returns(r, v, d, last, n=5, gamma=1.0)
    want = naive_nstep(r.astype(np.float64), v.astype(np.float64), d, last, 5, 1.0)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_nstep_with_huge_n_equals_discounted_returns():
    rng = np.random.default_rng(1)
    r, v, d, last = rollout(rng, 60, 3, 0.1)
    np.testing.assert_allclose(
        rlprim.nstep_returns(r, v, d, last, n=10_000, gamma=0.97),
        rlprim.discounted_returns(r, d, last, gamma=0.97),
        rtol=1e-10,
        atol=1e-10,
    )


def test_gae_lambda_one_equals_returns_minus_values():
    rng = np.random.default_rng(2)
    r, v, d, last = rollout(rng, 80, 4, 0.05)
    adv, ret = rlprim.gae(r, v, d, last, gamma=0.99, lam=1.0)
    np.testing.assert_allclose(ret, rlprim.discounted_returns(r, d, last, gamma=0.99), atol=1e-10)
    np.testing.assert_allclose(adv, ret - v, atol=1e-10)


def test_1d_input_gives_1d_output_and_matches_2d():
    rng = np.random.default_rng(3)
    r, v, d, last = rollout(rng, 30, 1, 0.1)
    adv2, ret2 = rlprim.gae(r, v, d, last)
    adv1, ret1 = rlprim.gae(r[:, 0], v[:, 0], d[:, 0], float(last[0]))
    assert adv1.shape == (30,)
    np.testing.assert_array_equal(adv1, adv2[:, 0])
    np.testing.assert_array_equal(ret1, ret2[:, 0])


def test_dtypes():
    rng = np.random.default_rng(4)
    r, v, d, last = rollout(rng, 50, 3, 0.1)
    adv64, _ = rlprim.gae(r, v, d, last)
    adv32, ret32 = rlprim.gae(r.astype(np.float32), v.astype(np.float32), d, last)
    assert adv64.dtype == np.float64
    assert adv32.dtype == np.float32 and ret32.dtype == np.float32
    np.testing.assert_allclose(adv32, adv64, rtol=1e-4, atol=1e-4)
    # float dones and python lists work too
    adv_f, _ = rlprim.gae(r, v, d.astype(np.float32), last)
    np.testing.assert_array_equal(adv_f, adv64)
    adv_l, _ = rlprim.gae(r[:, 0].tolist(), v[:, 0].tolist(), d[:, 0].tolist(), float(last[0]))
    np.testing.assert_array_equal(adv_l, adv64[:, 0])


def test_non_contiguous_input():
    rng = np.random.default_rng(5)
    r, v, d, last = rollout(rng, 40, 6, 0.1)
    adv, _ = rlprim.gae(r[::2, ::3], v[::2, ::3], d[::2, ::3], last[::3])
    want, _ = naive_gae(r[::2, ::3], v[::2, ::3], d[::2, ::3], last[::3], 0.99, 0.95)
    np.testing.assert_allclose(adv, want, atol=1e-10)


def test_empty_rollout():
    e = np.zeros((0, 3))
    adv, ret = rlprim.gae(e, e, e.astype(bool), np.zeros(3))
    assert adv.shape == (0, 3) and ret.shape == (0, 3)
    assert rlprim.nstep_returns(e, e, e.astype(bool), np.zeros(3), n=3).shape == (0, 3)


def test_bad_input_raises():
    r = np.zeros((5, 2))
    d = np.zeros((5, 2), dtype=bool)
    with pytest.raises(ValueError):
        rlprim.gae(r, np.zeros((5, 3)), d, 0.0)
    with pytest.raises(ValueError):
        rlprim.gae(r, r, np.zeros((4, 2), dtype=bool), 0.0)
    with pytest.raises(ValueError):
        rlprim.gae(r, r, d, np.zeros(3))
    with pytest.raises(ValueError):
        rlprim.gae(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2, 2), dtype=bool), 0.0)
    with pytest.raises(ValueError):
        rlprim.nstep_returns(r, r, d, 0.0, n=0)


def naive_vtrace(lr, r, v, d, last, gamma, lam, clip_rho, clip_c, clip_pg):
    steps, envs = r.shape
    rho_raw = np.exp(lr)
    rho = np.minimum(clip_rho, rho_raw)
    c = lam * np.minimum(clip_c, rho_raw)
    disc = np.where(d, 0.0, gamma)
    vnext = np.concatenate([v[1:], last[None]], axis=0)
    delta = rho * (r + disc * vnext - v)
    vs = np.zeros((steps, envs))
    for e in range(envs):
        for s in range(steps):
            total, coef = 0.0, 1.0
            for t in range(s, steps):
                total += coef * delta[t, e]
                coef *= disc[t, e] * c[t, e]
            vs[s, e] = v[s, e] + total
    vs_next = np.concatenate([vs[1:], last[None]], axis=0)
    pg = np.minimum(clip_pg, rho_raw) * (r + disc * vs_next - v)
    return vs, pg


def naive_retrace(r, q, v, lr, d, last, gamma, lam):
    steps, envs = r.shape
    c = lam * np.minimum(1.0, np.exp(lr))
    disc = np.where(d, 0.0, gamma)
    vnext = np.concatenate([v[1:], last[None]], axis=0)
    delta = r + disc * vnext - q
    out = np.zeros((steps, envs))
    for e in range(envs):
        for t in range(steps):
            total, coef = 0.0, 1.0
            for s in range(t, steps):
                total += coef * delta[s, e]
                if s + 1 < steps:
                    coef *= disc[s, e] * c[s + 1, e]
            out[t, e] = q[t, e] + total
    return out


@pytest.mark.parametrize("steps,envs,p_done", CASES)
@pytest.mark.parametrize("lam,clips", [(1.0, (1.0, 1.0, 1.0)), (0.9, (2.0, 0.7, 1.5)), (1.0, (1e9, 1e9, 1e9))])
def test_vtrace_matches_naive(steps, envs, p_done, lam, clips):
    rng = np.random.default_rng(steps * 17 + envs)
    r, v, d, last = rollout(rng, steps, envs, p_done)
    lr = rng.normal(scale=0.7, size=(steps, envs))
    vs, pg = rlprim.vtrace(lr, r, v, d, last, gamma=0.97, lam=lam,
                           clip_rho=clips[0], clip_c=clips[1], clip_pg_rho=clips[2])
    want_vs, want_pg = naive_vtrace(lr, r, v, d, last, 0.97, lam, *clips)
    np.testing.assert_allclose(vs, want_vs, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(pg, want_pg, rtol=1e-9, atol=1e-9)


def test_vtrace_on_policy_equals_discounted_returns():
    # with pi == mu every rho is 1, so vs is the plain bootstrapped return
    rng = np.random.default_rng(6)
    r, v, d, last = rollout(rng, 70, 3, 0.05)
    vs, pg = rlprim.vtrace(np.zeros_like(r), r, v, d, last, gamma=0.99)
    want = rlprim.discounted_returns(r, d, last, gamma=0.99)
    np.testing.assert_allclose(vs, want, atol=1e-10)
    np.testing.assert_allclose(pg, want - v, atol=1e-10)


@pytest.mark.parametrize("steps,envs,p_done", CASES)
@pytest.mark.parametrize("lam", [1.0, 0.8, 0.0])
def test_retrace_matches_naive(steps, envs, p_done, lam):
    rng = np.random.default_rng(steps * 19 + envs)
    r, v, d, last = rollout(rng, steps, envs, p_done)
    q = rng.normal(size=(steps, envs))
    lr = rng.normal(scale=0.7, size=(steps, envs))
    got = rlprim.retrace(r, q, v, lr, d, last, gamma=0.97, lam=lam)
    np.testing.assert_allclose(got, naive_retrace(r, q, v, lr, d, last, 0.97, lam), rtol=1e-9, atol=1e-9)


def test_retrace_lambda_zero_is_one_step_target():
    rng = np.random.default_rng(7)
    r, v, d, last = rollout(rng, 40, 2, 0.1)
    q = rng.normal(size=r.shape)
    got = rlprim.retrace(r, q, v, np.zeros_like(r), d, last, gamma=0.9, lam=0.0)
    vnext = np.concatenate([v[1:], last[None]], axis=0)
    np.testing.assert_allclose(got, r + np.where(d, 0.0, 0.9) * vnext, atol=1e-12)


def test_offpolicy_float32_and_errors():
    rng = np.random.default_rng(8)
    r, v, d, last = rollout(rng, 30, 2, 0.1)
    lr = rng.normal(scale=0.5, size=r.shape)
    f = np.float32
    vs32, pg32 = rlprim.vtrace(lr.astype(f), r.astype(f), v.astype(f), d, last)
    vs64, _ = rlprim.vtrace(lr, r, v, d, last)
    assert vs32.dtype == f and pg32.dtype == f
    np.testing.assert_allclose(vs32, vs64, rtol=1e-4, atol=1e-4)
    assert rlprim.retrace(r.astype(f), v.astype(f), v.astype(f), lr.astype(f), d, last).dtype == f
    with pytest.raises(ValueError):
        rlprim.vtrace(lr[:-1], r, v, d, last)
    with pytest.raises(ValueError):
        rlprim.retrace(r, v[:, :1], v, lr, d, last)
