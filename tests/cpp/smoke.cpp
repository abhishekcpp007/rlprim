// Proves the headers work on their own, with no Python around.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <rlprim/returns.hpp>
#include <rlprim/sampler.hpp>
#include <rlprim/sum_tree.hpp>

#define CHECK(cond)                                              \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                  \
    }                                                            \
  } while (0)

static bool near(double a, double b) { return std::fabs(a - b) < 1e-12; }

int main() {
  // sum tree
  rlprim::SumTree tree(5);
  tree.set(0, 1.0);
  tree.set(3, 2.0);
  tree.set(4, 1.0);
  CHECK(tree.total() == 4.0);
  CHECK(tree.find(0.0) == 0);
  CHECK(tree.find(0.999) == 0);
  CHECK(tree.find(1.0) == 3);
  CHECK(tree.find(2.999) == 3);
  CHECK(tree.find(3.0) == 4);
  CHECK(tree.find(100.0) == 4);

  const double us[] = {0.0, 0.999, 1.0, 2.999, 3.0, 100.0};
  std::int64_t found[6];
  tree.find_many(us, 6, found);
  for (int k = 0; k < 6; ++k) CHECK(found[k] == static_cast<std::int64_t>(tree.find(us[k])));

  // V-trace with pi == mu is the plain bootstrapped return
  {
    const std::vector<double> lr = {0.0, 0.0}, rr = {1.0, 2.0}, vv = {0.3, 0.4};
    const std::vector<std::uint8_t> dd = {0, 0};
    const double boot = 4.0;
    std::vector<double> vs(2), pg(2);
    rlprim::vtrace<double>(lr.data(), rr.data(), vv.data(), dd.data(), &boot, 2, 1, 0.5, 1.0, 1.0,
                           1.0, 1.0, vs.data(), pg.data());
    CHECK(near(vs[1], 2.0 + 0.5 * 4.0) && near(vs[0], 1.0 + 0.5 * 4.0));
    CHECK(near(pg[0], 1.0 + 0.5 * vs[1] - 0.3));
  }

  // GAE on one env, episode ends at step 1
  const std::vector<double> r = {1.0, 2.0, 3.0};
  const std::vector<double> v = {0.5, 0.25, 1.0};
  const std::vector<std::uint8_t> d = {0, 1, 0};
  const double last = 2.0;
  std::vector<double> adv(3), ret(3);
  rlprim::gae<double>(r.data(), v.data(), d.data(), &last, 3, 1, 0.5, 0.5, adv.data(), ret.data());
  const double a2 = 3.0 + 0.5 * 2.0 - 1.0;
  const double a1 = 2.0 - 0.25;
  const double a0 = (1.0 + 0.5 * 0.25 - 0.5) + 0.25 * a1;
  CHECK(near(adv[2], a2) && near(adv[1], a1) && near(adv[0], a0));
  CHECK(near(ret[0], a0 + 0.5));

  // 2-step returns on the same rollout
  std::vector<double> ns(3);
  rlprim::nstep_returns<double>(r.data(), v.data(), d.data(), &last, 3, 1, 2, 0.5, ns.data());
  CHECK(near(ns[0], 1.0 + 0.5 * 2.0));
  CHECK(near(ns[1], 2.0));
  CHECK(near(ns[2], 3.0 + 0.5 * 2.0));

  // sampler
  rlprim::PrioritizedSampler sampler(100, 1.0, 42);
  sampler.update(10, 1.0);
  sampler.update(20, 3.0);
  CHECK(sampler.size() == 2);
  std::vector<std::int64_t> idx(1000);
  std::vector<double> w(1000);
  sampler.sample(1000, 1.0, idx.data(), w.data());
  int twenty = 0;
  for (std::size_t k = 0; k < idx.size(); ++k) {
    CHECK(idx[k] == 10 || idx[k] == 20);
    CHECK(near(w[k], idx[k] == 10 ? 1.0 : 1.0 / 3.0));
    twenty += idx[k] == 20;
  }
  CHECK(twenty >= 745 && twenty <= 755);  // stratified, so very close to 750

  std::printf("rlprim smoke test passed\n");
  return 0;
}
