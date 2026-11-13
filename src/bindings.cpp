#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <rlprim/normalize.hpp>
#include <rlprim/returns.hpp>
#include <rlprim/sampler.hpp>
#include <rlprim/sum_tree.hpp>

namespace py = pybind11;

namespace {

template <class T>
using carr = py::array_t<T, py::array::c_style | py::array::forcecast>;

bool is_float32(const py::object& o) {
  return py::isinstance<py::array>(o) &&
         py::reinterpret_borrow<py::array>(o).dtype().equal(py::dtype::of<float>());
}

struct Dims {
  std::size_t steps;
  std::size_t envs;
};

Dims dims_of(const py::array& a, const char* name) {
  if (a.ndim() == 1) return {static_cast<std::size_t>(a.shape(0)), 1};
  if (a.ndim() == 2) return {static_cast<std::size_t>(a.shape(0)), static_cast<std::size_t>(a.shape(1))};
  throw py::value_error(std::string(name) + " must have shape (steps,) or (steps, envs)");
}

void require_same_shape(const py::array& a, const py::array& b, const char* name) {
  bool same = a.ndim() == b.ndim();
  for (py::ssize_t i = 0; same && i < a.ndim(); ++i) same = a.shape(i) == b.shape(i);
  if (!same) throw py::value_error(std::string(name) + " must have the same shape as rewards");
}

template <class T>
carr<T> as_array(const py::object& o, const char* name) {
  auto a = carr<T>::ensure(o);
  if (!a) {
    PyErr_Clear();
    throw py::value_error(std::string(name) + " can't be read as a numeric array");
  }
  return a;
}

template <class T>
std::vector<T> last_values_of(const py::object& o, std::size_t envs) {
  auto a = as_array<T>(o, "last_value");
  if (a.size() == 1) return std::vector<T>(envs, *a.data());
  if (static_cast<std::size_t>(a.size()) != envs)
    throw py::value_error("last_value must be a scalar or have one entry per env");
  return std::vector<T>(a.data(), a.data() + envs);
}

template <class T>
py::array_t<T> like(const py::array& a) {
  return py::array_t<T>(std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim()));
}

const std::uint8_t* bytes(const carr<bool>& d) {
  return reinterpret_cast<const std::uint8_t*>(d.data());
}

template <class T>
py::tuple gae_impl(const py::object& rewards, const py::object& values, const py::object& dones,
                   const py::object& last_value, double gamma, double lam) {
  auto r = as_array<T>(rewards, "rewards");
  auto v = as_array<T>(values, "values");
  auto d = as_array<bool>(dones, "dones");
  require_same_shape(r, v, "values");
  require_same_shape(r, d, "dones");
  const Dims dm = dims_of(r, "rewards");
  const auto last = last_values_of<T>(last_value, dm.envs);
  auto adv = like<T>(r);
  auto ret = like<T>(r);
  {
    py::gil_scoped_release release;
    rlprim::gae<T>(r.data(), v.data(), bytes(d), last.data(), dm.steps, dm.envs,
                   static_cast<T>(gamma), static_cast<T>(lam), adv.mutable_data(),
                   ret.mutable_data());
  }
  return py::make_tuple(adv, ret);
}

template <class T>
py::array discounted_impl(const py::object& rewards, const py::object& dones,
                          const py::object& last_value, double gamma) {
  auto r = as_array<T>(rewards, "rewards");
  auto d = as_array<bool>(dones, "dones");
  require_same_shape(r, d, "dones");
  const Dims dm = dims_of(r, "rewards");
  const auto last = last_values_of<T>(last_value, dm.envs);
  auto out = like<T>(r);
  {
    py::gil_scoped_release release;
    rlprim::discounted_returns<T>(r.data(), bytes(d), last.data(), dm.steps, dm.envs,
                                  static_cast<T>(gamma), out.mutable_data());
  }
  return std::move(out);
}

template <class T>
py::array nstep_impl(const py::object& rewards, const py::object& values, const py::object& dones,
                     const py::object& last_value, std::size_t n, double gamma) {
  auto r = as_array<T>(rewards, "rewards");
  auto v = as_array<T>(values, "values");
  auto d = as_array<bool>(dones, "dones");
  require_same_shape(r, v, "values");
  require_same_shape(r, d, "dones");
  const Dims dm = dims_of(r, "rewards");
  const auto last = last_values_of<T>(last_value, dm.envs);
  auto out = like<T>(r);
  {
    py::gil_scoped_release release;
    rlprim::nstep_returns<T>(r.data(), v.data(), bytes(d), last.data(), dm.steps, dm.envs, n,
                             static_cast<T>(gamma), out.mutable_data());
  }
  return std::move(out);
}

template <class T>
py::tuple vtrace_impl(const py::object& log_rhos, const py::object& rewards,
                      const py::object& values, const py::object& dones,
                      const py::object& last_value, double gamma, double lam, double clip_rho,
                      double clip_c, double clip_pg_rho) {
  auto r = as_array<T>(rewards, "rewards");
  auto lr = as_array<T>(log_rhos, "log_rhos");
  auto v = as_array<T>(values, "values");
  auto d = as_array<bool>(dones, "dones");
  require_same_shape(r, lr, "log_rhos");
  require_same_shape(r, v, "values");
  require_same_shape(r, d, "dones");
  const Dims dm = dims_of(r, "rewards");
  const auto last = last_values_of<T>(last_value, dm.envs);
  auto vs = like<T>(r);
  auto pg = like<T>(r);
  {
    py::gil_scoped_release release;
    rlprim::vtrace<T>(lr.data(), r.data(), v.data(), bytes(d), last.data(), dm.steps, dm.envs,
                      static_cast<T>(gamma), static_cast<T>(lam), static_cast<T>(clip_rho),
                      static_cast<T>(clip_c), static_cast<T>(clip_pg_rho), vs.mutable_data(),
                      pg.mutable_data());
  }
  return py::make_tuple(vs, pg);
}

template <class T>
py::array retrace_impl(const py::object& rewards, const py::object& q_values,
                       const py::object& values, const py::object& log_rhos,
                       const py::object& dones, const py::object& last_value, double gamma,
                       double lam) {
  auto r = as_array<T>(rewards, "rewards");
  auto q = as_array<T>(q_values, "q_values");
  auto v = as_array<T>(values, "values");
  auto lr = as_array<T>(log_rhos, "log_rhos");
  auto d = as_array<bool>(dones, "dones");
  require_same_shape(r, q, "q_values");
  require_same_shape(r, v, "values");
  require_same_shape(r, lr, "log_rhos");
  require_same_shape(r, d, "dones");
  const Dims dm = dims_of(r, "rewards");
  const auto last = last_values_of<T>(last_value, dm.envs);
  auto out = like<T>(r);
  {
    py::gil_scoped_release release;
    rlprim::retrace<T>(r.data(), q.data(), v.data(), lr.data(), bytes(d), last.data(), dm.steps,
                       dm.envs, static_cast<T>(gamma), static_cast<T>(lam), out.mutable_data());
  }
  return std::move(out);
}

struct PyNorm {
  std::vector<py::ssize_t> shape;
  rlprim::RunningNorm core;
};

PyNorm make_norm(const py::object& shape, double epsilon) {
  std::vector<py::ssize_t> dims;
  if (py::isinstance<py::int_>(shape)) {
    dims.push_back(shape.cast<py::ssize_t>());
  } else {
    for (auto item : py::reinterpret_borrow<py::tuple>(py::tuple(shape))) dims.push_back(item.cast<py::ssize_t>());
  }
  std::size_t dim = 1;
  for (auto v : dims) {
    if (v < 1) throw py::value_error("every dimension of shape must be >= 1");
    dim *= static_cast<std::size_t>(v);
  }
  if (!(epsilon > 0.0)) throw py::value_error("epsilon must be > 0");
  return PyNorm{dims, rlprim::RunningNorm(dim, epsilon)};
}

// Rows in `a`, which holds either one item of `shape` or a batch of them.
std::size_t norm_rows(const py::array& a, const std::vector<py::ssize_t>& shape) {
  const auto nd = static_cast<py::ssize_t>(shape.size());
  const py::ssize_t off = a.ndim() - nd;
  bool ok = off == 0 || off == 1;
  for (py::ssize_t i = 0; ok && i < nd; ++i) ok = a.shape(i + off) == shape[static_cast<std::size_t>(i)];
  if (!ok) throw py::value_error("x must have the normalizer's shape, or a batch of it");
  return off == 1 ? static_cast<std::size_t>(a.shape(0)) : 1;
}

template <class T>
void norm_update(PyNorm& n, const py::object& x) {
  auto a = as_array<T>(x, "x");
  const std::size_t rows = norm_rows(a, n.shape);
  py::gil_scoped_release release;
  n.core.update<T>(a.data(), rows);
}

template <class T>
py::array norm_apply(const PyNorm& n, const py::object& x, double clip) {
  auto a = as_array<T>(x, "x");
  const std::size_t rows = norm_rows(a, n.shape);
  auto out = like<T>(a);
  {
    py::gil_scoped_release release;
    n.core.normalize<T>(a.data(), rows, out.mutable_data(), clip);
  }
  return std::move(out);
}

py::array_t<double> norm_stat(const PyNorm& n, const std::vector<double>& v) {
  py::array_t<double> out(n.shape);
  std::copy(v.begin(), v.end(), out.mutable_data());
  return out;
}

void check_index(std::int64_t i, std::size_t capacity) {
  if (i < 0 || static_cast<std::size_t>(i) >= capacity)
    throw py::index_error("index " + std::to_string(i) + " is out of range");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "rlprim C++ core";

  m.def(
      "gae",
      [](const py::object& rewards, const py::object& values, const py::object& dones,
         const py::object& last_value, double gamma, double lam) {
        return is_float32(rewards) ? gae_impl<float>(rewards, values, dones, last_value, gamma, lam)
                                   : gae_impl<double>(rewards, values, dones, last_value, gamma, lam);
      },
      py::arg("rewards"), py::arg("values"), py::arg("dones"), py::arg("last_value") = 0.0,
      py::arg("gamma") = 0.99, py::arg("lam") = 0.95,
      "Generalized Advantage Estimation. Returns (advantages, returns).\n\n"
      "rewards, values and dones share the shape (steps,) or (steps, envs).\n"
      "dones[t] marks an episode that ended at step t. last_value is the value\n"
      "of the state after the final step, a scalar or one entry per env.\n"
      "float32 rewards give float32 outputs, anything else gives float64.");

  m.def(
      "discounted_returns",
      [](const py::object& rewards, const py::object& dones, const py::object& last_value,
         double gamma) {
        return is_float32(rewards) ? discounted_impl<float>(rewards, dones, last_value, gamma)
                                   : discounted_impl<double>(rewards, dones, last_value, gamma);
      },
      py::arg("rewards"), py::arg("dones"), py::arg("last_value") = 0.0, py::arg("gamma") = 0.99,
      "Discounted reward-to-go, cut at episode ends and bootstrapped from last_value.");

  m.def(
      "nstep_returns",
      [](const py::object& rewards, const py::object& values, const py::object& dones,
         const py::object& last_value, py::ssize_t n, double gamma) {
        if (n < 1) throw py::value_error("n must be >= 1");
        const auto un = static_cast<std::size_t>(n);
        return is_float32(rewards) ? nstep_impl<float>(rewards, values, dones, last_value, un, gamma)
                                   : nstep_impl<double>(rewards, values, dones, last_value, un, gamma);
      },
      py::arg("rewards"), py::arg("values"), py::arg("dones"), py::arg("last_value") = 0.0,
      py::arg("n") = 3, py::arg("gamma") = 0.99,
      "n-step bootstrapped returns in O(steps), whatever n is.");

  m.def(
      "vtrace",
      [](const py::object& log_rhos, const py::object& rewards, const py::object& values,
         const py::object& dones, const py::object& last_value, double gamma, double lam,
         double clip_rho, double clip_c, double clip_pg_rho) {
        return is_float32(rewards)
                   ? vtrace_impl<float>(log_rhos, rewards, values, dones, last_value, gamma, lam,
                                        clip_rho, clip_c, clip_pg_rho)
                   : vtrace_impl<double>(log_rhos, rewards, values, dones, last_value, gamma, lam,
                                         clip_rho, clip_c, clip_pg_rho);
      },
      py::arg("log_rhos"), py::arg("rewards"), py::arg("values"), py::arg("dones"),
      py::arg("last_value") = 0.0, py::arg("gamma") = 0.99, py::arg("lam") = 1.0,
      py::arg("clip_rho") = 1.0, py::arg("clip_c") = 1.0, py::arg("clip_pg_rho") = 1.0,
      "V-trace targets from IMPALA. Returns (vs, pg_advantages).\n\n"
      "log_rhos = log pi(a|s) - log mu(a|s), target policy over behaviour policy.");

  m.def(
      "retrace",
      [](const py::object& rewards, const py::object& q_values, const py::object& values,
         const py::object& log_rhos, const py::object& dones, const py::object& last_value,
         double gamma, double lam) {
        return is_float32(rewards)
                   ? retrace_impl<float>(rewards, q_values, values, log_rhos, dones, last_value,
                                         gamma, lam)
                   : retrace_impl<double>(rewards, q_values, values, log_rhos, dones, last_value,
                                          gamma, lam);
      },
      py::arg("rewards"), py::arg("q_values"), py::arg("values"), py::arg("log_rhos"),
      py::arg("dones"), py::arg("last_value") = 0.0, py::arg("gamma") = 0.99, py::arg("lam") = 1.0,
      "Retrace(lambda) Q targets.\n\n"
      "q_values[t] = Q(s_t, a_t), values[t] = E_pi Q(s_t, .), last_value = E_pi Q of the\n"
      "state after the final step, log_rhos = log pi(a|s) - log mu(a|s).");

  py::class_<PyNorm>(m, "RunningNorm",
                     "Running mean and variance per feature, for observation or reward normalization.\n\n"
                     "Same update rule and starting state as Gymnasium's RunningMeanStd.")
      .def(py::init(&make_norm), py::arg("shape"), py::arg("epsilon") = 1e-4)
      .def_property_readonly("count", [](const PyNorm& n) { return n.core.count(); })
      .def_property_readonly("mean", [](const PyNorm& n) { return norm_stat(n, n.core.mean()); })
      .def_property_readonly("var", [](const PyNorm& n) { return norm_stat(n, n.core.var()); })
      .def(
          "update",
          [](PyNorm& n, const py::object& x) {
            if (is_float32(x)) norm_update<float>(n, x);
            else norm_update<double>(n, x);
          },
          py::arg("x"), "Fold in one item or a batch of items.")
      .def(
          "normalize",
          [](const PyNorm& n, const py::object& x, double clip) {
            if (!(clip > 0.0)) throw py::value_error("clip must be > 0");
            return is_float32(x) ? norm_apply<float>(n, x, clip) : norm_apply<double>(n, x, clip);
          },
          py::arg("x"), py::arg("clip") = 10.0,
          "clip((x - mean) / sqrt(var + 1e-8), -clip, clip), same shape and float type as x.");

  py::class_<rlprim::SumTree>(m, "SumTree",
                              "Sum tree over non-negative weights with O(log n) update and lookup.")
      .def(py::init([](py::ssize_t capacity) {
             if (capacity < 1) throw py::value_error("capacity must be >= 1");
             return rlprim::SumTree(static_cast<std::size_t>(capacity));
           }),
           py::arg("capacity"))
      .def_property_readonly("capacity", &rlprim::SumTree::capacity)
      .def_property_readonly("total", &rlprim::SumTree::total)
      .def(
          "set",
          [](rlprim::SumTree& t, const py::object& indices, const py::object& values) {
            auto idx = as_array<std::int64_t>(indices, "indices");
            auto val = as_array<double>(values, "values");
            if (idx.size() != val.size())
              throw py::value_error("indices and values must have the same length");
            const auto* ip = idx.data();
            const auto* vp = val.data();
            for (py::ssize_t k = 0; k < idx.size(); ++k) {
              check_index(ip[k], t.capacity());
              if (!(vp[k] >= 0.0) || !std::isfinite(vp[k]))
                throw py::value_error("weights must be finite and >= 0");
            }
            for (py::ssize_t k = 0; k < idx.size(); ++k) t.set(static_cast<std::size_t>(ip[k]), vp[k]);
          },
          py::arg("indices"), py::arg("values"), "Set weights at the given indices.")
      .def(
          "get",
          [](const rlprim::SumTree& t, const py::object& indices) {
            auto idx = as_array<std::int64_t>(indices, "indices");
            py::array_t<double> out(idx.size());
            const auto* ip = idx.data();
            auto* op = out.mutable_data();
            for (py::ssize_t k = 0; k < idx.size(); ++k) {
              check_index(ip[k], t.capacity());
              op[k] = t.get(static_cast<std::size_t>(ip[k]));
            }
            return out;
          },
          py::arg("indices"), "Weights at the given indices.")
      .def(
          "find",
          [](const rlprim::SumTree& t, const py::object& prefixes) {
            auto u = as_array<double>(prefixes, "prefixes");
            if (!(t.total() > 0.0)) throw py::value_error("the tree is empty");
            py::array_t<std::int64_t> out(u.size());
            t.find_many(u.data(), static_cast<std::size_t>(u.size()), out.mutable_data());
            return out;
          },
          py::arg("prefixes"),
          "For each prefix mass u in [0, total), the index i where the running sum first passes u.");

  py::class_<rlprim::PrioritizedSampler>(
      m, "PrioritizedSampler",
      "Proportional prioritized replay sampling over the slot indices of a buffer.")
      .def(py::init([](py::ssize_t capacity, double alpha, std::uint64_t seed) {
             if (capacity < 1) throw py::value_error("capacity must be >= 1");
             if (!(alpha >= 0.0)) throw py::value_error("alpha must be >= 0");
             return rlprim::PrioritizedSampler(static_cast<std::size_t>(capacity), alpha, seed);
           }),
           py::arg("capacity"), py::arg("alpha") = 0.6, py::arg("seed") = 0)
      .def_property_readonly("capacity", &rlprim::PrioritizedSampler::capacity)
      .def_property_readonly("alpha", &rlprim::PrioritizedSampler::alpha)
      .def_property_readonly("total", &rlprim::PrioritizedSampler::total)
      .def_property_readonly("max_priority", &rlprim::PrioritizedSampler::max_priority)
      .def("__len__", &rlprim::PrioritizedSampler::size)
      .def(
          "update",
          [](rlprim::PrioritizedSampler& s, const py::object& indices, const py::object& priorities) {
            auto idx = as_array<std::int64_t>(indices, "indices");
            auto pri = as_array<double>(priorities, "priorities");
            const bool broadcast = pri.size() == 1;
            if (!broadcast && idx.size() != pri.size())
              throw py::value_error("priorities must be a scalar or match indices in length");
            const auto* ip = idx.data();
            const auto* pp = pri.data();
            const py::ssize_t np_ = pri.size();
            for (py::ssize_t k = 0; k < idx.size(); ++k) check_index(ip[k], s.capacity());
            for (py::ssize_t k = 0; k < np_; ++k)
              if (!(pp[k] > 0.0) || !std::isfinite(pp[k]))
                throw py::value_error("priorities must be finite and > 0");
            for (py::ssize_t k = 0; k < idx.size(); ++k)
              s.update(static_cast<std::size_t>(ip[k]), broadcast ? pp[0] : pp[k]);
          },
          py::arg("indices"), py::arg("priorities"),
          "Set raw priorities for the given slots. A scalar priority applies to all of them.")
      .def(
          "remove",
          [](rlprim::PrioritizedSampler& s, const py::object& indices) {
            auto idx = as_array<std::int64_t>(indices, "indices");
            const auto* ip = idx.data();
            for (py::ssize_t k = 0; k < idx.size(); ++k) check_index(ip[k], s.capacity());
            for (py::ssize_t k = 0; k < idx.size(); ++k) s.remove(static_cast<std::size_t>(ip[k]));
          },
          py::arg("indices"), "Empty the given slots so they can't be sampled.")
      .def(
          "priorities",
          [](const rlprim::PrioritizedSampler& s, const py::object& indices) {
            auto idx = as_array<std::int64_t>(indices, "indices");
            py::array_t<double> out(idx.size());
            const auto* ip = idx.data();
            auto* op = out.mutable_data();
            for (py::ssize_t k = 0; k < idx.size(); ++k) {
              check_index(ip[k], s.capacity());
              op[k] = s.scaled_priority(static_cast<std::size_t>(ip[k]));
            }
            return out;
          },
          py::arg("indices"), "Stored priority**alpha for the given slots, 0 for empty ones.")
      .def(
          "sample",
          [](rlprim::PrioritizedSampler& s, py::ssize_t batch_size, double beta) {
            if (batch_size < 1) throw py::value_error("batch_size must be >= 1");
            if (!(beta >= 0.0)) throw py::value_error("beta must be >= 0");
            if (s.size() == 0) throw py::value_error("can't sample from an empty sampler");
            py::array_t<std::int64_t> idx(batch_size);
            py::array_t<double> w(batch_size);
            s.sample(static_cast<std::size_t>(batch_size), beta, idx.mutable_data(), w.mutable_data());
            return py::make_tuple(idx, w);
          },
          py::arg("batch_size"), py::arg("beta") = 0.4,
          "Stratified sample. Returns (indices, importance_weights), weights scaled to at most 1.");
}
