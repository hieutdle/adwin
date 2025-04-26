#include <adwin.h>
#include <cmath>
#include <numeric>
#include <pybind11/pybind11.h>

namespace py = pybind11;

double Adwin::mean(const std::vector<double> &s) {
  return std::accumulate(s.begin(), s.end(), 0.0) / s.size();
}

double Adwin::variance(const std::vector<double> &s) {
  double mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
  double var = 0.0;
  for (double value : s) {
    var += std::pow(value - mean, 2);
  }
  return var / s.size();
}

double Adwin::harmonic_mean(double n1, double n2) {
  if (n1 == 0 || n2 == 0) {
    return 0;
  }
  return 1.0 / (1.0 / n1 + 1.0 / n2);
}

double Adwin::epsilonConservative(double delta, int n, int n1, int n2) {
  double hm = harmonic_mean(double(n1), double(n2));
  double deltaDash = delta / n;
  return std::sqrt((1.0 / (2.0 * hm)) * std::log(4.0 / deltaDash));
}

double Adwin::epsilon(double v, double delta, int n, int n1, int n2) {
  double hm = harmonic_mean(double(n1), double(n2));
  double deltaDash = delta / n;
  double l = std::log(2.0 / deltaDash);
  return std::sqrt((2 / hm) * v * l) + (2 / (3.0 * hm)) * l;
}

bool Adwin::should_drop() {
  if (window.size() < 2) {
    return false;
  }

  // Use variance only if not conservative
  double v = 0.0;
  if (!conservative) {
    v = variance(window);
  }

  size_t n = window.size();
  // Loop through the window and check for potential change points
  for (size_t i = 1; i < n; ++i) {
    std::vector<double> w0(window.begin(), window.begin() + i);
    std::vector<double> w1(window.begin() + i, window.end());

    size_t n0 = w0.size();
    size_t n1 = w1.size();
    double e = 0.0;

    if (conservative) {
      e = epsilonConservative(delta, window.size(), n0,
                              n1); // Conservative epsilon
    } else {
      e = epsilon(v, delta, window.size(), n0, n1); // Non-conservative epsilon
    }

    // Calculate the means of the two windows
    double muw0 = mean(w0);
    double muw1 = mean(w1);

    // If the difference between means is greater than the epsilon, declare a
    // change
    if (std::abs(muw0 - muw1) >= e) {
      return true;
    }
  }

  return false;
}

int Adwin::add(double x) {
  x -= min_value;
  x /= (max_value - min_value);
  window.push_back(x);
  int count = 0;
  while (should_drop()) {
    window.erase(window.begin());
    ++count;
  }
  return count;
}

std::vector<std::pair<int, int>> Adwin::parse(const py::iterable &s) {
  std::vector<double> data =
      py::cast<std::vector<double>>(s); // Convert Python iterable to C++ vector
  std::vector<std::pair<int, int>> result;
  for (double x : data) {
    result.push_back({add(x), window.size()});
  }
  return result;
}

// Binding Adwin class to Python
PYBIND11_MODULE(adwin, m) {
  m.doc() = "Adwin module";

  py::class_<Adwin>(m, "Adwin")
      .def(py::init<double, double, double, bool>(), py::arg("delta"),
           py::arg("min_value"), py::arg("max_value"),
           py::arg("conservative") = false)
      .def("add", &Adwin::add)
      .def("parse", &Adwin::parse)
      .def("should_drop", &Adwin::should_drop)
      .def("mean", &Adwin::mean)
      .def("variance", &Adwin::variance)
      .def("harmonic_mean", &Adwin::harmonic_mean)
      .def("epsilonConservative", &Adwin::epsilonConservative)
      .def("epsilon", &Adwin::epsilon)
      .def("should_drop", &Adwin::should_drop);
}