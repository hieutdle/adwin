#pragma once

#ifndef ADWIN_H
#define ADWIN_H

#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

class Adwin {
public:
  Adwin(double delta, double min_value, double max_value,
        bool conservative = false)
      : delta(delta), min_value(min_value), max_value(max_value),
        conservative(conservative) {}

  int add(double x);
  std::vector<std::pair<int, int>> parse(const py::iterable &s);

  double mean(const std::vector<double> &s);
  double variance(const std::vector<double> &s);
  double harmonic_mean(double n1, double n2);
  double epsilonConservative(double delta, int n, int n1, int n2);
  double epsilon(double variance, double delta, int n, int n1, int n2);
  bool should_drop();

private:
  bool conservative;
  double delta;
  double min_value;
  double max_value;
  std::vector<double> window;
};

#endif // ADWIN_H
