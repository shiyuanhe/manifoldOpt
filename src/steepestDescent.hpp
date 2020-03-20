#ifndef STEEPEST_DESCENT_CLASS
#define STEEPEST_DESCENT_CLASS

#include "optimization.hpp"

class steepestDescent : public algorithm_base {
public:
  steepestDescent() { init_control(); }
  void solve();

private:
  double tol, alpha, beta, sigma;
  void init_control();
  void solve_init();
  void armijo_rule(double expected_desc);
};

#endif
