#ifndef CONJUGATEGRADIENT_CLASS
#define CONJUGATEGRADIENT_CLASS

#include "optimization.hpp"

class conjugateGradient : public algorithm_base {
public:
  conjugateGradient() { init_control(); }
  void solve();
  
private:
  tangent_vector XI_desc_full, XI_desc_forward;
  double tol, alpha, beta, sigma;
  void init_control();
  void solve_init();
  void armijo_rule(double expected_desc);
  
};

#endif
