#ifndef TRUST_REGION_CLASS
#define TRUST_REGION_CLASS

#include "optimization.hpp"

class trustRegion : public algorithm_base {
public:
  trustRegion() { init_control(); }
  void solve();

private:
  void init_control();
  void solve_init();

  double rhoMin, tol, theta, kappa, DeltaMax;

  // trust region method
  double rrNorm, rrNorm0, XI_norm, hessQterm;
  tangent_vector dd, XI_desc_old, rr, dd_old;
  // the sub problem of trust region, using truncated conjugate gradient
  bool tR_subproblem(double Delta);
  void tR_subp_initialVecs();
  void tR_subp_updateVecs();
  // find tau, such that (eta + dd*tau) has norm Delta
  void tR_boundary_XIDesc( tangent_vector &eta,  tangent_vector &dd,
                          double Delta);
  // second order approximation of the objective function
  double tR_secondOrderApprox( tangent_vector &eta);
  // resize trust region
  double tR_updateDelta(double Delta, double rho, bool bdryTouched);
};

#endif
