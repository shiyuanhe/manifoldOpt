#ifndef AUGMENTED_LAGRANGIAN_CLASS
#define AUGMENTED_LAGRANGIAN_CLASS

#include "optimization.hpp"
#include "steepestDescent.hpp"
#include "trustRegion.hpp"

class augmentedLagrangian : public algorithm_base {
public:
  augmentedLagrangian();

  void solve();

private:
  // internel parameters
  double n, p, size_scale;
  double tol_dual, tol_primal, rho, beta;
  arma::mat G, Lagr, X, X_pre;

  shared_ptr<FunPlusPenaltyClass> subp_loss;
  shared_ptr<FunPlusPenaltyClass> optFunctn_downcast;

  // overwrite base
  // X doesn't belong to manifold, and is sparse
  mat get_currentY_secondary() { return X; }
  
  
  void init_control();

  void ALM_Init();
  void subprob_Init();
  void Lagr_Init();
  void params_Init();
  void threshFun_Init();

  void ALM_Outer_Loop();
  void ALM_Inner_Loop();
  void armijo_rule(double expected_desc);
};

#endif
