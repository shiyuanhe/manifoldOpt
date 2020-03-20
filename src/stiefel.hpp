#ifndef STIEFEL_CLASS
#define STIEFEL_CLASS

#include "manifold.hpp"

class stiefel : public manifold {
public:
  stiefel(unsigned, unsigned, unsigned);

  void initial_point(vecMatIter rObj);

  auto evalGradient(vecMatIter amb_grad) -> tangent_vector;

  arma::mat evalNormal(const arma::mat &ambM);

  auto evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                   tangent_vector &Z_tv) -> tangent_vector;

  auto retraction(tangent_vector &) -> manifold_ptr;
  auto retract_QR(tangent_vector &xi) -> manifold_ptr;

  auto WeingartenMap(const arma::mat &xi_normal, const tangent_vector &Z_tv)
      -> tangent_vector;
};

#endif
