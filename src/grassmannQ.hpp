#ifndef GRASSMANQ_CLASS
#define GRASSMANQ_CLASS

#include "manifold.hpp"

//! Grassmann manifold
/*!
//The quotient version of grassmann manifold.
G(p,r) as r dimensional subspace of R^p.
*/

class grassmannQ : public manifold {
public:
  grassmannQ(unsigned, unsigned, unsigned);

  void initial_point(vecMatIter amb_grad);

  auto evalGradient(vecMatIter amb_grad) -> tangent_vector;

  auto evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                   tangent_vector &Z_tv) -> tangent_vector;

  auto retraction(tangent_vector &) -> manifold_ptr;
  auto retract_QR(const tangent_vector &) -> manifold_ptr;
  auto retract_Exp(const tangent_vector &) -> manifold_ptr;

  // New added
  // arma::mat genretract(double stepSize, const arma::mat &Z);
  // void vectorTrans();
  // void update_conjugateD(double);
  // void set_particle();
};

#endif
