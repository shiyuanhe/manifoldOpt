#ifndef FIXRANKPSD_CLASS
#define FIXRANKPSD_CLASS

#include "manifold.hpp"

/*!
For a symmetric positive semidefinite
X = YY^T in R^{n\times n}
Y in R^{n\times r}
User supplied funcion is w.r.t X
M. JOURNEE, F. BACH et al. (2010)
LOW-RANK OPTIMIZATION ON THE CONE OF POSITIVE SEMIDEFINITE MATRICES
*/

class fixRankPSD : public manifold {
public:
  fixRankPSD(unsigned, unsigned, unsigned);
  auto evalGradient(vecMatIter amb_grad) -> tangent_vector;
  auto evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                   tangent_vector &Z_tv) -> tangent_vector;

  auto retraction(tangent_vector &) -> manifold_ptr;
  arma::mat self2mat();
  vecMat self2vecMat();
  arma::mat tangent2mat(tangent_vector &xi);
  vecMat tangent2vecMat(tangent_vector &xi);
  //  virtual void set_descD(arma::mat );
  // void vectorTrans();
  // void update_conjugateD(double);
  // void set_particle();
  //  virtual void acceptY();
};

#endif
