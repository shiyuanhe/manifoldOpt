#ifndef PSDCONE_CLASS
#define PSDCONE_CLASS

#include "manifold.hpp"

/*!
Smith (2006). Covariance, subspace and intrinsic cramer-rao bound.
*/

class psdCone : public manifold {
public:
  psdCone(unsigned, unsigned, unsigned);
  auto evalGradient(vecMatIter amb_grad) -> tangent_vector;
  auto evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                   tangent_vector &Z_tv) -> tangent_vector;

  auto retraction(tangent_vector &) -> manifold_ptr;

  const arma::mat &get_Wroot() {
    if (!has_Ysqroot)
      compute_Ysqroot();
    return Ysqroot;
  }
  
  auto vectorTrans(tangent_vector& object, 
                   tangent_vector& forwardDirection,
                   manifold_ptr forwardY) -> tangent_vector;
  
  
  double metric(tangent_vector &xi, tangent_vector &eta);
  
  // arma::mat self2mat();
  // vecMat self2vecMat();
  arma::mat tangent2mat(tangent_vector &xi);
  // vecMat tangent2vecMat(const tangent_vector &xi);
  //  virtual void set_descD(arma::mat );
  // void vectorTrans();
  // void update_conjugateD(double);
  // void set_particle();
  //  virtual void acceptY();
  bool has_Ysqroot;

private:
  mat Ysqroot;

  void compute_Ysqroot();
  mat matExp(const mat & Z);
};

#endif
