#ifndef EIGENREG_CLASS
#define EIGENREG_CLASS

#include "manifold.hpp"
#include "psdCone.hpp"
#include "stiefel.hpp"

/*
 Symmetric version of
 Mishra et al (2013), Low-rank regression with trace norm penalty
 (V, W): V Stiefel, W Positive Definite
 */
class eigenReg : public manifold {
public:
  eigenReg(unsigned, unsigned, unsigned);
  auto evalGradient(vecMatIter amb_grad) -> tangent_vector;
  auto evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                   tangent_vector &Z_tv) -> tangent_vector;
  
  auto retraction(tangent_vector &) -> manifold_ptr;
  // arma::mat self2mat();
  vecMat self2vecMat();
  vecMat tangent2vecMat(tangent_vector &xi);
  
  double metric(tangent_vector &xi, tangent_vector &eta);
  
  void initial_point(vecMatIter rObj);
  
  
  auto vectorTrans(tangent_vector& object, 
                   tangent_vector& forwardDirection,
                   manifold_ptr forwardY) -> tangent_vector;
  
  // arma::mat tangent2mat(const tangent_vector &xi);
  // vecMat tangent2vecMat(const tangent_vector &xi);
  //  virtual void set_descD(arma::mat );
  // void vectorTrans();
  // void update_conjugateD(double);
  // void set_particle();
  //  virtual void acceptY();
  
  shared_ptr<stiefel> Vptr;
  shared_ptr<psdCone> Wptr;
};

#endif
