#ifndef FIXRANK_CLASS
#define FIXRANK_CLASS

#include "manifold.hpp"

// quotient version of grassmann manifold

class fixRank : public manifold {
private:
  arma::mat U, V;
  arma::vec Sigma;

  arma::mat tv_as_mat(const tangent_vector &);
  arma::mat evalNormal(vecMatIter amb_grad);
  auto WeingartenMap(const arma::mat &xi_normal, const tangent_vector &Z_tv)
      -> tangent_vector;

public:
  fixRank(unsigned n_, unsigned p_, unsigned r_);

  void initial_point(vecMatIter rObj);

  auto evalGradient(vecMatIter amb_grad) -> tangent_vector;

  auto evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                   tangent_vector &Z_tv) -> tangent_vector;

  auto retraction(tangent_vector &) -> manifold_ptr;
  arma::mat tangent2mat(tangent_vector &xi);
  vecMat tangent2vecMat(tangent_vector &tv);
  double metric(tangent_vector &, tangent_vector &);

  // void vectorTrans();
  // void update_conjugateD(double);
  // void set_particle();
};

#endif
