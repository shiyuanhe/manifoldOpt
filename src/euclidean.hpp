#ifndef MANIFOLD_EUCLIDEAN_CLASS
#define MANIFOLD_EUCLIDEAN_CLASS

#include "manifold.hpp"

class euclidean : public manifold {
public:
  euclidean(unsigned n_, unsigned p_, unsigned r_) : manifold(n_, p_, r_) {
    tangent_size = 1;
    Y = arma::mat(n, p, arma::fill::zeros);
  }

  ~euclidean() {}

  auto retraction(tangent_vector &xi) -> manifold_ptr {
    xi.ownership_check(id_code);
    shared_ptr<euclidean> res = make_shared<euclidean>(n, p, r);
    res->Y = Y + xi(0);
    return res;
  }

  auto evalGradient(vecMatIter amb_grad) -> tangent_vector {
    tangent_vector grad;
    grad = create_tangentV_at_current_point();
    grad() = amb_grad[0];
    return grad;
  }

  auto evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                   tangent_vector &Z_tv) -> tangent_vector {
    Z_tv.ownership_check(id_code);
    tangent_vector hessZ;
    hessZ = create_tangentV_at_current_point();
    hessZ() = amb_hess[0];
    return hessZ;
  }
};

#endif
