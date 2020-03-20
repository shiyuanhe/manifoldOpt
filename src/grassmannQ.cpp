#include "grassmannQ.hpp"

// void grassmannQ::evalGradient(arma::mat gradF, std::string method){
//   xi=gradF-Y*Y.t()*gradF;
//   eDescent=arma::dot(gradF,xi);
//   if(method=="steepest"){
//     descD=-xi;
//   }else if(method=="particleSwarm"){
//     descD=xi;
//   }else if(method=="trustRegion"){
//
//   }
// }

//! Evaluate gradient
/* !
Compute gradient in the tangent space from the gradient
in the ambient space. amb_grad.
(I-YY^T)*amb_grad
*/
auto grassmannQ::evalGradient(vecMatIter amb_grad) -> tangent_vector {
  tangent_vector grad;
  grad = create_tangentV_at_current_point();
  grad("xi") = *amb_grad - Y * (Y.t() * (*amb_grad));
  return grad;
}

//! Evaluate retraction
/*!
Either retract by QR decomposition or geodesics.
Retract in the direction of the tangent vector, xi;
Return a point on the manifold (as a pointer).
*/
auto grassmannQ::retraction(tangent_vector &xi) -> manifold_ptr {
  xi.ownership_check(id_code);

  manifold_ptr result;
  result = retract_QR(xi);

  return result;
}

auto grassmannQ::retract_QR(const tangent_vector &xi) -> manifold_ptr {
  arma::mat retract_U, retract_V;
  shared_ptr<grassmannQ> result = make_shared<grassmannQ>(n, p, r);

  arma::mat Yt = Y + xi(0);
  arma::qr_econ(retract_U, retract_V, Yt);
  if (retract_V(0, 0) < 0) {
    retract_U = -retract_U;
  }
  result->Y = retract_U;

  return result;
}

auto grassmannQ::retract_Exp(const tangent_vector &xi) -> manifold_ptr {
  mat retract_U, retract_V, Yt;
  vec retract_Sigma;
  shared_ptr<grassmannQ> result = make_shared<grassmannQ>(n, p, r);

  arma::mat U_svd;
  arma::svd_econ(U_svd, retract_Sigma, retract_V, xi(0));
  retract_U = arma::mat(n, 2 * p, arma::fill::zeros);
  retract_U(span::all, span(0, p - 1)) = Y * retract_V;
  retract_U(span::all, span(p, 2 * p - 1)) = U_svd;

  arma::mat retract_middle;
  retract_middle = mat(2 * p, p, arma::fill::zeros);
  retract_middle(span(0, p - 1), span::all) = diagmat(cos(retract_Sigma));
  retract_middle(span(p, 2 * p - 1), span::all) = diagmat(sin(retract_Sigma));
  Yt = retract_U * retract_middle * retract_V.t();
  result->Y = Yt;

  return result;
}

grassmannQ::grassmannQ(unsigned n1, unsigned p1, unsigned r1)
    : manifold(n1, p1, r1) {
  Y = mat(n1, p1, fill::eye);
}

void grassmannQ::initial_point(vecMatIter rObj) {
  arma::mat retract_U, retract_V;

  arma::qr_econ(retract_U, retract_V, *rObj);
  Y = retract_U;
}

/*!
Hessian operator on Z_tv. Z_tv is already a tangent vector.
The amb_grad is supplementary information, which is necessary
for some manifolds. The amb_hess is computed from
D(Df(Y))[Z_tv].
*/
auto grassmannQ::evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                             tangent_vector &Z_tv) -> tangent_vector {
  Z_tv.ownership_check(id_code);
  // project euclidian Hessian onto tangent space
  arma::mat Z = Z_tv(0);

  tangent_vector hessZ = evalGradient(amb_hess);
  hessZ(0) = hessZ(0) - Z * (Y.t() * (*amb_grad));
  return hessZ;
}

//
// void grassmannQ::set_particle(){
//   arma::mat y_temp=arma::randn(n,p);
//   arma::mat Q,R;
//   arma::qr_econ(Q,R,y_temp);
//   Y=Q;
//   arma::mat velocity_temp=arma::randn(n,p);
//   //psedo gradient as velocity;
//   evalGradient(velocity_temp,"particleSwarm");
// }
//
