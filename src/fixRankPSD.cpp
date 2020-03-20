#include "fixRankPSD.hpp"

fixRankPSD::fixRankPSD(unsigned n1, unsigned p1, unsigned r1)
    : manifold(n1, p1, r1) {
  Y = arma::mat(n, r, arma::fill::eye);
}

arma::mat fixRankPSD::self2mat() { return Y * Y.t(); }

vecMat fixRankPSD::self2vecMat() {
  vecMat res;
  res.push_back(Y * Y.t());
  return res;
}

arma::mat fixRankPSD::tangent2mat(tangent_vector &xi) {
  xi.ownership_check(id_code);
  arma::mat Z = xi(0) * Y.t() + Y * xi(0).t();
  return Z;
}

vecMat fixRankPSD::tangent2vecMat(tangent_vector &xi) {
  // check if xi belong to the tangent space at current Y
  xi.ownership_check(id_code);
  arma::mat Z = xi(0) * Y.t() + Y * xi(0).t();
  vecMat res;
  res.push_back(Z);
  return res;
}

auto fixRankPSD::evalGradient(vecMatIter amb_grad) -> tangent_vector {
  tangent_vector grad;
  arma::mat tmp = *amb_grad;
  tmp = (tmp + tmp.t()) * Y;
  grad = create_tangentV_at_current_point();
  grad("xi") = tmp;
  return grad;
}

// evaluate Hessian operator on Z
// eucH is the ordinary Hessian matrix on Z in the ambient space
// return <Z, Hessian*Z>_Y
auto fixRankPSD::evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                             tangent_vector &Z_tv) -> tangent_vector {
  arma::mat tmp, tmp2;
  tmp = *amb_hess;
  tmp = (tmp + tmp.t()) * Y;
  tmp2 = *amb_grad;
  tmp2 = (tmp2 + tmp2.t()) * Z_tv();
  tmp += tmp2;

  arma::mat YtY = Y.t() * Y;
  arma::mat C = Y.t() * tmp;
  C = C.t() - C;
  arma::mat DOmega = arma::syl(YtY, YtY, C);

  arma::mat hessian_Z = tmp - Y * DOmega;

  tangent_vector hess;
  hess = create_tangentV_at_current_point();
  hess("xi") = hessian_Z;
  return hess;
}

// second argument unused
// rectract with step size
// first used to avoid computation repetition
auto fixRankPSD::retraction(tangent_vector &xi) -> manifold_ptr {
  shared_ptr<fixRankPSD> result = make_shared<fixRankPSD>(n, p, r);
  result->Y = Y + xi(0);
  return result;
}

// vector transport of conjugate direction(update descent direction)
// from Y to Yt, evaluated before accept Y

// void fixRankPSD::vectorTrans() {
//   arma::mat A = Yt.t() * Yt;
//   arma::mat C = Yt.t() * descD;
//   C = C.t() - C;
//   // solve the equation A*Omega+Omega*A+C=0
//   arma::mat Omega = arma::syl(A, A, C);
//   descD = descD - Yt * Omega;
// }
//
// void fixRankPSD::update_conjugateD(double eta) { descD = eta * descD - xi; }
//
// void fixRankPSD::set_particle() {
//   Y = arma::randn(n, r);
//   arma::mat velocity_temp = arma::randn(n, r);
//   // psuedo gradient as velocity;
//   evalGradient(velocity_temp, "particleSwarm");
// }
