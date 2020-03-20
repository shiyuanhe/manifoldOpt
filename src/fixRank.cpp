
#include "fixRank.hpp"

fixRank::fixRank(unsigned n_, unsigned p_, unsigned r_) : manifold(n_, p_, r_) {

  U = mat(n, r, arma::fill::eye);
  V = mat(p, r, arma::fill::eye);
  Sigma = arma::vec(r, arma::fill::ones);
  Y = U * arma::diagmat(Sigma) * V.t();

  tangent_names.clear();
  tangent_names.push_back("Up");
  tangent_names.push_back("Vp");
  tangent_names.push_back("M");
  tangent_size = 3;
}

void fixRank::initial_point(vecMatIter init_mat) {
  Y = *init_mat;
  arma::svd_econ(U, Sigma, V, Y);
  U = U(span::all, span(0, r - 1));
  V = V(span::all, span(0, r - 1));
  Sigma = Sigma(span(0, r - 1));
}

auto fixRank::evalGradient(vecMatIter amb_grad) -> tangent_vector {
  arma::mat Ru, Rv;

  tangent_vector grad = create_tangentV_at_current_point();
  Ru = (*amb_grad).t() * U;
  Rv = (*amb_grad) * V;
  grad("M") = U.t() * Rv;
  grad("Up") = Rv - U * grad("M");
  grad("Vp") = Ru - V * grad("M").t();
  return grad;
}

arma::mat fixRank::tangent2mat(tangent_vector &xi) { return tv_as_mat(xi); }

vecMat fixRank::tangent2vecMat(tangent_vector &tv) {
  vecMat res;
  res.push_back(tv_as_mat(tv));
  return res;
}

arma::mat fixRank::tv_as_mat(const tangent_vector &tv) {
  tv.ownership_check(id_code);
  arma::mat xi;
  xi = U * tv("M") * V.t() + tv("Up") * V.t() + U * tv("Vp").t();
  return xi;
}

auto fixRank::evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                          tangent_vector &Z_tv) -> tangent_vector {
  Z_tv.ownership_check(id_code);
  // project euclidian Hessian onto tangent space
  tangent_vector hessZ = evalGradient(amb_hess);
  arma::mat xi_normal = evalNormal(amb_grad);
  hessZ = hessZ + WeingartenMap(xi_normal, Z_tv);
  return hessZ;
}

auto fixRank::WeingartenMap(const arma::mat &xi_normal,
                            const tangent_vector &Z_tv) -> tangent_vector {
  // psuedo inverse
  arma::vec Sigma_inv = Sigma;
  for (unsigned i = 0; i < r; i++)
    Sigma_inv[i] = 1 / Sigma_inv[i];
  arma::mat Y_plus = V * diagmat(Sigma_inv) * U.t();

  arma::mat Z = tv_as_mat(Z_tv);
  arma::mat Weingarten;
  Weingarten = xi_normal * Z.t() * Y_plus.t() + Y_plus.t() * Z.t() * xi_normal;

  tangent_vector result = create_tangentV_at_current_point();
  result("Up") = Weingarten * V;
  result("Vp") = Weingarten.t() * U;
  result("M") = arma::mat(r, r, arma::fill::zeros);
  return result;
}

// vector orthogonal to the tangent space
arma::mat fixRank::evalNormal(vecMatIter amb_grad) {
  arma::mat xi_normal, UUtG, GVVt;
  UUtG = U * U.t() * (*amb_grad);
  GVVt = (*amb_grad) * V * V.t();
  xi_normal = (*amb_grad) - UUtG - GVVt + U * U.t() * GVVt;
  return xi_normal;
}

auto fixRank::retraction(tangent_vector &xi) -> manifold_ptr {
  xi.ownership_check(id_code);
  bool conv;
  arma::mat Qu, Ru, Qv, Rv;
  conv = arma::qr_econ(Qu, Ru, xi("Up"));
  if (!conv)
    throw runtime_error("QR failed in retraction!");
  conv = arma::qr_econ(Qv, Rv, xi("Vp"));
  if (!conv)
    throw runtime_error("QR failed in retraction!");

  arma::mat S, Us, Vs;
  arma::vec Sigma_s;
  S = arma::mat(2 * r, 2 * r, arma::fill::zeros);
  S(span(0, r - 1), span(0, r - 1)) = arma::diagmat(Sigma) + xi("M");
  S(span(r, 2 * r - 1), span(0, r - 1)) = Ru;
  S(span(0, r - 1), span(r, 2 * r - 1)) = Rv.t();
  conv = arma::svd_econ(Us, Sigma_s, Vs, S);
  if (!conv)
    throw runtime_error("SVD failed in retraction!");
  // if(Sigma_s(r-1)<1e-2) throw runtime_error("Unstable Rank after
  // Retraction!");

  shared_ptr<fixRank> result = make_shared<fixRank>(n, p, r);
  result->Sigma = Sigma_s(span(0, r - 1));
  result->U = U * Us(span(0, r - 1), span(0, r - 1)) +
              Qu * Us(span(r, 2 * r - 1), span(0, r - 1));
  result->V = V * Vs(span(0, r - 1), span(0, r - 1)) +
              Qv * Vs(span(r, 2 * r - 1), span(0, r - 1));
  result->Y = (result->U) * arma::diagmat(result->Sigma) * (result->V).t();

  return result;
}

// xi = U*M1*V^T + Up1 * V^T + U* Vp1^T
// eta = U*M2*V^T + Up2 * V^T + U* Vp2^T
// <xi,eta> = <M1,M2> + <Up1, Up2> + <Vp1, Vp2>
double fixRank::metric(tangent_vector &xi, tangent_vector &eta) {
  xi.ownership_check(id_code);
  eta.ownership_check(id_code);
  double result;
  result = arma::dot(xi("M"), eta("M"));
  result += arma::dot(xi("Up"), eta("Up"));
  result += arma::dot(xi("Vp"), eta("Vp"));

  return result;
}

//
// //vector transport of conjugate direction(update descent direction)
// //from Y to Yt, evaluated before accept Y
// void fixRank::vectorTrans(){
//     arma::mat Av,Au,Bv,Bu;
//     Av=V.t()*Vt;
//     Au=U.t()*Ut;
//     Bv=Vp_desc.t()*Vt;
//     Bu=Up_desc.t()*Ut;
//     Up_desc=U*M_desc*Av+Up_desc*Av+U*Bv;
//     Vp_desc=V*M_desc.t()*Au+V*Bu+Vp_desc*Au;
//     M_desc=Au.t()*M_desc*Av+Bu.t()*Av+Au.t()*Bv;
//     Up_desc=Up_desc-Ut*Ut.t()*Up_desc;
//     Vp_desc=Vp_desc-Vt*Vt.t()*Vp_desc;
// }
//
//
// void fixRank::update_conjugateD(double eta){
//     M_desc=M_desc*eta-M;
//     Up_desc=Up_desc*eta-Up;
//     Vp_desc=Vp_desc*eta-Vp;
// }
//
// void fixRank::set_particle(){
//     U=arma::randn(n,r);
//     V=arma::randn(p,r);
//     arma::mat Q,R;
//     arma::qr_econ(Q,R,U);
//     U=Q;
//     arma::qr_econ(Q,R,V);
//     V=Q;
//     Sigma=arma::vec(r,arma::fill::ones);
//     Y=U*arma::diagmat(Sigma)*V.t();
//     //////////////////
//     arma::mat velocity_temp=arma::randn(n,p);
//     // unresolved
//     //evalGradient(velocity_temp,"steepest");
// }
