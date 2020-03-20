#include "psdCone.hpp"

psdCone::psdCone(unsigned n1, unsigned p1, unsigned r1) : manifold(n1, p1, r1) {
  Y = arma::mat(n, n, arma::fill::eye);
  
  tangent_names.clear();
  tangent_names.push_back("K_psdc");
  tangent_size = 1;
  
  has_Ysqroot = false;
}


double psdCone::metric(tangent_vector &xi, tangent_vector &eta) {
  // xi.ownership_check(id_code);
  // eta.ownership_check(id_code);
  arma::mat m1 =  xi("K_psdc");
  arma::mat m2 = eta("K_Psdc");
  //m1 = solve(Y, m1);
  //m1 = solve(Y, m1.t());
  //m2 = solve(Y, m2);
  //m2 = solve(Y, m2.t());
  return arma::dot(m1, m2);
}

auto psdCone::evalGradient(vecMatIter amb_grad) -> tangent_vector {
  tangent_vector grad;
  grad = create_tangentV_at_current_point();
  grad("K_psdc") = 0.5 * (amb_grad[0] + amb_grad[0].t());
  if (!has_Ysqroot)
    compute_Ysqroot();
  grad("K_psdc") = Ysqroot * grad("K_psdc") * Ysqroot;
  return grad;
}

arma::mat psdCone::tangent2mat(tangent_vector &xi) {
  arma::mat result;
  if (!has_Ysqroot)
    compute_Ysqroot();
  result = Ysqroot * xi("K_psdc") * Ysqroot;
  
  return result;
}

auto psdCone::evalHessian(vecMatIter amb_grad, 
                          vecMatIter amb_hess,
                          tangent_vector &Z_tv) 
  -> tangent_vector {
    tangent_vector tv;
    // empty
    // not implemented yet
    return tv;
  }

auto psdCone::retraction(tangent_vector &xi) -> manifold_ptr {
  shared_ptr<psdCone> result = make_shared<psdCone>(n, p, r);
  if (!has_Ysqroot)
    compute_Ysqroot();
  result->Y = Ysqroot * matExp(xi("K_psdc")) * Ysqroot;
  return result;
}



auto psdCone::vectorTrans(tangent_vector &object, 
                          tangent_vector &forwardDirection,
                          manifold_ptr forwardY) 
  -> tangent_vector {
    
    tangent_vector grad;
    grad = create_tangentV_at_current_point();
    if (!has_Ysqroot)
      compute_Ysqroot();
    
    arma::mat tmp = forwardDirection("K_psdc") * 0.5;
    tmp =  matExp(tmp);
    grad("K_psdc") = tmp * object("K_psdc") * tmp.t();
    return grad;
    
  };


mat psdCone::matExp(const mat & Z){
  vec eigval;
  mat eigvec;
  eig_sym( eigval, eigvec, Z);
  //eigval.elem( find(eigval < 0) ).zeros();
  eigval = exp(eigval/2);
  // if(eigval.max() / eigval.min() > 1e8 )
  //     throw(runtime_error("Conditional Number Too Large!"));
  //Rcpp::Rcout << eigval << std::endl;
  eigvec.each_row() %= eigval.t();
  mat result = eigvec * eigvec.t();
  return result;
}

void psdCone::compute_Ysqroot(){
  vec eigval;
  mat eigvec;
  eig_sym( eigval, eigvec, Y);
  eigval.elem( find(eigval < 0) ).zeros();
  eigval = sqrt(sqrt(eigval));
  eigvec.each_row() %= eigval.t();
  Ysqroot = eigvec *  eigvec.t();
  // Ysqroot = sqrtmat_sympd(Y);
  has_Ysqroot = true;
}