#include "optfunction.hpp"

FunctionClass::FunctionClass() {
  objective_f = R_NilValue;
  gradient_f = R_NilValue;
  hessian_f = R_NilValue;
}

double FunctionClass::objective_at(const manifold_ptr mPoint) {
  return ambient_objective(mPoint);
}

auto FunctionClass::gradient_at(const manifold_ptr mPoint) -> tangent_vector {
  vecMat abtGrad = ambient_gradient(mPoint);
  tangent_vector result;
  result = mPoint->evalGradient(abtGrad.begin());
  return result;
}

auto FunctionClass::hessian_at(const manifold_ptr mPoint, tangent_vector &Z)
    -> tangent_vector {
  vecMat amb_hess = ambient_hessian(mPoint, Z);
  vecMat amb_grad = ambient_gradient(mPoint);
  tangent_vector result;
  result = mPoint->evalHessian(amb_grad.begin(), amb_hess.begin(), Z);
  return result;
}

double FunctionClass::ambient_objective(const manifold_ptr mPoint) {
  double tmpVal;
  unsigned nc = mPoint->get_comp_num();
  Function obj_fun = as<Function>(objective_f);

  if (nc == 1) {
    arma::mat Y4R = mPoint->self2mat();
    tmpVal = as<double>(obj_fun(Y4R));
  } else {
    vecMat vecMatForm = mPoint->self2vecMat();
    List Y4R = vecMat2List(vecMatForm);
    tmpVal = as<double>(obj_fun(Y4R));
  }

  return tmpVal;
}

vecMat FunctionClass::ambient_gradient(const manifold_ptr mPoint) {
  vecMat grad_result;
  SEXP grad_sexp;
  Function grad_fun = as<Function>(gradient_f);
  unsigned nc = mPoint->get_comp_num();

  if (nc == 1) {
    arma::mat Y4R = mPoint->self2mat();
    grad_sexp = grad_fun(Y4R);
  } else {
    vecMat vecMatForm = mPoint->self2vecMat();
    List Y4R = vecMat2List(vecMatForm);
    grad_sexp = grad_fun(Y4R);
  }

  grad_result = SEXP_to_vecMat(grad_sexp, nc);
  return grad_result;
}

vecMat FunctionClass::ambient_hessian(const manifold_ptr mPoint,
                                       tangent_vector &Z) {
  // SEXP mPoint4R = mPoint_to_SEXP(mPoint);
  unsigned nc = mPoint->get_comp_num();
  Function hess_fun = as<Function>(hessian_f);
  SEXP hess_sexp;

  if (nc == 1) {
    arma::mat Y4R = mPoint->self2mat();
    arma::mat Z4R = mPoint->tangent2mat(Z);
    hess_sexp = hess_fun(Y4R, Z4R);
  } else {
    vecMat vecMatForm = mPoint->self2vecMat();
    vecMat tv_vecMat = mPoint->tangent2vecMat(Z);
    List Y4R = vecMat2List(vecMatForm);
    List Z4R = vecMat2List(tv_vecMat);
    hess_sexp = hess_fun(Y4R, Z4R);
  }
  vecMat hess_result = SEXP_to_vecMat(hess_sexp, nc);
  return hess_result;
}

// SEXP FunctionClass::mPoint_to_SEXP(const manifold_ptr mPoint) {
//   SEXP result;
//   vecMat vecMatForm = mPoint->self2vecMat();
//   unsigned cn = mPoint->get_comp_num();
//   result = vecMat_to_SEXP(vecMatForm, cn);
//   return result;
// }

// SEXP FunctionClass::vecMat_to_SEXP(const vecMat &vmObject, unsigned nc) {
//   SEXP result;
//   if (nc == 1) {
//     result = wrap(vmObject[0]);
//   } else if (nc > 1) {
//     Rcpp::List res_ = Rcpp::List();
//     for (unsigned i = 0; i < nc; i++)
//       res_.push_back(vmObject[i]);
//     result = wrap(res_);
//   } else {
//     throw runtime_error("Incorrect component number!");
//   }
//   return result;
// }
