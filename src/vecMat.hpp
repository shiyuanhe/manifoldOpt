#ifndef MANIFOLD_VECMAT_HEADER
#define MANIFOLD_VECMAT_HEADER

#include <RcppArmadillo.h>
#include <vector>
using namespace Rcpp;
using namespace std;
using namespace arma;

typedef vector<arma::mat> vecMat;
typedef std::vector<arma::mat>::iterator vecMatIter;

inline List vecMat2List(vecMat input) {
  unsigned nc = input.size();
  List result(nc);
  for (unsigned i = 0; i < nc; i++)
    result[i] = input.at(i);
  return result;
}

inline vecMat SEXP_to_vecMat(SEXP sexpObject, unsigned nc) {
  vecMat result;
  arma::mat ires_;
  if (nc == 1) {
    ires_ = as<arma::mat>(sexpObject);
    result.push_back(ires_);
  } else if (nc > 1) {
    Rcpp::List rList = as<List>(sexpObject);
    for (unsigned i = 0; i < nc; i++) {
      ires_ = as<arma::mat>(rList.at(i));
      result.push_back(ires_);
    }
  }
  return result;
}

#endif
