#ifndef MANIFOLD_BASE_INCLUDE_HEADER
#define MANIFOLD_BASE_INCLUDE_HEADER

#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace Rcpp;
using namespace std;
using namespace arma;

class manifold;

typedef shared_ptr<manifold> manifold_ptr;
typedef Rcpp::Nullable<Rcpp::Function> nullable_type;
typedef vector<arma::mat>::size_type size_type;

// inline void std_exception_handling(std::exception& e){
//   Rcpp::stop(e.what());
// }
//
// inline void status_clear() {
//   ofstream status_output;
//   status_output.open("/Users/shiyuanhe/optstatus.txt", ios::out);
//   status_output.close();
// }
//
// inline void status_print(string s) {
//   ofstream status_output;
//   status_output.open("/Users/shiyuanhe/optstatus.txt", ios::app);
//   status_output << s << std::endl;
//   status_output.close();
// }

#endif
