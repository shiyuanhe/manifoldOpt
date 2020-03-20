#include "penaltyClass.hpp"

double FunPlusPenaltyClass::ALMpartI_obj(const arma::mat & X){
    double result = 0;
    result = arma::norm(X-ALM_G,"fro");
    result = result*result*rho*0.5;
    return result;
}

arma::mat FunPlusPenaltyClass::ALMpartI_grad(const arma::mat & X){
    arma::mat res_grad;
    res_grad = rho*(X-ALM_G);
    return res_grad;
}

arma::mat FunPlusPenaltyClass::ALMpartI_hess(const arma::mat & X, const arma::mat & Z){
    arma::mat res_hessZ;
    res_hessZ = rho*Z;
    return res_hessZ;
}


arma::mat FunPlusPenaltyClass::ALMpartI_thresh(const arma::mat & X){
    return X;
}
