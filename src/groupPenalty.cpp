#include "penaltyClass.hpp"

//Group Lasso and Group SCAD penalty
//row-wise


double FunPlusPenaltyClass::grouplasso_obj(const arma::mat & X)
{
    double result = 0;
    double rNorm = 0;
    for(unsigned i=0; i< X.n_rows; i++){
        rNorm = rowl2norm(X(i,span::all));
        result += lambda(i) * rNorm;
    }
    return result;
}


arma::mat FunPlusPenaltyClass::grouplasso_grad(const arma::mat & X){
    unsigned n = X.n_rows, 
        p = X.n_cols;
    arma::mat glGrad(n,p,arma::fill::zeros);
    
    for(unsigned i=0; i< n; i++)
        glGrad(i, span::all) = 
            grouplasso_grad_onerow(X(i,span::all), lambda(i));
    return glGrad;
}


arma::mat FunPlusPenaltyClass::grouplasso_hess(const arma::mat & X, 
                                               const arma::mat & Z){
    unsigned n = X.n_rows, 
        p = X.n_cols;
    arma::mat glHess(n,p,arma::fill::zeros);
    
    for(unsigned i=0; i< n; i++)
        glHess(i, span::all) = 
            grouplasso_hess_onerow(X(i,span::all), 
                                   Z(i,span::all), 
                                   lambda(i));
    return glHess;
}




arma::mat FunPlusPenaltyClass::grouplasso_thresh(const arma::mat & X){
    unsigned n = X.n_rows, 
        p = X.n_cols;
    arma::mat Xthresh(n,p, arma::fill::zeros);
    double rNorm, shrink;
    for(unsigned i=0; i<n; i++){
        rNorm = rowl2norm(X(i,span::all));        
        shrink = lasso_shrinkValue(rNorm, lambda(i)*ladjust);
        Xthresh(i,span::all) = shrink * X(i,span::all);
    }
    return Xthresh;
}


double FunPlusPenaltyClass::groupSCAD_obj(const arma::mat & X){
    double result = 0;
    double rNorm = 0;
    for(unsigned i=0; i< X.n_rows; i++){
        rNorm = rowl2norm(X(i,span::all));
        result += single_SCAD_penalty(rNorm, lambda(i));
    }
    return result;
}


double FunPlusPenaltyClass::
    single_SCAD_penalty(const double elem_norm,
                        const double lambda_c){
        double tmp;
        if(elem_norm < lambda_c)
            tmp = lambda_c * elem_norm;
        else if(elem_norm < alpha*lambda_c){
            tmp = elem_norm*elem_norm- 
                2* alpha* lambda_c *elem_norm + lambda_c*lambda_c;
            tmp = -tmp/2.0/(alpha-1.0);
        }else{
            tmp = (alpha+1.0) * lambda_c* lambda_c/2.0;
        }
        return tmp;
    }


arma::mat FunPlusPenaltyClass::groupSCAD_grad(const arma::mat & X){
    unsigned n = X.n_rows, 
        p = X.n_cols;
    arma::mat scadGrad(n,p,arma::fill::zeros);
    double rNorm = 0;
    arma::rowvec tmp;
    
    for(unsigned i=0; i< n; i++){
        rNorm = rowl2norm(X(i,span::all));
        tmp =  grouplasso_grad_onerow(X(i,span::all), lambda(i));
        if(rNorm <lambda(i)){
            
        }else if(rNorm <lambda(i)*alpha){
            tmp = tmp * alpha/(alpha - 1);
            tmp += -X(i,span::all)/(1-alpha); 
        }else{
            tmp *= 0;
        }
        scadGrad(i, span::all) = tmp;
    }
    return scadGrad;
}



arma::mat FunPlusPenaltyClass::groupSCAD_hess(const arma::mat & X,
                                              const arma::mat & Z){
    unsigned n = X.n_rows, 
        p = X.n_cols;
    arma::mat scadHess(n,p,arma::fill::zeros);
    double rNorm = 0;
    arma::rowvec tmp;
    
    for(unsigned i=0; i< n; i++){
        rNorm = rowl2norm(X(i,span::all));
        tmp =  grouplasso_hess_onerow(X(i,span::all), 
                                      Z(i,span::all), 
                                      lambda(i));
        if(rNorm <lambda(i)){
            
        }else if(rNorm <lambda(i)*alpha){
            tmp = tmp * alpha/(alpha - 1);
            tmp += -Z(i,span::all)/(1-alpha); 
        }else{
            tmp *= 0;
        }
        scadHess(i, span::all) = tmp;
    }
    return scadHess;
}


arma::mat FunPlusPenaltyClass::groupSCAD_thresh(const arma::mat & X){
    unsigned n = X.n_rows, 
        p = X.n_cols;
    arma::mat Xthresh(n,p, arma::fill::zeros);
    double rNorm = 0,  shrinkV;
    
    for(unsigned i=0; i<n; i++){
        rNorm = rowl2norm(X(i,span::all));
        shrinkV = SCAD_shrinkValue(rNorm, lambda(i)*ladjust);
        Xthresh(i,span::all) = shrinkV * X(i, span::all);
    }
    
    return Xthresh;
}






double FunPlusPenaltyClass::rowl2norm(const arma::rowvec & onerow){
    // sum of squared elements in each row
    double rsum = sum(onerow % onerow);
    rsum += epsilon;
    rsum = sqrt(rsum);
    return rsum;
}


arma::rowvec  FunPlusPenaltyClass::
    grouplasso_grad_onerow(const arma::rowvec & onerow, 
                           double lambda_c){
        double rNorm = rowl2norm(onerow);
        arma::rowvec res_onerow;
        res_onerow = lambda_c * onerow / rNorm;
        return res_onerow;
    }



arma::rowvec FunPlusPenaltyClass::
    grouplasso_hess_onerow(const arma::rowvec & onerow_X,
                           const arma::rowvec & onerow_Z,
                           double lambda_c){
        arma::rowvec onerow_res;
        double rNorm = rowl2norm(onerow_X);
        double sZX = sum(onerow_X % onerow_Z);
        onerow_res = lambda_c * onerow_Z / rNorm-
            lambda_c * onerow_X *sZX/pow(rNorm,3.0);
        return onerow_res;
    }

