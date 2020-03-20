#ifndef GROUPLASSO_CLASS
#define GROUPLASSO_CLASS

#include "optfunction.hpp"

class FunPlusPenaltyClass: 
    public FunctionClass{
public:

    enum PenaltyType { groupLASSO, groupSCAD, ALMLoss };

    
    FunPlusPenaltyClass(){
        penaltyObj_ptr = &FunPlusPenaltyClass::grouplasso_obj;
        penaltyGrad_ptr = &FunPlusPenaltyClass::grouplasso_grad;
        penaltyHess_ptr = &FunPlusPenaltyClass::grouplasso_hess;
        epsilon = 0.0;
        alpha = 2.7;
        rho = 1e4;
        ladjust = 1;
    }    

    void set_lambda(arma::vec lambda_){
        lambda = lambda_;
    };
    
    
    void set_ALM_params(arma::mat G_, double rho_){
        ALM_G = G_;
        rho = rho_;
    }
    
    void set_epsilon(double epsilon_){
        epsilon = epsilon_;
    }
    

    ~FunPlusPenaltyClass(){}

    void penalty_check(const manifold_ptr mPoint);
    
    
    void set_penalty(PenaltyType s);
    arma::mat thresh_matrix(const arma::mat & inputmat,
                            double lambda_adjust_){
        ladjust = lambda_adjust_;
        return (this->*penaltyThresh_ptr)(inputmat);
    }
    
private:
    //ladjust: lambda*ladjust 
    // tuning parameter adjustment in thresholding 
    double epsilon, rho, alpha, ladjust;
    arma::vec lambda;
    arma::mat ALM_G;
    
    // function pointer to select and execute specific penalty choice
    double (FunPlusPenaltyClass::*penaltyObj_ptr)(const arma::mat & );
    arma::mat (FunPlusPenaltyClass::*penaltyGrad_ptr)(const arma::mat & );
    arma::mat (FunPlusPenaltyClass::*penaltyHess_ptr)(const arma::mat &,
               const arma::mat & );
    arma::mat (FunPlusPenaltyClass::*penaltyThresh_ptr)(const arma::mat & );
    
    
    double ambient_objective(const manifold_ptr mPoint);
    vecMat ambient_gradient(const manifold_ptr mPoint);
    vecMat ambient_hessian(const manifold_ptr mPoint, 
                            tangent_vector & Z);
    
    
    //Group Lasso
    double grouplasso_obj(const arma::mat & X);
    arma::mat grouplasso_grad(const arma::mat & X);
    arma::mat grouplasso_hess(const arma::mat & X, const arma::mat & Z);
    arma::mat grouplasso_thresh(const arma::mat & );
    //Lower level function of group lasso
    double rowl2norm(const arma::rowvec & );
    arma::rowvec  grouplasso_grad_onerow(const arma::rowvec &, double);
    arma::rowvec grouplasso_hess_onerow(const arma::rowvec & onerow_X,
                                        const arma::rowvec & onerow_Z, 
                                        double lambda_c);
    arma::rowvec grouplasso_thresh_onerow(const arma::rowvec &, double);

        
    // SCAD
    double groupSCAD_obj(const arma::mat & X);
    arma::mat groupSCAD_grad(const arma::mat & X);
    arma::mat groupSCAD_hess(const arma::mat & X, const arma::mat & Z);
    arma::mat groupSCAD_thresh(const arma::mat & );
    

    inline double lasso_shrinkValue(const double rNorm, const double lambda_c){
        double shrink;
        if(rNorm>0){
            shrink = 1 - lambda_c/rNorm;
        }else{
            shrink = 0;
        }
        if(shrink<0) shrink = 0;
        return shrink;
    }
    
    
    inline double SCAD_shrinkValue(const double rNorm, double lambda_c){
        double shrink;
        if(rNorm < 2*lambda_c){
            shrink = lasso_shrinkValue(rNorm, lambda_c);
        }else if(rNorm < alpha*lambda_c){
            lambda_c = lambda_c *alpha/(alpha-1);
            shrink = lasso_shrinkValue(rNorm, lambda_c);
            shrink *= (alpha-1.0)/(alpha-2.0);
        }else{
            shrink = 1;
        }
        return shrink;
    }
    
    double single_SCAD_penalty(const double elem_norm,
                            const double lambda_c);
    
    //Lasso
    // double lasso_obj(const arma::mat & X);
    // arma::mat lasso_grad(const arma::mat & X);
    // arma::mat lasso_hess(const arma::mat & X, const arma::mat & Z);
    // arma::mat lasso_thresh(const arma::mat & );
    // //Lower level function of group lasso
    // double single_elem_norm(double );
    // arma::rowvec signle_lasso_thresh(double );
    
    
    
    // For the first part of augmented lagrange method
    double ALMpartI_obj(const arma::mat & X);
    arma::mat ALMpartI_grad(const arma::mat & X);
    arma::mat ALMpartI_hess(const arma::mat & X, const arma::mat & Z);
    arma::mat ALMpartI_thresh(const arma::mat & X);
};


#endif