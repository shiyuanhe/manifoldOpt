#include "penaltyClass.hpp"

void FunPlusPenaltyClass::penalty_check(const manifold_ptr mPoint){
    vecMat tmp = mPoint->self2vecMat();
    unsigned nrow = tmp[0].n_rows;
    if(lambda.size()==nrow){
        
    }else if(lambda.size()==1){
        lambda = arma::vec(nrow,arma::fill::ones)*lambda(0);
    }else{
        throw runtime_error("Incorrect length of vector lambda!");
    }
}


void FunPlusPenaltyClass::set_penalty(PenaltyType s){
    switch(s){
    case groupLASSO:
        penaltyObj_ptr = &FunPlusPenaltyClass::grouplasso_obj;
        penaltyGrad_ptr = &FunPlusPenaltyClass::grouplasso_grad;
        penaltyHess_ptr = &FunPlusPenaltyClass::grouplasso_hess;
        penaltyThresh_ptr = &FunPlusPenaltyClass::grouplasso_thresh;
        break;
    case groupSCAD:
        penaltyObj_ptr = &FunPlusPenaltyClass::groupSCAD_obj;
        penaltyGrad_ptr = &FunPlusPenaltyClass::groupSCAD_grad;
        penaltyHess_ptr = &FunPlusPenaltyClass::groupSCAD_hess;
        penaltyThresh_ptr = &FunPlusPenaltyClass::groupSCAD_thresh;
        break;
    case ALMLoss:
        penaltyObj_ptr = &FunPlusPenaltyClass::ALMpartI_obj;
        penaltyGrad_ptr = &FunPlusPenaltyClass::ALMpartI_grad;
        penaltyHess_ptr = &FunPlusPenaltyClass::ALMpartI_hess;
        penaltyThresh_ptr = &FunPlusPenaltyClass::ALMpartI_thresh;
        break;
    }
}





double FunPlusPenaltyClass::ambient_objective(const manifold_ptr mPoint) { 
    double objValue = FunctionClass::ambient_objective(mPoint);
    vecMat mPoint_vmForm = mPoint->self2vecMat();
    objValue += (this->*penaltyObj_ptr)(mPoint_vmForm[0]);
    return objValue;
}


vecMat FunPlusPenaltyClass::ambient_gradient(const manifold_ptr mPoint) { 
    vecMat grad_result = FunctionClass::ambient_gradient(mPoint);
    
    vecMat mPoint_vmForm = mPoint->self2vecMat();
    grad_result[0] += (this->*penaltyGrad_ptr)(mPoint_vmForm[0]);
    
    return grad_result;
}


vecMat FunPlusPenaltyClass::ambient_hessian(const manifold_ptr mPoint,
                                             tangent_vector& Z_tv){
    vecMat hess_result = FunctionClass::ambient_hessian(mPoint, Z_tv);
    vecMat mPoint_vmForm = mPoint->self2vecMat();
    vecMat grad_vmForm = mPoint->tangent2vecMat(Z_tv);
    hess_result[0] += (this->*penaltyHess_ptr)(mPoint_vmForm[0], grad_vmForm[0]);
    return hess_result;
}

