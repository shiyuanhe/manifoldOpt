#include "eigenRegUnsym.hpp"


eigenRegUnsym::eigenRegUnsym(unsigned n1, unsigned p1, unsigned r1)
    : manifold(n1, p1, r1) {
    // R^{n1\tiems p1} with n1=p1, matrix rank r1
    Uptr = make_shared<stiefel>(n1, r1, r1);
    Wptr = make_shared<psdCone>(r1, r1, r1);
    Vptr = make_shared<stiefel>(p1, r1, r1);
    
    tangent_names.clear();
    tangent_names.push_back("Unkown~");
    tangent_size = 1;
    comp_num = 3;
}

void eigenRegUnsym::initial_point(vecMatIter rObj) {
    Uptr->initial_point(rObj);
    Wptr->initial_point(rObj + 1);
    Vptr->initial_point(rObj + 2);
    
}

auto eigenRegUnsym::evalGradient(vecMatIter amb_grad) -> tangent_vector {
    tangent_vector grad, grad1, grad2, grad3;
    grad1 = Uptr->evalGradient(amb_grad);     // amb_grad[0]
    grad2 = Wptr->evalGradient(amb_grad + 1); // amb_grad[1]
    grad3 = Vptr->evalGradient(amb_grad + 2); // amb_grad[2]
    
    // Compute Omega
    mat sA1, sA2, sB, sC, sUpdate;
    const mat &sWroot = Wptr->get_Wroot();// Square root?
    sA1 = Uptr->self2mat();
    sA1 = sA1.t() * grad1(0);
    sA1 = sWroot * (sA1 - sA1.t()) * sWroot * 0.5;
    sA2 = Vptr->self2mat();
    sA2 = sA2.t() * grad3(0);
    sA2 = sWroot * (sA2 - sA2.t()) * sWroot * 0.5;
    
    sB = Wptr->self2mat();
    sC = sB * grad2(0); // grad2 = W^{-1/2}*ActualGrad*W^{-1/2}
    sC = (sC - sC.t())*0.5;
    sA1 += sC + sA2;
    sB = sB * sB;
    arma::mat Omega = arma::syl(sB, sB, -sA1);
    
    // update grad1, grad2, combine to grad
    sUpdate = sWroot * Omega * sWroot;
    grad1(0) -= (Uptr->self2mat()) * sUpdate;
    grad3(0) -= (Vptr->self2mat()) * sUpdate;
    
    sUpdate = (Wptr->self2mat()) * Omega;
    grad2(0) -= sUpdate + sUpdate.t();
    grad = grad1 * grad2 * grad3; // gradient tensor product
    grad.ownership_set(id_code);
    return grad;
}

// This function is not implemented
auto eigenRegUnsym::evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                           tangent_vector &Z_tv) -> tangent_vector {
                               tangent_vector tv;
                               // empty
                               
                               return tv;
                           }

auto eigenRegUnsym::retraction(tangent_vector &xi) -> manifold_ptr {
    manifold_ptr tmpPtr;
    shared_ptr<eigenRegUnsym> result = make_shared<eigenRegUnsym>(n, p, r);
    xi.set_currentViewK(0);
    tmpPtr = Uptr->retraction(xi);
    result->Uptr = std::dynamic_pointer_cast<stiefel>(tmpPtr);
    xi.set_currentViewK(1);
    tmpPtr = Wptr->retraction(xi);
    result->Wptr = std::dynamic_pointer_cast<psdCone>(tmpPtr);
    xi.set_currentViewK(2);
    tmpPtr = Vptr->retraction(xi);
    result->Vptr = std::dynamic_pointer_cast<stiefel>(tmpPtr);
    
    xi.set_currentViewK(0);
    
    // double diff1, diff2;
    // diff1 = norm(Vptr->self2mat() - result->Vptr->self2mat(), "fro");
    // diff2 = norm(Wptr->self2mat() - result->Wptr->self2mat(), "fro");
    // Rcout << "diff = " <<diff1<< " " << diff2 << std::endl;
    
    return result;
}

double eigenRegUnsym::metric(tangent_vector &xi, tangent_vector &eta) {
    
    double result = 0, res2;
    xi.set_currentViewK(0);
    eta.set_currentViewK(0);
    result = Uptr->metric(xi, eta);
    xi.set_currentViewK(1);
    eta.set_currentViewK(1);
    res2 = 0.5 * Vptr->metric(xi, eta);
    xi.set_currentViewK(2);
    eta.set_currentViewK(2);
    result += Vptr->metric(xi, eta);
    
    // Rcout << "metric = " <<result<< " " << res2 << std::endl;
    
    result += res2;
    xi.set_currentViewK(0);
    eta.set_currentViewK(0);
    
    return result;
}

vecMat eigenRegUnsym::self2vecMat() {
    vecMat result;
    result.resize(3);
    result.at(0) = Uptr->self2mat();
    result.at(1) = Wptr->self2mat();
    result.at(2) = Vptr->self2mat();
    return result;
}

vecMat eigenRegUnsym::tangent2vecMat(tangent_vector &xi) {
    vecMat result;
    xi.set_currentViewK(0);
    result.push_back(Uptr->tangent2mat(xi));
    xi.set_currentViewK(1);
    result.push_back(Wptr->tangent2mat(xi));
    xi.set_currentViewK(2);
    result.push_back(Vptr->tangent2mat(xi));
    
    return result;
}


// This function is not implemented
auto eigenRegUnsym::vectorTrans(tangent_vector &object, 
                           tangent_vector &forwardDirection,
                           manifold_ptr forwardY) 
    -> tangent_vector {
        
        // vecMat ambM;
        // if (forwardY->get_comp_num() == 1)
        //   ambM.push_back(forwardY->tangent2mat(object));
        // else
        //   ambM = forwardY->tangent2vecMat(object);
        // 
        // tangent_vector eta = forwardY->evalGradient(ambM.begin());
        // return eta;
        
        //
        // vecMat ambM;
        // if (forwardY->get_comp_num() == 1)
        //     ambM.push_back(forwardY->tangent2mat(object));
        // else
        //     ambM = forwardY->tangent2vecMat(object);
        
        
        tangent_vector grad, grad1, grad2;
        // grad1 = Vptr->evalGradient(ambM.begin());     // amb_grad[0]
        // 
        // object.set_currentViewK(1);
        // forwardDirection.set_currentViewK(1);
        // grad2 = Wptr->vectorTrans(object, forwardDirection,
        //                           forwardY);
        // object.set_currentViewK(0);
        // forwardDirection.set_currentViewK(0);
        // 
        // // Compute Omega
        // mat sA, sB, sC, sUpdate;
        // const mat &sWroot = Wptr->get_Wroot();
        // sA = Vptr->self2mat();
        // sA = sA.t() * grad1(0);
        // sA = sWroot * (sA - sA.t()) * sWroot;
        // sB = Wptr->self2mat();
        // sC = sB * grad2(0);
        // sC = sC - sC.t();
        // sA += sC;
        // sB = sB * sB;
        // arma::mat Omega = arma::syl(sB, sB, -sA);
        // 
        // // update grad1, grad2, combine to grad
        // sUpdate = sWroot * Omega * sWroot;
        // grad1(0) -= (Vptr->self2mat()) * sUpdate;
        // sUpdate = (Wptr->self2mat()) * Omega;
        // grad2(0) -= sUpdate + sUpdate.t();
        // grad = grad1 * grad2;
        // grad.ownership_set(id_code);
        return grad;
        
    };

