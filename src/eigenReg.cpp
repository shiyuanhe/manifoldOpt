#include "eigenReg.hpp"


eigenReg::eigenReg(unsigned n1, unsigned p1, unsigned r1)
    : manifold(n1, p1, r1) {
    // R^{n1\tiems p1} with n1=p1, matrix rank r1
    Wptr = make_shared<psdCone>(r1, r1, r1);
    Vptr = make_shared<stiefel>(n1, r1, r1);
    
    tangent_names.clear();
    tangent_names.push_back("Unkown~");
    tangent_size = 1;
    comp_num = 2;
}

void eigenReg::initial_point(vecMatIter rObj) {
    Vptr->initial_point(rObj);
    Wptr->initial_point(rObj + 1);
}

auto eigenReg::evalGradient(vecMatIter amb_grad) -> tangent_vector {
    tangent_vector grad, grad1, grad2;
    grad1 = Vptr->evalGradient(amb_grad);     // amb_grad[0]
    grad2 = Wptr->evalGradient(amb_grad + 1); // amb_grad[1]
    
    // Compute Omega
    mat sA, sB, sC, sUpdate;
    const mat &sWroot = Wptr->get_Wroot();
    sA = Vptr->self2mat();
    sA = sA.t() * grad1(0);
    sA = sWroot * (sA - sA.t()) * sWroot;
    sB = Wptr->self2mat();
    sC = sB * grad2(0);
    sC = sC - sC.t();
    sA += sC;
    sB = sB * sB;
    arma::mat Omega = arma::syl(sB, sB, -sA);
    
    // update grad1, grad2, combine to grad
    sUpdate = sWroot * Omega * sWroot;
    grad1(0) -= (Vptr->self2mat()) * sUpdate;
    sUpdate = (Wptr->self2mat()) * Omega;
    grad2(0) -= sUpdate + sUpdate.t();
    grad = grad1 * grad2;
    grad.ownership_set(id_code);
    return grad;
}

auto eigenReg::evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                           tangent_vector &Z_tv) -> tangent_vector {
                               tangent_vector tv;
                               // empty
                               
                               return tv;
                           }

auto eigenReg::retraction(tangent_vector &xi) -> manifold_ptr {
    manifold_ptr tmpPtr;
    shared_ptr<eigenReg> result = make_shared<eigenReg>(n, p, r);
    xi.set_currentViewK(0);
    tmpPtr = Vptr->retraction(xi);
    result->Vptr = std::dynamic_pointer_cast<stiefel>(tmpPtr);
    xi.set_currentViewK(1);
    tmpPtr = Wptr->retraction(xi);
    result->Wptr = std::dynamic_pointer_cast<psdCone>(tmpPtr);
    xi.set_currentViewK(0);
    
    // double diff1, diff2;
    // diff1 = norm(Vptr->self2mat() - result->Vptr->self2mat(), "fro");
    // diff2 = norm(Wptr->self2mat() - result->Wptr->self2mat(), "fro");
    // Rcout << "diff = " <<diff1<< " " << diff2 << std::endl;
    
    return result;
}

double eigenReg::metric(tangent_vector &xi, tangent_vector &eta) {
    
    double result = 0, res2;
    xi.set_currentViewK(0);
    eta.set_currentViewK(0);
    result = Vptr->metric(xi, eta);
    xi.set_currentViewK(1);
    eta.set_currentViewK(1);
    res2 = 0.5 * Vptr->metric(xi, eta);
    
    // Rcout << "metric = " <<result<< " " << res2 << std::endl;
    
    result += res2;
    xi.set_currentViewK(0);
    eta.set_currentViewK(0);
    
    return result;
}

vecMat eigenReg::self2vecMat() {
    vecMat result;
    result.resize(2);
    result.at(0) = Vptr->self2mat();
    result.at(1) = Wptr->self2mat();
    return result;
}

vecMat eigenReg::tangent2vecMat(tangent_vector &xi) {
    vecMat result;
    xi.set_currentViewK(0);
    result.push_back(Vptr->tangent2mat(xi));
    xi.set_currentViewK(1);
    result.push_back(Wptr->tangent2mat(xi));
    return result;
}



auto eigenReg::vectorTrans(tangent_vector &object, 
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
        vecMat ambM;
        if (forwardY->get_comp_num() == 1)
            ambM.push_back(forwardY->tangent2mat(object));
        else
            ambM = forwardY->tangent2vecMat(object);
        
        
        tangent_vector grad, grad1, grad2;
        grad1 = Vptr->evalGradient(ambM.begin());     // amb_grad[0]
        
        object.set_currentViewK(1);
        forwardDirection.set_currentViewK(1);
        grad2 = Wptr->vectorTrans(object, forwardDirection,
                                  forwardY);
        object.set_currentViewK(0);
        forwardDirection.set_currentViewK(0);
        
        // Compute Omega
        mat sA, sB, sC, sUpdate;
        const mat &sWroot = Wptr->get_Wroot();
        sA = Vptr->self2mat();
        sA = sA.t() * grad1(0);
        sA = sWroot * (sA - sA.t()) * sWroot;
        sB = Wptr->self2mat();
        sC = sB * grad2(0);
        sC = sC - sC.t();
        sA += sC;
        sB = sB * sB;
        arma::mat Omega = arma::syl(sB, sB, -sA);
        
        // update grad1, grad2, combine to grad
        sUpdate = sWroot * Omega * sWroot;
        grad1(0) -= (Vptr->self2mat()) * sUpdate;
        sUpdate = (Wptr->self2mat()) * Omega;
        grad2(0) -= sUpdate + sUpdate.t();
        grad = grad1 * grad2;
        grad.ownership_set(id_code);
        return grad;
        
    };

