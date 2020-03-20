#include "trustRegion.hpp"

void trustRegion::init_control(){
    control.insert("iterMax", 200);
    control.insert("iterSubMax", 50);
    control.insert("tol", 0.01);
    control.insert("Delta0", 5.0);
    control.insert("DeltaMax", 50);
    control.insert("rhoMin", 0.1);
    control.insert("kappa", 0.1);
    control.insert("theta", 0.1);
    control.insert("verbose", 0);
}


void trustRegion::solve_init(){
    tol = control["tol"];
    rhoMin = control["rhoMin"];
    theta = control["theta"];
    kappa = control["kappa"];
    DeltaMax = control["DeltaMax"];
    iterMax = control["iterMax"];
    iterSubMax = control["iterSubMax"];
    verbosePrint = control["verbose"];
}

void  trustRegion::solve(){
    solve_init();

    double  Delta = control["Delta0"];
    bool flag = true, bdryTouched;
    double rho, xi_norm, objValue_trial_approx, obj_desc, approx_desc;
    
    objValue = optFunctn->objective_at(currentY);
    //begin iteration
    iter = 0;
    while(iter<iterMax && flag){
        
        GRADf = optFunctn->gradient_at(currentY);

        //subproblem, find optimal XI_desc
        bdryTouched = tR_subproblem(Delta);
        xi_norm = currentY->metric(XI_desc, XI_desc);
        
        // move along XI_desc, compute objective fun at trialY
        trial_moveBy_XIDesc();
        obj_desc = objValue - objValue_trial;
        
        //with obj value and GRADf computed, (hidden dependence)
        HESSf = optFunctn->hessian_at(currentY, XI_desc);
        objValue_trial_approx = tR_secondOrderApprox(XI_desc);
        approx_desc = objValue - objValue_trial_approx + 1e-10;
        
        rho = obj_desc/approx_desc;
        Delta = tR_updateDelta(Delta, rho, bdryTouched);
        if(rho > rhoMin) accept_trialY();

        // convergence check
        iter++;
        if(xi_norm < tol) flag = false;
        
        if(verbosePrint > 0){
          Rcout << "Iter = " << iter <<
            "; Approx descent = " << approx_desc <<
              "; Actual descent = " << obj_desc << std::endl;
        }
        
    }// outer iteration
    
}//end of function



bool trustRegion::tR_subproblem(double Delta){
    iterSub = 0;
    tR_subp_initialVecs();
    while(iterSub < iterSubMax && rrNorm>rrNorm0){
        HESSf = optFunctn->hessian_at(currentY, dd);

        hessQterm= currentY->metric(HESSf, dd);  
        if(hessQterm<=0.0001){  // the direction of negative curvature
            tR_boundary_XIDesc(XI_desc, dd, Delta);
            return true;
        }
        tR_subp_updateVecs();
        XI_norm = currentY->metric(XI_desc, XI_desc);
        if(XI_norm >0.9999*Delta*Delta){ //The optimizer is out of trust region
            tR_boundary_XIDesc(XI_desc_old, dd_old, Delta);
            return true;
        }
        iterSub++;
    }
    return false;
}


// controlling rr,dd,XI_desc, X_desc_old, rrNorm
void trustRegion::tR_subp_initialVecs(){
    
    
    rr = GRADf;
    dd = -GRADf; //little delta, n-by-p
    XI_desc = GRADf * 0.0;
    rrNorm = currentY->metric(rr,rr);
    rrNorm0 = rrNorm * min(pow(rrNorm,theta),kappa);
}

// controlling rr,dd,XI_desc, X_desc_old, rrNorm
void trustRegion::tR_subp_updateVecs(){
    double alpha, beta;
    XI_desc_old = XI_desc;
    dd_old = dd;
    
    alpha = rrNorm/hessQterm;
    XI_desc = XI_desc + dd*alpha;
    rr = rr + HESSf * alpha;
    beta = currentY->metric(rr,rr)/rrNorm;
    dd = -rr + dd * beta;
    rrNorm = beta*rrNorm;
}

//find tau, such that (eta + xi*tau) has norm Delta
// ?? guaranttee solution??
// ||eta||<Delta ??
void trustRegion::tR_boundary_XIDesc( tangent_vector & eta, 
                                      tangent_vector & xi, 
                                     double Delta){
    double qA, qB, qC,tau,DELTA;
    qA = currentY->metric(xi,xi);
    qB = 2* currentY->metric(eta,xi);
    qC = -Delta*Delta + currentY->metric(eta,eta);
    DELTA = qB*qB-4*qA*qC;
    if(DELTA<0 || qA==0) throw runtime_error("DELTA Error!");
    tau = sqrt(qB*qB-4*qA*qC);
    tau = (tau-qB)/(2*qA);
    if(tau<0) throw runtime_error("DELTA<0");
    XI_desc = eta + xi*tau;
}

//given obj value and GRADf (hidden dependence)
double trustRegion::tR_secondOrderApprox( tangent_vector & eta){
    double res = objValue;
    res += currentY->metric(GRADf, eta);
    res += currentY->metric(HESSf, eta) * 0.5;
    return res;
}


double trustRegion::tR_updateDelta(double Delta, 
                                   double rho,
                                   bool bdryTouched){
    if(rho<0.25){
        Delta = 0.25*Delta;
    }else if(rho>0.75 && bdryTouched){
        Delta = min(2*Delta,DeltaMax);
    }
    return Delta;
}



