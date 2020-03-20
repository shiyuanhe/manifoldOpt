#include "conjugateGradient.hpp"


void conjugateGradient::solve() {
  solve_init();
  
  bool flag = true;
  double eDescent, etaCG, gradNorm_previous, gradNorm_new;
  tangent_vector GRADf_forward, GRADf_previous;
  
  iter = 0;
  objValue = optFunctn->objective_at(currentY);
  GRADf = optFunctn->gradient_at(currentY);
  eDescent = currentY->metric(GRADf, GRADf);
  GRADf_previous = GRADf;
  gradNorm_previous = eDescent;
  XI_desc = -GRADf;
  
  
  while (iter < iterMax && flag) {
    
    // gradient of objective funtion
    iter++;
    XI_desc_full = XI_desc; // full size XI_desc
    armijo_rule(eDescent);
    // if(iter==2){
    //   accept_trialY();
    //   break;
    // }
    // move XI_desc_full by the amount of XI_desc
    XI_desc_forward = currentY->vectorTrans(XI_desc_full,
                                            XI_desc, trialY);
    
    GRADf_forward = currentY->vectorTrans(GRADf_previous,
                                          XI_desc, trialY);
    // Rcout << "Compare2 = " << currentY->metric(GRADf_previous, GRADf_previous) << 
    //   " " << trialY->metric(GRADf_forward, GRADf_forward) << std::endl;
    // 
    accept_trialY();

    if(verbosePrint > 0){
      Rcout << "Iter = " << iter <<
        "; Expected descent = " << eDescent <<
        "; Actual descent = " << obj_desc << std::endl;
    }
    
    GRADf = optFunctn->gradient_at(currentY);
    
    
    GRADf_previous = GRADf;
    //Rcout << "gradNorm_new = " << std::endl;
    
    gradNorm_new = currentY->metric(GRADf, GRADf);
    
    // Fletcher-Reeves
    //etaCG = gradNorm_new / gradNorm_previous;
    
    // Polak-Ribiere, conjMethod==1
    GRADf_forward = GRADf - GRADf_forward;
    //Rcout << "etaCG = "  << std::endl;
    etaCG = currentY->metric(GRADf, GRADf_forward);
    etaCG /= gradNorm_previous;
    
    XI_desc = -GRADf + XI_desc_forward * etaCG;
    
    // check the angel of xi and descD, if >-(0.05) set descD=-xi
    // this makes sure descD is a descent direction
    
    //Rcout << "XIDesc = " << std::endl;
    currentY->metric(XI_desc, XI_desc);
    
    //Rcout << "ang_check = " <<  std::endl;
    double ang_check = currentY->metric(XI_desc, GRADf);
    eDescent = abs(ang_check);
    gradNorm_previous = gradNorm_new;
    ang_check /= sqrt(gradNorm_previous) * sqrt(gradNorm_new);
    if (ang_check > (-0.01)) {
      XI_desc = -GRADf;
      eDescent = gradNorm_previous;
    }
    
    if (tol > gradNorm_previous || obj_desc < tol)
      flag = false;
    
  } // outer iteration
}




void conjugateGradient::init_control() {
  control.insert("iterMax", 200);
  control.insert("iterSubMax", 50);
  control.insert("tol", 0.01);
  control.insert("sigma", 0.2);
  control.insert("beta", 0.63);
  control.insert("alpha", 0.5);
  control.insert("verbose", 0);
}

void conjugateGradient::solve_init() {
  tol = control["tol"];
  beta = control["beta"];
  sigma = control["sigma"];
  alpha = control["alpha"];
  iterMax = control["iterMax"];
  iterSubMax = control["iterSubMax"];
  verbosePrint = control["verbose"];
}


// choose appropirate step size according to Armijo rule
void conjugateGradient::armijo_rule(double expected_desc) {
  expected_desc *= sigma;
  iterSub = 0;
  XI_desc = XI_desc * alpha;
  obj_desc = expected_desc;
  do {
    iterSub++;
    expected_desc = expected_desc * beta;
    XI_desc = XI_desc * beta;
    try{
        trial_moveBy_XIDesc();
    }catch(const std::exception& e){
        //Rcpp::Rcerr << "Caught In Manifold!"<<endl;
        obj_desc = expected_desc;
    }
  } while (obj_desc < expected_desc && iterSub < iterSubMax);
  
}
