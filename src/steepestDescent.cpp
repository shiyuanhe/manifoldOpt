#include "steepestDescent.hpp"

void steepestDescent::init_control() {
  control.insert("iterMax", 200);
  control.insert("iterSubMax", 50);
  control.insert("tol", 0.01);
  control.insert("sigma", 0.2);
  control.insert("beta", 0.63);
  control.insert("alpha", 0.5);
  control.insert("verbose", 0);
}

void steepestDescent::solve_init() {
  tol = control["tol"];
  beta = control["beta"];
  sigma = control["sigma"];
  alpha = control["alpha"];
  iterMax = control["iterMax"];
  iterSubMax = control["iterSubMax"];
  verbosePrint = control["verbose"];
}

void steepestDescent::solve() {

  solve_init();
  // Initialization of functions and control parameters

  double gradient_norm;
  iter = 0;
  bool flag = true;

  objValue = optFunctn->objective_at(currentY);

  while (iter < iterMax && flag) {
    iter++;
    GRADf = optFunctn->gradient_at(currentY);
    gradient_norm = currentY->metric(GRADf, GRADf); //??
    XI_desc = -GRADf;
    armijo_rule(gradient_norm);
    // accept trialY
    accept_trialY();
    if (gradient_norm < tol  || obj_desc < tol)
      flag = false;
    
    if(verbosePrint > 0){
      Rcout << "Iter = " << iter <<
        "; Expected descent = " << gradient_norm <<
          "; Actual descent = " << obj_desc << std::endl;
    }
    
    
  } // outer iteration

} // end of function

// choose appropirate step size according to Armijo rule
void steepestDescent::armijo_rule(double expected_desc) {
  expected_desc *= sigma;
  iterSub = 0;
  XI_desc = XI_desc * alpha;
  do {
    try{
      iterSub++;
      expected_desc = expected_desc * beta;
      XI_desc = XI_desc * beta;
      trial_moveBy_XIDesc();
      
      if(verbosePrint > 1){
          Rcout << "iterSub = " << iterSub <<
              "; Expected descent = " << expected_desc <<
                  "; Actual descent = " << obj_desc << std::endl;
      }
      
    }catch(std::logic_error e){
      Rcout << e.what();
    }catch(std::runtime_error e){
      Rcout << e.what();
    }

  } while (obj_desc < expected_desc && iterSub < iterSubMax);
}
