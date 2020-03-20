#ifndef MANIFOLD_OPTIMIZATION_CLASS
#define MANIFOLD_OPTIMIZATION_CLASS

#include "manifold.hpp"
#include "optfunction.hpp"
#include "options.hpp"
#include "penaltyClass.hpp"

class algorithm_base;
typedef shared_ptr<algorithm_base> algorithm_ptr;

class algorithm_base {
public:
  algorithm_base() {
    optFunctn = make_shared<FunctionClass>();
    currentY = nullptr;
    trialY = nullptr;
    init_control();
  }

  virtual ~algorithm_base() {}

  virtual void solve() {}

  rOptions get_control() { return control; }

  void set_control(rOptions control_) { control = control_; }

  void set_optfun(function_ptr tmpF) { optFunctn = tmpF; }
  void set_currentY(manifold_ptr tmpY) { currentY = tmpY; }
  // Get the main optimizer. Reside on the manifold. 
  auto get_currentY() -> manifold_ptr { return currentY; }
  // Get secondary optimizer, e.g., in augmented lagrange algorithm
  // to increase sparsity of the solution
  virtual mat get_currentY_secondary() { return arma::mat(1,1); }

  void set_sub_algorithm(algorithm_ptr tmpA) { sub_algorithm = tmpA; }

protected:
  rOptions control;
  function_ptr optFunctn;
  manifold_ptr currentY, trialY;
  algorithm_ptr sub_algorithm;

  double objValue, objValue_trial, obj_desc; // the current obj function value
  // descent direction in the tangent space of currentY
  tangent_vector GRADf, HESSf, XI_desc;
  int iterMax, iterSubMax, iter, iterSub;
  int verbosePrint;
  
  virtual void init_control() {}

  void trial_moveBy_XIDesc() {
    trialY = currentY->retraction(XI_desc);
    objValue_trial = optFunctn->objective_at(trialY);
    obj_desc = objValue - objValue_trial;
  }

  void accept_trialY() {
    currentY = trialY;
    objValue = objValue_trial;
  }
};

#endif
