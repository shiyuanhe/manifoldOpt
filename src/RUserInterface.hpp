#ifndef RUserInterface_CLASS
#define RUserInterface_CLASS

#include "options.hpp"

#include "augmentedLagrangian.hpp"
#include "conjugateGradient.hpp"
#include "fista.hpp"
#include "penaltyClass.hpp"
#include "steepestDescent.hpp"
#include "trustRegion.hpp"

#include "eigenReg.hpp"
#include "eigenRegUnsym.hpp"

#include "euclidean.hpp"
#include "fixRank.hpp"
#include "grassmannQ.hpp"
#include "stiefel.hpp"
#include "fixRankPSD.hpp"

class RUserInterface {
public:
  RUserInterface() {
    objective_fun = make_shared<FunctionClass>();
    currentY = nullptr;
    main_algorithm = nullptr;
    select_algorithm("sd"); // steepest descent by default
    sub_algorithm_clear();
  }

  void solve() {
    solve_check();
    solve_init();
    main_algorithm->solve();
    currentY = main_algorithm->get_currentY();
  }

  void initial_point(SEXP rObject) {
    unsigned nc = currentY->get_comp_num();
    vecMat init_mats = SEXP_to_vecMat(rObject, nc);
    currentY->initial_point(init_mats.begin());
  }

  SEXP get_optimizer();
  mat get_optimizer_secondary();
    
  void update_control(List rList);
  void display_control();
  void update_subControl(List rList);
  void display_subControl();

  void select_algorithm(string aName);
  void select_sub_algorithm(string aName);

  void sub_algorithm_check();
  void sub_algorithm_clear();

  void setObjective(nullable_type ftmp) { objective_fun->setObjective(ftmp); }
  void setGradient(nullable_type ftmp) { objective_fun->setGradient(ftmp); }
  void setHessian(nullable_type ftmp) { objective_fun->setHessian(ftmp); }
  void addPenalty(string pName, arma::vec lambda_);
  void removePenalty();

  void set_euclidean(int n, int p) {
    currentY = make_shared<euclidean>(n, p, 0);
  }
  void set_stiefel(int n, int p) { currentY = make_shared<stiefel>(n, p, 0); }
  void set_grassmannQ(int n, int p) {
    currentY = make_shared<grassmannQ>(n, p, 0);
  }
  void set_fixRank(int n, int p, int r) {
    currentY = make_shared<fixRank>(n, p, r);
  }
  
  void set_fixRankPSD(int n, int r) { currentY = make_shared<fixRankPSD>(n, n, r); }
  
  void set_eigenReg(int n, int r) { currentY = make_shared<eigenReg>(n, n, r); }
  void set_eigenRegUnsym(int n, int p, int r) { currentY = make_shared<eigenRegUnsym>(n, p, r); }
  
private:
  string algorithm_name, sub_algorithm_name;
  rOptions control, sub_control;
  algorithm_ptr main_algorithm, sub_algorithm;
  function_ptr objective_fun;
  manifold_ptr currentY;

  auto select_algorithm_base(string aName) -> algorithm_ptr;
  void solve_check();
  void solve_init();
};

#endif
