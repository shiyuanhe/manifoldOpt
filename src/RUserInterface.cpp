#include "RUserInterface.hpp"

void RUserInterface::addPenalty(string pName, arma::vec lambda_) {
  shared_ptr<FunPlusPenaltyClass> tmp = make_shared<FunPlusPenaltyClass>();
  std::transform(pName.begin(), pName.end(), pName.begin(), ::tolower);
  tmp->copy_functions(objective_fun);
  tmp->set_lambda(lambda_);

  if (pName == "grouplasso") {
    tmp->set_penalty(FunPlusPenaltyClass::PenaltyType::groupLASSO);
  } else if (pName == "groupscad") {
    tmp->set_penalty(FunPlusPenaltyClass::PenaltyType::groupSCAD);
  } else {
    throw runtime_error("Unkown Penalty Type!");
  }

  objective_fun = tmp;
}

SEXP RUserInterface::get_optimizer() {
  unsigned nc = currentY->get_comp_num();
  SEXP result;
  if (nc == 1) {
    result = Rcpp::wrap(currentY->self2mat());
  } else {
    vecMat tmp;
    tmp = currentY->self2vecMat();
    result = Rcpp::wrap(vecMat2List(tmp));
  }
  return result;
}

mat RUserInterface::get_optimizer_secondary(){
  return main_algorithm->get_currentY_secondary();
}

void RUserInterface::removePenalty() {
  function_ptr tmp = make_shared<FunctionClass>();
  tmp->copy_functions(objective_fun);
  objective_fun = tmp;
}

void RUserInterface::solve_check() {
  if (!currentY)
    throw runtime_error("Manifold Type Unspecified!");

  if (objective_fun->objIsNull())
    throw runtime_error("Objective Function Unspecified!");
  if (objective_fun->gradIsNull())
    throw runtime_error("Gradient Function Unspecified!");
  if ((algorithm_name == "tr" || sub_algorithm_name == "tr") &&
      objective_fun->hessIsNull())
    throw runtime_error("Hessian Function Unspecified!");

  shared_ptr<FunPlusPenaltyClass> tmp =
      dynamic_pointer_cast<FunPlusPenaltyClass>(objective_fun);
  if (tmp && currentY->get_comp_num() > 1) {
    throw runtime_error(
        "Only single component manifold could specify penalty!");
    // lambda check?
  }

  if (algorithm_name == "alm" && !sub_algorithm) {
    throw runtime_error("Subalgorithm Unspecified for ALM!");
  }
}

void RUserInterface::solve_init() {
  main_algorithm->set_currentY(currentY);
  main_algorithm->set_optfun(objective_fun);
  main_algorithm->set_control(control);
  if (algorithm_name == "alm") {
    sub_algorithm->set_control(sub_control);
    main_algorithm->set_sub_algorithm(sub_algorithm);
  }
}

void RUserInterface::update_control(List rList) { control.Rset_options(rList); }
void RUserInterface::display_control() { control.Rprint_options(); }
void RUserInterface::update_subControl(List rList) {
  sub_algorithm_check();
  sub_control.Rset_options(rList);
}
void RUserInterface::display_subControl() {
  sub_algorithm_check();
  sub_control.Rprint_options();
}

void RUserInterface::select_algorithm(string aName) {
  std::transform(aName.begin(), aName.end(), aName.begin(), ::tolower);
  if (aName == algorithm_name)
    return;

  main_algorithm = select_algorithm_base(aName);
  algorithm_name = aName;
  control = main_algorithm->get_control();
  sub_algorithm_clear();
}
void RUserInterface::select_sub_algorithm(string aName) {
  std::transform(aName.begin(), aName.end(), aName.begin(), ::tolower);
  if (aName == sub_algorithm_name)
    return;

  sub_algorithm_check();
  sub_algorithm = select_algorithm_base(aName);
  sub_algorithm_name = aName;
  sub_control = sub_algorithm->get_control();
}

auto RUserInterface::select_algorithm_base(string aName) -> algorithm_ptr {
  algorithm_ptr ptr;
  int typeID = 0;
  vector<string> allAlgorithms({"sd", "tr", "alm", "fista", "cg"});
  vector<string>::iterator it =
      find(allAlgorithms.begin(), allAlgorithms.end(), aName);
  typeID = it - allAlgorithms.begin();

  switch (typeID) {
  case 0:
    ptr = make_shared<steepestDescent>();
    break;
  case 1:
    ptr = make_shared<trustRegion>();
    break;
  case 2:
    ptr = make_shared<augmentedLagrangian>();
    break;
  case 3:
    ptr = make_shared<FISTACLASS>();
    break;
  case 4:
    ptr = make_shared<conjugateGradient>();
    break;
  default:
    throw runtime_error("Unkonw Algorithm Type!");
    break;
  }

  return ptr;
}

void RUserInterface::sub_algorithm_clear() {
  sub_control.clear();
  sub_algorithm_name = "";
  sub_algorithm = nullptr;
}

void RUserInterface::sub_algorithm_check() {
  if (algorithm_name != "alm")
    throw runtime_error("Only ALM method could set sub control!");
}

RCPP_MODULE(manifold_mod) {

  class_<RUserInterface>("manifoldOpt")
      .constructor()
      .method("solve", &RUserInterface::solve)
      .method("addPenalty", &RUserInterface::addPenalty)
      .method("removePenalty", &RUserInterface::removePenalty)
      .method("setObjective", &RUserInterface::setObjective)
      .method("setGradient", &RUserInterface::setGradient)
      .method("setHessian", &RUserInterface::setHessian)

      .method("initial_point", &RUserInterface::initial_point)
      .method("get_optimizer", &RUserInterface::get_optimizer)
      .method("get_optimizer_secondary", &RUserInterface::get_optimizer_secondary)
      .method("select_algorithm", &RUserInterface::select_algorithm)
      .method("select_sub_algorithm", &RUserInterface::select_sub_algorithm)

      .method("update_control", &RUserInterface::update_control)
      .method("display_control", &RUserInterface::display_control)
      .method("update_subControl", &RUserInterface::update_subControl)
      .method("display_subControl", &RUserInterface::display_subControl)

      .method("set_euclidean", &RUserInterface::set_euclidean)
      .method("set_fixRank", &RUserInterface::set_fixRank)
      .method("set_fixRankPSD", &RUserInterface::set_fixRankPSD)
      .method("set_stiefel", &RUserInterface::set_stiefel)
      .method("set_grassmannQ", &RUserInterface::set_grassmannQ)
      .method("set_eigenReg", &RUserInterface::set_eigenReg)
      .method("set_eigenRegUnsym", &RUserInterface::set_eigenRegUnsym);
;
};
