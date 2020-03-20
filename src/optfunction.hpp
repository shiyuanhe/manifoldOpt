#ifndef MANIFOLD_OPT_FUNCTION_CLASS
#define MANIFOLD_OPT_FUNCTION_CLASS

#include "manifold.hpp"

class FunctionClass;
typedef shared_ptr<FunctionClass> function_ptr;

class FunctionClass {
public:
  FunctionClass();
  virtual ~FunctionClass() {}

  void setObjective(nullable_type ftmp) { objective_f = ftmp; }
  void setGradient(nullable_type ftmp) { gradient_f = ftmp; }
  void setHessian(nullable_type ftmp) { hessian_f = ftmp; }
  void copy_functions(function_ptr tmp) {
    objective_f = tmp->objective_f;
    gradient_f = tmp->gradient_f;
    hessian_f = tmp->hessian_f;
  }

  bool objIsNull() { return objective_f.isNull(); }
  bool gradIsNull() { return gradient_f.isNull(); }
  bool hessIsNull() { return hessian_f.isNull(); }

  double objective_at(const manifold_ptr mPoint);
  auto gradient_at(const manifold_ptr mPoint) -> tangent_vector;
  auto hessian_at(const manifold_ptr mPoint, tangent_vector &Z)
      -> tangent_vector;

  // SEXP mPoint_to_SEXP(const manifold_ptr mPoint);
  // vecMat SEXP_to_vecMat(SEXP sexpObject, unsigned nc);

protected:
  nullable_type objective_f, gradient_f, hessian_f;

  virtual double ambient_objective(const manifold_ptr mPoint);
  virtual vecMat ambient_gradient(const manifold_ptr mPoint);
  virtual vecMat ambient_hessian(const manifold_ptr mPoint,
                                  tangent_vector &Z);

  // SEXP vecMat_to_SEXP(const vecMat &, unsigned nc);
};

#endif
