#ifndef MANIFOLD_CLASS
#define MANIFOLD_CLASS

#include "manifold_include.hpp"
#include "tangentVector.hpp"
#include "vecMat.hpp"

class manifold {
protected:
  arma::mat Y; // Y is the maniold matrix
  unsigned n, p, r;
  unsigned retraction_type; // retraction: !!! implement as enumerate type!!!!
  
  vector<string> tangent_names;
  
  unsigned comp_num;     // the number of product component
  unsigned tangent_size; // tangent vector usually represented by only one
  // matrix, = 1
  
  string id_code; // for tangent vector operation, in same tangent space
  
  // generate a unique ID_Code, each time the value Y changes.
  // This is to ensure only the tangent vectors in the same
  // tangent space could be added or subtracted.
  void generate_id_code() {
    static const char alphanum[] = "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";
    int len = 10;
    id_code.clear();
    for (int i = 0; i < len; ++i) {
      char tmp = alphanum[rand() % (sizeof(alphanum) - 1)];
      id_code.push_back(tmp);
    }
  }
  
  auto create_tangentV_at_current_point() -> tangent_vector {
    tangent_vector xi(id_code);
    xi.resize(tangent_size);
    xi.assign_names(tangent_names);
    return xi;
  }
  
public:
  manifold(unsigned n_, unsigned p_, unsigned r_) {
    n = n_;
    p = p_;
    r = r_;
    retraction_type = 0;
    comp_num = 1;
    tangent_size = 1;
    tangent_names.clear();
    tangent_names.push_back("xi");
    generate_id_code();
  }
  
  virtual ~manifold() {}
  
  unsigned get_comp_num() { return comp_num; }
  
  // **** must change in every derived class ****
  // Compute ambient gradient to its tangent space
  virtual auto evalGradient(vecMatIter amb_grad) -> tangent_vector = 0;
  
  // Compute ambient hessian to its tangent space
  // Ambient gradient is usually needed in the computation
  // Hess(Z_tv)
  virtual auto evalHessian(vecMatIter amb_grad, vecMatIter amb_hess,
                           tangent_vector &Z_tv) -> tangent_vector = 0;
  
  virtual auto retraction(tangent_vector &) -> manifold_ptr = 0;
  
  // vector transport on xi
  // The previous Manifold is Mpre
  // It was moved in the direction Mdir
  // , const manifold_ptr &Mpre, tangent_vector &Mdir
  virtual auto vectorTrans(tangent_vector & object, 
                           tangent_vector & forwardDirection,
                           manifold_ptr forwardY) 
    -> tangent_vector {
      
      vecMat ambM;
      if (forwardY->get_comp_num() == 1)
        ambM.push_back(forwardY->tangent2mat(object));
      else
        ambM = forwardY->tangent2vecMat(object);
      
      tangent_vector eta = forwardY->evalGradient(ambM.begin());
      return eta;
    };
  
  // virtual void update_conjugateD ( .... );
  
  // following are operation for single component manifold
  // and single component tangent vector representation
  // change them in derived class accordingly
  virtual double metric(tangent_vector &xi, tangent_vector &eta) {
    // xi.ownership_check(id_code);
    // eta.ownership_check(id_code);
    return arma::dot(xi(), eta());
  }
  
  virtual void initial_point(vecMatIter rObj) { Y = *rObj; }
  
  virtual vecMat self2vecMat() {
    vecMat res;
    res.push_back(Y);
    return res;
  }
  
  virtual arma::mat self2mat() { return Y; }
  
  virtual arma::mat tangent2mat(tangent_vector &xi) {
    // xi.ownership_check(id_code);
    return xi(0);
  }
  
  virtual vecMat tangent2vecMat(tangent_vector &xi) {
    // check if xi belong to the tangent space at current Y
    // xi.ownership_check(id_code);
    vecMat res;
    res.push_back(xi(0));
    return res;
  }
};

#endif

// gradient and hessian in the ambient space
// arma::mat gradF,hessianF;

// xi and xi_normal: gradient in the tangent space and normal space
// hessian_Z: hessian operator on Z, as a result of function
// evalHessian_manifold() below
// descD: descent direction, used as conjugate direction in conjugate descent
// algorithm
// conjugdateD: conjugate descent direction
// arma::mat xi,xi_normal,hessian_Z,descD;
// Y is a n-by-p matrix of rank r
// retraction is type of retraction expressed in 0,1,2,3,....
//' Set the conjugate direction
//'
//' @param set eta as the new conjugate (tangent) direction conjugate_temp
// virtual void retrieve_steepest(){descD=-xi;}

//' Set initial particle swarm;
// virtual void set_particle () {}

// double gradMetric(){return metric(xi,xi);}
// double descDMetric(){return metric(descD,descD);}
// double grad_descD_Metric(){return metric(xi,descD);}
//
// //return related value or matrix
// double get_eDescent(){ return eDescent; }
//
// arma::mat get_hessianZ() {return hessian_Z; }
// arma::mat get_Gradient(){return xi;}
// arma::mat get_descD(){return descD;}
// //set the descent direction other than the steepest descent direction
// virtual void set_descD(arma::mat xi_temp){descD=xi_temp;}
