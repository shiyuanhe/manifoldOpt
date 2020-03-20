#ifndef FISTA_CLASS
#define FISTA_CLASS


#include "optimization.hpp"

class FISTACLASS:
    public algorithm_base{
public:
    FISTACLASS();
    
    void solve();


private:
    // internel parameters
    double n, p, size_scale, tol, LipConst;
    double t, t_pre, Xerr;
    
    arma::mat X, X_pre;

    function_ptr subp_loss;
    shared_ptr<FunPlusPenaltyClass> threshFun;
    

    void update_t();
    void init_control();

    void Fista_Init();
    void subprob_Init();
    void params_Init();
    void threshFun_Init();
    
    void FistaLoop();
};

#endif
