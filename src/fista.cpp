#include "fista.hpp"

FISTACLASS::FISTACLASS(){
    init_control();
}

void FISTACLASS::init_control(){
    control.insert("iterMax", 200);
    control.insert("tol", 0.01);
    control.insert("LipConst", 0.01);
    control.insert("verbose", 0);
}


void FISTACLASS::solve(){
    Fista_Init();
    FistaLoop();
}


void FISTACLASS::Fista_Init(){
    subprob_Init();
    params_Init();
    threshFun_Init();
}





void FISTACLASS::params_Init(){
    iterMax = control["iterMax"];
    tol = control["tol"];
    LipConst = control["LipConst"];

    X = currentY->self2mat();
    n = X.n_rows;
    p = X.n_cols;
    size_scale = sqrt(n*p);
    X_pre = X;
}

void FISTACLASS::subprob_Init(){
    subp_loss = make_shared<FunctionClass>();
    subp_loss->copy_functions(optFunctn);
}


void FISTACLASS::threshFun_Init(){
    threshFun = dynamic_pointer_cast<FunPlusPenaltyClass>(optFunctn);
    if(!threshFun){
        throw runtime_error("Penalty unspecified for FISTA algorithm~");
    }
}

inline void FISTACLASS::update_t(){
    t = 1.0 + sqrt( 1.0 + 4.0 * t_pre * t_pre);
    t /= 2.0;
}


void FISTACLASS::FistaLoop(){
    iter = 0;
    bool flag = true;
    arma::mat XDiff;
    vecMat Y;
    Y.resize(1);

    t_pre = 1.0;


    while(flag){
        
        //move forward
        GRADf = subp_loss->gradient_at(currentY);
        XI_desc =  - GRADf * (1.0 / LipConst);
        trial_moveBy_XIDesc();
        accept_trialY();
        
        //retraction
        X = currentY->self2mat();
        X = threshFun->thresh_matrix(X, 1/LipConst);

        //new expansion point
        update_t();
        Y.at(0) = X + (t_pre - 1)/t * (X - X_pre);
        currentY->initial_point(Y.begin());
        
        //check for convergence
        Xerr = norm(X - X_pre, "fro");
        if(iter>iterMax) flag = false;
        if(iter>400 && Xerr<tol*size_scale) flag = false;
        if(control["verbose"] > 0  && iter % 10 == 0)
            Rcout<<iter<<" "<<setprecision(4)<<
                Xerr<<std::endl;
        
        X_pre = X;
        t_pre = t;
        iter++;
        
    }
    
    Y.at(0) = X;
    currentY->initial_point(Y.begin());
    
}


