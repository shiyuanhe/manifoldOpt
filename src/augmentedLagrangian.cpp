#include "augmentedLagrangian.hpp"

void augmentedLagrangian::init_control(){
    control.insert("iterMax", 200);
    control.insert("iterSubMax", 50);
    control.insert("tol_dual", 0.01);
    control.insert("tol_primal", 0.01);
    control.insert("rho", 5000.0);
    control.insert("beta", 1.1);
    control.insert("verbose", 0);
}



augmentedLagrangian::augmentedLagrangian(){
    init_control();
}



void augmentedLagrangian::threshFun_Init(){
    optFunctn_downcast = dynamic_pointer_cast<FunPlusPenaltyClass>(optFunctn);
    if(!optFunctn_downcast){
        throw runtime_error("Penalty type unspecified for ALM algorithm~");
    }
}


void augmentedLagrangian::ALM_Init(){
    subprob_Init();
    Lagr_Init();
    params_Init();
    threshFun_Init();
}

void augmentedLagrangian::params_Init(){
    iterMax = control["iterMax"];
    iterSubMax = control["iterSubMax"];
    rho = control["rho"];
    X = currentY->self2mat();
    n = X.n_rows;
    p = X.n_cols;
    size_scale = sqrt(n*p);
    X_pre = X;
    tol_dual = control["tol_dual"];
    tol_primal = control["tol_primal"];
    beta = control["beta"];
    
}

void augmentedLagrangian::subprob_Init(){
    //setup loss for core algorithm
    subp_loss = make_shared<FunPlusPenaltyClass>();
    subp_loss->copy_functions(optFunctn);
    subp_loss->set_penalty(FunPlusPenaltyClass::PenaltyType::ALMLoss);
    sub_algorithm->set_optfun(subp_loss);
    sub_algorithm->set_currentY(currentY);
}

void augmentedLagrangian::Lagr_Init(){
    //initial Lagrangian multiplier
    function_ptr tmp_loss = make_shared<FunctionClass>();
    tmp_loss->copy_functions(optFunctn);
    tangent_vector L_tv = tmp_loss->gradient_at(currentY);
    Lagr =  - currentY->tangent2mat(L_tv);
}



void augmentedLagrangian::solve(){
    ALM_Init();
    ALM_Outer_Loop();
}


void augmentedLagrangian::ALM_Outer_Loop(){
    iter = 0;
    bool flag = true;
    arma::mat primal_diff;
    double primal_err;
    
    while(flag){
        iter++;
        
        ALM_Inner_Loop();
        primal_diff = currentY->self2mat() - X;
        Lagr += primal_diff*rho;
        rho *= beta;
        
        primal_err = norm(primal_diff, "fro");
        if(iter>iterMax) flag = false;
        if(iter>0 && primal_err<tol_primal*size_scale) 
            flag = false;
        if(control["verbose"] > 0)
            Rcout<<"Main Iter: "<<iter<<" "<<
                std::setprecision(4)<<primal_err<<std::endl;
    }
    
    // vecMat Y;
    // Y.resize(1);
    // Y.at(0) = X;
    // currentY->initial_point(Y);
    
}


void augmentedLagrangian::ALM_Inner_Loop(){
    iterSub = 0;
    arma::mat dual_diff;
    double dual_err;
    bool flag = true;
    while(flag){
        iterSub++;
        //Update Y
        G = X - Lagr/rho;
        subp_loss->set_ALM_params(G, rho);
        sub_algorithm->solve();
        currentY = sub_algorithm->get_currentY();
        //Update X
        X = currentY->self2mat() + Lagr/rho;
        X = optFunctn_downcast->thresh_matrix(X, 1/rho);
        //Dual Error
        dual_diff = rho * (X-X_pre);
        dual_err = norm(dual_diff,"fro");
        X_pre = X;
        //Convergence check
        if(iterSub>iterSubMax) flag = false;
        if(iterSub>1 && dual_err<tol_dual*size_scale) flag = false;
        if(control["verbose"] > 0 && 
           iterSub % 10 == 1)
            Rcout<<"Sub Iter: "<<iterSub<<
                " "<<std::setprecision(4)<<
                    dual_err<<std::endl;
    }
    
}


