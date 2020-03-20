library(rOptManifold)
library(testthat)
context("Group Penalty")
set.seed(126)
n = 200
p = 40
partI = 1:100
partII = 101:200
Y_init =  matrix(0, n, p) 
Y0 =  matrix(runif(n*p), n, p) - 0.5
Y0[partI,] = Y0[partI,] * 15

obj_fun <- function(Y){
    val = sum((Y-Y0)^2) * 0.5
    return(val)
}
grad_fun <- function(Y){
    val = Y - Y0
    return(val)
}
hess_fun <- function(Y,Z){
    return(Z)
}

test_that("The Effect of Group Lasso, Group SCAD, vs without penalty",{

optObj = new(manifoldOpt)
optObj$select_algorithm("tr")
optObj$update_control(list(tol = 1e-3))
optObj$set_euclidean(n,p)
optObj$setObjective(obj_fun)
optObj$setGradient(grad_fun)
optObj$setHessian(hess_fun)
####
optObj$initial_point(Y_init)
optObj$solve()
YoptPlain = optObj$get_optimizer()
####
optObj$initial_point(Y_init)
optObj$addPenalty("grouplasso", rep(2,n))
optObj$solve()
YoptLasso = optObj$get_optimizer()
####
optObj$initial_point(Y_init)
optObj$addPenalty("groupscad", rep(2,n))
optObj$solve()
YoptSCAD = optObj$get_optimizer()


Q11 = sum(abs(YoptPlain[partI,]))
Q12 = sum(abs(YoptPlain[partII,]))
Q21 = sum(abs(YoptLasso[partI,]))
Q22 = sum(abs(YoptLasso[partII,]))
Q31 = sum(abs(YoptSCAD[partI,]))
Q32 = sum(abs(YoptSCAD[partII,]))

expect_gt(Q31/Q21,1.05)  
expect_lt(Q22/Q12,0.1)  
expect_lt(Q32/Q12,0.1)  
print("Done")
##optObj$initial_point(Y_init)
#optObj$addPenalty("grouplasso", c(2,2))
#expect_error(optObj$solve(),"Incorrect length of vector lambda!")

})

