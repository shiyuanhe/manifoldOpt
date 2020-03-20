context("Fixed Rank Manifold")
set.seed(100)
rm(list=ls())
library(rOptManifold)

n = 60; p = 50; r = 3
U=matrix(runif(n*r)*10,ncol=r)
U = qr.Q(qr(U))
V=matrix(runif(p*r)*10,ncol=r)
V = qr.Q(qr(V))
sigma=diag(c(2e3,6e2,4e2))
lowR=U%*%sigma%*%t(V)

mask=matrix(runif(n*p)>0.5,ncol=p)
obj = function(X){
    temp=0.5*(lowR-X)^2
    val = sum(temp[mask])
    return(val)
}

grad = function(X){
    temp=-lowR+X
    temp[!mask] = 0
    return(temp)
}

hessian = function(X,Z){
    temp=Z
    temp[!mask] = 0
    return(temp)
}


## What if when err too small??
test_that("Error Setup Check",{
    problem = new(manifoldOpt)
    problem$select_algorithm("tr")
    expect_error(problem$solve(),"Manifold Type Unspecified!")
    problem$set_fixRank(n,p,r)
    expect_error(problem$solve(),"Objective Function Unspecified!")
    problem$setObjective(obj)
    expect_error(problem$solve(),"Gradient Function Unspecified!")
    problem$setGradient(grad)
    expect_error(problem$solve(),"Hessian Function Unspecified!")
})


## What if when err too small??
test_that("Matrix Completion Trust Region",{
    problem = new(manifoldOpt)
    problem$select_algorithm("tr")
    problem$set_fixRank(n,p,r)
    problem$update_control(list(tol = 1e-3*sqrt(n*p)))
    problem$setObjective(obj)
    problem$setGradient(grad)
    problem$setHessian(hessian)
    problem$solve()
    temp=problem$get_optimizer()-lowR
    recovery_err = sqrt(sum(temp[!mask]^2))/sqrt(sum(!mask))
    expect_lt(recovery_err, 1e-1)
})


test_that("Matrix Completion Steepest Descent",{
    problem = new(manifoldOpt)
    problem$set_fixRank(n,p,r)
    problem$update_control(list(tol = 1e-6*sqrt(n*p)))
    problem$setObjective(obj)
    problem$setGradient(grad)
    problem$solve()
    temp=problem$get_optimizer()-lowR
    recovery_err = sqrt(sum(temp[!mask]^2))/sqrt(sum(!mask))
    expect_lt(recovery_err, 1e-1)
})
