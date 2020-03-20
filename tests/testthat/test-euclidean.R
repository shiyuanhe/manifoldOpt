context("Euclidean Space")

set.seed(100)
n = 100
p = 40
Y0 =  matrix(runif(n*p), n, p) * 100 - 50
Y_init = Y0 * 0
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


optObj = new(manifoldOpt)
optObj$update_control(list(tol = 1e-3))
optObj$set_euclidean(n,p)
optObj$setObjective(obj_fun)
optObj$setGradient(grad_fun)
optObj$setHessian(hess_fun)


test_that("Euclidean Steepest Descent",{
    optObj$solve()
    Yopt = optObj$get_optimizer()
    error_y = mean(abs(Yopt-Y0))
    expect_lt(error_y,0.1)
})


test_that("Euclidean Trust Region",{
    optObj$select_algorithm("tr")
    expect_error(optObj$select_algorithm("dd"))
    optObj$initial_point(Y_init)
    optObj$solve()
    Yopt = optObj$get_optimizer()
    error_y = sum(abs(Yopt-Y0))
    expect_lt(error_y,0.1)
})


