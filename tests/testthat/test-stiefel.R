 set.seed(110)
n = 100
p = 2
A = matrix(runif(n^2), n, n)
A = A+t(A)
N = diag(c(10,5))

objf = function(X){
  -sum(diag(t(X)%*%A%*%X%*%N))
}

#set gradient function
gradf = function(X){
  -2*A%*%X%*%N
}

#set hessian function
hessf = function(X,Z){
  -2*A%*%Z%*%N
}


test_that("Stiefel Steepest Descent",{
  problem = new(manifoldOpt)
  problem$set_stiefel(n, p)
  problem$select_algorithm("sd")
  problem$update_control(list(tol = 1e-7, iterMax = 2000))
  problem$setObjective(objf)
  problem$setGradient(gradf)
  problem$setHessian(hessf)
  problem$solve()
  
  ev = problem$get_optimizer()
  ev0 = eigen(A)$vectors[,1:2]
  
  if(ev[1,1]*ev0[1,1]<0)
    ev[,1] = -ev[,1]
  if(ev[1,2]*ev0[1,2]<0)
    ev[,2] = -ev[,2]
  err = sum(abs(ev - ev0))
  expect_lt(err, 5e-2)
})


test_that("Stiefel Trust Region",{
  problem = new(manifoldOpt)
  problem$set_stiefel(n, p)
  problem$select_algorithm("tr")
  problem$update_control(list(tol = 1e-6*sqrt(n*p)))
  problem$setObjective(objf)
  problem$setGradient(gradf)
  problem$setHessian(hessf)
  problem$solve()
  
  ev = problem$get_optimizer()
  ev0 = eigen(A)$vectors[,1:2]
  
  if(ev[1,1]*ev0[1,1]<0)
    ev[,1] = -ev[,1]
  if(ev[1,2]*ev0[1,2]<0)
    ev[,2] = -ev[,2]
  err = sum(abs(ev - ev0))
  expect_lt(err, 5e-2)
})
