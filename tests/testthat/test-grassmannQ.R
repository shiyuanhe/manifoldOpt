set.seed(120)
n = 20
p = 2
A = matrix(runif(n^2), n, n)
A = A + t(A)


objf = function(X){
  -sum(diag(t(X)%*%A%*%X))
}
#set gradient function
gradf = function(X){
  -2*A%*%X
}
#set hessian function
hessf = function(X,Z){
  -2*A%*%Z
}




test_that("Find Dominant Subspace",{
  
  problem = new(manifoldOpt)
  problem$set_grassmannQ(n, p)
  problem$select_algorithm("tr")
  problem$update_control(list(tol = 1e-6*sqrt(n*p)))
  problem$setObjective(objf)
  problem$setGradient(gradf)
  problem$setHessian(hessf)
  problem$solve()
  
  spHat = problem$get_optimizer()
  spTrue = eigen(A)$vectors[,1:2]
  v1 = -objf(spHat)
  v2 = sum(eigen(A)$values[1:2])
  expect_lt(abs(v1-v2), 1e-2)
  
  spS = t(spHat)%*%spTrue
  expect_gt(sum(svd(spS)$d),1.9)
})