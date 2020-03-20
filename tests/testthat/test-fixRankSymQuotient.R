set.seed(120)
nn=50
rr=3
A=matrix(runif(nn*rr),nn,rr)
A=A%*%t(A)

#add noise
B=A+matrix(rnorm(nn*nn,sd=0.1),nn,nn)
objF = function(X){
  val = sum((B-X)^2)
  return(val)
}
gradF = function(X){
  2*(X-B)
}
hessF = function(X,Z){
  2*Z
}



test_that("Quotient PSD",{
problem = new(manifoldOpt)
problem$set_fixRankPSD(nn, rr)
problem$select_algorithm("tr")
problem$update_control(list(tol = 1e-5*sqrt(nn*nn)))
problem$setObjective(objF)
problem$setGradient(gradF)
problem$setHessian(hessF)
problem$solve()
temp = problem$get_optimizer() - A
recovery_err = sum(temp^2)
recovery_err = sqrt(recovery_err / nn^2)
expect_lt(recovery_err, 0.05)
})

