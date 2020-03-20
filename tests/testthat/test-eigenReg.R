
context("eigenReg Manifold~")
set.seed(100)
rm(list=ls())
library(rOptManifold)

n = 60;  r = 3
U=matrix(runif(n*r)*10,ncol=r)
U = qr.Q(qr(U))
sigma=diag(c(100, 60, 40))
lowR = U%*%sigma%*%t(U)
errMat =  matrix(rnorm(n^2, sd = 0.1), n, n)
lowRObs = lowR + (errMat + t(errMat)) / 2


obj = function(X){
  lowRHat = X[[1]] %*% X[[2]] %*% t(X[[1]])
  temp = (lowRObs-lowRHat)^2
  val = 0.5 * sum(temp)
  return(val)
}

grad = function(X){
  lowRHat = X[[1]] %*% X[[2]] %*% t(X[[1]])
  grad0 = -lowRObs + lowRHat
  grad2 = t(X[[1]]) %*% grad0 %*% X[[1]]
  grad1 =  grad0 %*% X[[1]] %*% X[[2]]
  return(list(grad1, grad2))
}



# gradInit = eigen(lowRObs)
# XInit = list(gradInit$vectors[, 1:r], diag(gradInit$values[1:r]))


problem = new(manifoldOpt)
problem$set_eigenReg(n,r)
problem$select_algorithm("cg")
problem$update_control(list(tol = 1e-6*sqrt(n*n), alpha = 2, iterMax = 1e4, sigma = 0.2))
problem$setObjective(obj)
problem$setGradient(grad)
# problem$initial_point(XInit)
problem$solve()
temp = problem$get_optimizer()
Xhat = temp[[1]] %*% temp[[2]] %*% t(temp[[1]])
summary(as.vector(abs(Xhat - lowR)) )

obj(temp)

library(ggplot2)
library(reshape2)
plotD = abs(Xhat - lowR)
plotD2 = melt(plotD)
p = ggplot(plotD2, aes(Var1, Var2, fill = value))
p + geom_tile() 

