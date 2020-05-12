#Adam Rothman's ridge-test.R code

setwd("C:/Users/Mitch/Documents/UofM/Fall 2017/STAT 8931/Project")
source("ridge-functions.r")
set.seed(8931)
n=10000
p=500
reps=10

## without loss of generality, set the mean
## to zero
mu=rep(0,p)

## create the population covariance matrix
Sigma=matrix(NA, nrow=p, ncol=p)
for(j in 1:p) for(k in 1:p)
  Sigma[j,k]=0.4*(j!=k) + 1*(j==k)

## compute the square root of Sigma
eout=eigen(Sigma, symmetric=TRUE)
Sigma.sqrt=eout$vectors%*%diag(eout$values^0.5)%*%t(eout$vectors)
Sigma.inv = eout$vectors%*%diag(eout$values^(-1))%*%t(eout$vectors)

## candidate tuning parameter values  
lam.vec=10^seq(from=-8, to=8, by=0.25)

## allocate memory for squared-Frobenius norm losses,
## one realized loss for each replication
loss.samp=numeric(reps)
loss.ridge=numeric(reps)

## allocate the memory for a vector to keep track
## of the selected tuning parameter for the ridge method:
picked.ridge=numeric(reps)

for(r in 1:reps)
{  
  ## generate the n by p matrix X with rows
  ## drawn iid N_p(mu, Sigma)
  Z = matrix(rnorm(n*p), nrow=n, ncol=p)
  X = rep(1,n)%*%t(mu) + Z %*% Sigma.sqrt
  
  ## MLE sample covariance matrix
  S.inv=qr.solve( cov(X)*((n-1)/n) )
  loss.samp[r]=sum( (S.inv - Sigma.inv)^2 )
  
  ## Ridge-penalized likelihood precision matrix 
  fit=ridgecov.cv(x=X, lam.vec=lam.vec)
  picked.ridge[r]=fit$best.lam
  loss.ridge[r]=sum( (fit$omega.hat - Sigma.inv)^2 )
  
  cat("Finished replication r=", r, "\n")
}

## make a simulation-based 99% approximate CI for the
## squared-Frobenius norm risk difference: 
## E\| S.inv - Sigma.inv\|^2  - E\| Omega.hat.lambda.hat - Sigma.inv\|^2
t.test(loss.samp-loss.ridge, conf.level=0.99)$conf.int