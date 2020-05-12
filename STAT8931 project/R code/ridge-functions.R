#Adam Rothman's ridge-functions.R code

ridgecov=function(S, lam)
{
  p=dim(S)[1]
  e.out=eigen(S, symmetric=TRUE)
  new.evs=(-e.out$val + sqrt( e.out$val^2 + 4*lam))/(2*lam)
  omega.hat=tcrossprod(e.out$vec*rep(new.evs, each=p),e.out$vec)
  return(omega.hat)
}

ridgecov.cv=function(x, lam.vec, ind=NULL, kfold=5, quiet = TRUE)
{
  ## x is the n row by p column matrix where the rows are a realization
  ##   of n independent copies of a p-variate random vector
  ## ind is a vector of a permutation of 1,...,n
  ## kfold is the number of folds to use
  
  n=dim(x)[1]
  p=dim(x)[2]
  
  ## if the user did not specify a permutation of 1,..,n, then randomly
  ## permute the sequence:
  if(is.null(ind)) ind=sample(n);
  
  ## allocate the memory for the loss matrix 
  ## (rows correspond to values of the tuning paramter)
  ## (columns correspond to folds)
  cv.loss = array(0, c(length(lam.vec), kfold))
  for (k in 1:kfold)
  {
    leave.out=ind[ (1+floor((k-1)*n/kfold)):floor(k*n/kfold) ]
    x.tr=x[-leave.out,,drop=FALSE]
    meanx=apply(x.tr, 2, mean)
    x.tr=scale(x.tr, center=meanx, scale=FALSE)
    x.va=x[leave.out,,drop=FALSE]
    x.va=scale(x.va, center=meanx, scale=FALSE)
    s.tr=crossprod(x.tr)/(dim(x.tr)[1])
    s.va=crossprod(x.va)/(dim(x.va)[1])
    
    ## compute the spectral decomposition of s.tr
    e.out=eigen(s.tr, symmetric=TRUE)
    
    for(i in 1:length(lam.vec))
    {
      lam=lam.vec[i]
      
      ## compute the ridge-penalized likelihood precision matrix
      ## estimator at the ith value in lam.vec:
      new.evs=(-e.out$val + sqrt( e.out$val^2 + 4*lam))/(2*lam)
      omega.hat=tcrossprod(e.out$vec*rep(new.evs, each=p),e.out$vec)
      
      ## compute the observed negative validation loglikelihood value
      cv.loss[i,k] = sum(omega.hat*s.va) - determinant(omega.hat, logarithm=TRUE)$modulus[1]
      
      if(!quiet) cat("Finished lam =", lam.vec[i], "in fold", k, "\n") 
    }
    if(!quiet) cat("Finished fold", k, "in ridgecov\n")       
  }  
  
  ## accumulate the error over the folds
  cv.err=apply(cv.loss, 1, sum)
  
  ## find the best tuning parameter value
  best.lam = lam.vec[which.min(cv.err)]
  
  ## compute final estimate at the best tuning parameter value
  samp.cov=cov(x)*((n-1)/n)  
  omega.hat=ridgecov(S=samp.cov, lam=best.lam)
  
  return(list(omega.hat=omega.hat, best.lam=best.lam, cv.err=cv.err, lam.vec=lam.vec)) 
}