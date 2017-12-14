#Adam Rothman's discriminant-analysis-functions.R code

## the following function fits the QDA model 
## Arguments:
##  X is an n by p matrix (the ith row is the values
##    of the predictor for the ith case).
##  y is the n entry response vector (the ith entry
##    is the response category in {1,...,C} for 
##    the ith case)
##  the function returns a list with the parameter 
##  estimates (see the code)
fit.qda=function(X, y, method=c("MLE", "diagonal", "ridge"), lam.vec=NULL, ...)
{
  method=match.arg(method)
  ## y has values in {1,...,C}
  C=max(y)
  n=length(y)
  
  pi.hats=numeric(C)
  mu.hats=list()
  Sigma.inv.hats=list()
  
  picked.ridge=numeric(C)
  
  for(k in 1:C)
  {
    indices=which(y==k)
    nk=length(indices)
    pi.hats[k]=nk/n
    theseXs=X[indices,, drop=FALSE]
    mu.hats[[k]]=apply(theseXs, 2, mean)
    
    if(method=="MLE")
    {
      S=cov(theseXs)*((nk-1)/nk)
      Sigma.inv.hats[[k]]=qr.solve(S)
    } else if (method=="diagonal")
    {
      S=cov(theseXs)*((nk-1)/nk)
      Sigma.inv.hats[[k]]=diag(1/diag(S))
    } else 
    {
      fit=ridgecov.cv(x=theseXs, lam.vec=lam.vec, ...)
      picked.ridge[k] = fit$best.lam 
      Sigma.inv.hats[[k]]=fit$omega.hat
    }
  }
  return(list(pi.hats=pi.hats, mu.hats=mu.hats, Sigma.inv.hats=Sigma.inv.hats, picked.ridge=picked.ridge))
}

## the following function classifies test data 
## using a fitted QDA model
## Arguments:
##  fit, this is a list with elements pi.hats, mu.hats, and Sigma.hats
##      where pi.hats is a list of C response category sample proportions
##            mu.hats is a list of C p-dimensional sample mean vectors
##            Sigma.hats is a list of C p by p Sample covariance matrices
##  Xtest, this is a matrix with ntest rows and p column,
##         each row is a test case
##  The function returns a vector of ntest entries, where
##  the ith entry is the estimated response category (some
##  value in {1,...,C} ) for the ith test case.
classify.qda = function(fit, Xtest)
{
  ntest=nrow(Xtest)
  C=length(fit$pi.hats)
  score.mat=matrix(NA, nrow=ntest, ncol=C)
  for(k in 1:C)
  {
    ## compute all ntest discriminant scores for category k 
    tec=scale(Xtest, center=fit$mu.hats[[k]], scale=FALSE)
    eout=eigen(fit$Sigma.inv.hats[[k]], symmetric=TRUE, only.values=TRUE)
    ld=sum(log(eout$values))
    score.mat[,k]=0.5*ld-0.5*diag(tec%*%fit$Sigma.inv.hats[[k]]%*%t(tec)) + log(fit$pi.hats[k])
  }
  ## determine the best category for each of the ntest cases
  pred.classes=apply(score.mat, 1, which.max)
  return(pred.classes)
}

## the following function fits the LDA model
## Arguments:
##  X is an n by p matrix (the ith row is the values
##    of the predictor for the ith case).
##  y is the n entry response vector (the ith entry
##    is the response category in {1,...,C} for 
##    the ith case)
##  the function returns a list with the parameter 
##  estimates (see the code)
fit.lda=function(X, y, method=c("MLE", "diagonal", "ridge"), lam.vec=NULL, ...)
{
  method=match.arg(method)
  ## y has values in {1,...,C}
  C=max(y)
  n=length(y)
  
  pi.hats=numeric(C)
  mu.hats=list()
  Sigma.inv.hats=list()
  
  picked.ridge = c()
  
  centeredX=NULL
  for(k in 1:C)
  {
    indices=which(y==k)
    nk=length(indices)
    pi.hats[k]=nk/n
    theseXs=X[indices,, drop=FALSE]
    mu.hats[[k]]=apply(theseXs, 2, mean)
    centeredX=rbind(centeredX, scale(theseXs, center=mu.hats[[k]], scale=FALSE))
  }
  if(method=="MLE")
  {
    S=crossprod(centeredX)/n
    Sigma.inv.hat=qr.solve(S)
  } else if ( method=="diagonal" )
  {
    S=crossprod(centeredX)/n
    Sigma.inv.hat=diag(1/diag(S))
  } else
  {
    fit=ridgecov.cv(x=centeredX, lam.vec=lam.vec, ...)
    picked.ridge = fit$best.lam 
    Sigma.inv.hat=fit$omega.hat
  }
  
  for(k in 1:C)
  {
    Sigma.inv.hats[[k]]=Sigma.inv.hat
  }
  
  return(list(pi.hats=pi.hats, mu.hats=mu.hats, Sigma.inv.hats=Sigma.inv.hats, picked.ridge=picked.ridge))
}