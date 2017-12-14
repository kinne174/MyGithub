#STAT 8054 Project
#implementation of FOCUSS for an additional option in pursuit
#based on slides from ...

FOCUSS.update <- function(D, y.now){
  #function to return the x vector for consideration of adding to returned X matrix based on D and column vector y.now
  #Input:
  #     D: n x K matrix, dictionary input
  # y.now: n x 1 vector, a single response vector
  #Output:
  # x.vec: K x 1 vector, sparse linear combiner vector
  
  #initialize max.iters and error
  max.iters <- 1e2
  error <- 1e-4
  
  #initialize approximate "prev" vector and weight matrix
  x.prev <- 1
  X.weight <- diag(ncol(D))
  
  #initialize iterating and increment counter
  iterating <- TRUE
  k <- 0
  
  while(iterating){
    k <- k + 1
    
    #assign candidate x.vec according to algorithm, ginv is moore penrole gen inverse
    x.vec <- (X.weight^2)%*%t(D)%*%ginv(D%*%(X.weight^2)%*%t(D))%*%cbind(y.now)
    colnames(x.vec) <- NULL
    
    #assign new weight matrix according to algorithm
    X.weight <- diag(abs(as.vector(x.vec))^(1-.5))
    
    #testing
    #norm(x.vec - x.prev, "2")
    
    #decide to exit, stopping rules were suggested
    if(norm(x.vec - x.prev, "2") < error || k >= max.iters){
      iterating <- FALSE
    }
    
    #update comparison variable
    x.prev <- x.vec
    
  }
  
  return(x.vec)
}

My.FOCUSS <- function(D, Y, Tol){
  #function to compute the sparse linear combiner matrix X, a pursuit algorithm
  #Input:
  # D: n x K matrix, dictionary
  # Y: n x N matrix, response matrix
  # Tol: tolerance scalar, max number of non zero elements allowed in each column of X
  #Output:
  # X.mat: K x N matrix, sparse linear combiner matrix
  
  #need for ginv
  library(MASS)
  
  #initialize parameters
  N <- ncol(Y)
  K <- ncol(D)
  
  #initialize returned X matrix
  X.mat <- matrix(0,ncol = N, nrow = K)
  
  #do for each column of response matrix Y
  for(i in 1:N){
    
    #get column of Y
    y.now <- Y[,i]
    
    #get initial column of X[,i]
    x.vec <- FOCUSS.update(D, y.now)
    
    #check to make sure Tol is satisfied
    if(sum(x.vec != 0) >= Tol){
      #see which columns of D contribute the most to mapping D to Y
      #only want Tol number of them too
      important.indices <- order(abs(x.vec))[(length(x.vec) - Tol + 1):length(x.vec)]
      
      #replace unimportant columns in dictionary with zeroes so they cannot be selected
      D.new <- D
      D.new[,-important.indices] <- 0
      
      #get new column with new dictionary, will now only return Tol number of non zero elements
      x.vec <- FOCUSS.update(D.new, y.now)
    }
    
    #assign column of X[,i]
    X.mat[,i] <- x.vec
  }
  
  return(X.mat)
}
