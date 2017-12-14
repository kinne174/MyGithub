#STAT 8054 Project
#attempt at KSVD algorithm

#Recall Y is a n x N matrix with N samples of vectors of length n
#       D is the dictionary and is a n x K matrix with K "classifications" to do linear combinations of
#       X is the linear combination matrix to determine how to fit together columns of D to represent Y, a K x N matrix


#sourced functions
source("C:/Users/Mitch/Documents/UofM/Spring 2017/Stat 8054/Project/Code/MatchingPursuit.R")
source("C:/Users/Mitch/Documents/UofM/Spring 2017/Stat 8054/Project/Code/FOCUSS.R")

# #trying it out
# set.seed(8054)
# n <- 40; N <- 500; K <- 50; Tol <- 3
# #randomly set Y values, should be a n x N matrix
# #Y <- matrix(c(rnorm(n*N/4),rnorm(mean = 3,n*N/4),rnorm(mean = -3,n*N/4),rnorm(mean = 10,n*N/4)),nrow = n, byrow = F)
# Y <- matrix(rnorm(n*N), nrow = n)
# #initialize C, C should be a n x K matrix
# D <- matrix(rnorm(n*K), ncol = K, nrow = n)
# D <- t(t(D)/apply(D,2,norm,type = "2"))
# 
# X <- My.matching.pursuit(D,Y,Tol,orthogonal = T)
# 
# system.time(invisible(replicate(10,Ksvd.update(D,Y,X))))
# system.time(invisible(replicate(10,My.matching.pursuit(D,Y,2,T))))
# system.time(invisible(replicate(10,My.FOCUSS(D,Y,2))))
# out <- My.Ksvd(Y,D,"OMP",Tol)
# plot(1:length(out$norms), out$norms, main = "Norms for n = 40", xlab = "iteration", ylab = "Frobenius norm")

Ksvd.update <- function(D,Y,X){
  #second part of Ksvd function, update D and parts of X matrix
  #Input:
  # D: n x K matrix, current dictionary
  # Y: n x N matrix, observation matrix
  # X: K x N matrix, current sparse linear combiner matrix
  #Ouput:
  # X: K x N matrix, new sparse linear combiner matrix
  # D: n x K matrix, new dictionary
  
  #could do parallel on this
  for(i in 1:ncol(D)){
    
    #check to see if any changes can even be made
    if(length(which(X[i,] != 0,arr.ind = T))>0){
      
      #compute error matrix, select only columns which are being used by ith col of D, and do SVD
      E <- Y - (D%*%X - cbind(D[,i])%*%X[i,])
      E.R <- E[,which(X[i,] != 0,arr.ind = T)]
      E.svd <- svd(E.R)
      
      #update column i of D
      D[,i] <- E.svd$u[,1]
      
      #update row i of X
      x.new <- E.svd$d[1] * E.svd$v[,1]
      for(j in 1:length(which(X[i,] != 0, arr.ind = T))){
        X[i,which(X[i,] != 0, arr.ind = T)[j]] <- x.new[j]
      }
    }
  }
  return(list(X = X, D = D))
}

My.Ksvd <- function(Y,init.D = NULL, pursuit, Tol, max.iters = 1e2){
  #function based on Aharon et al to do Ksvd algorithm
  #Input:
  # Y: n x N matrix, response matrix
  # init.D: n x K matrix, must provide initial dictonary as there is currently no initialization function
  # pursuit: which pursuit algorithm to use in updating X, currently can choose "MP", "OMP", or "F"
  # Tol: maximum number of non-zero elements in each vector of X
  # max.iters: maximum number of iterations
  #Output:
  # D: n x K matrix, overcomplete dictionary
  # x: K x N matrix, sparse linear combination matrix
  
  #make sure an inital D is provided
  if(is.null(init.D)){
    return("Please provide an initial D")
  }else{
    D <- init.D
  }
  
  #set preferred pursuit algorithm
  if(pursuit == "OMP"){
    orthogonal = TRUE
    MP = TRUE
  }
  if(pursuit == "MP"){
    orthogonal = FALSE
    MP = TRUE
  }
  if(pursuit == "F"){
    MP = FALSE
  }
  
  #initialize while loop and iteration counter and norm.prev
  iterating <- TRUE
  k <- 1
  norm.prev <- 0
  norm.save <- rep(NA,max.iters)
  
  #do until convergence
  while(iterating){
    #do a pursuit to find best X given Dictionary, Y
    if(MP){
      X.pursuit <- My.matching.pursuit(D,Y,Tol,orthogonal)
    }else{
      X.pursuit <- My.FOCUSS(D,Y,Tol)
    }

    #do update of dictionary D and rows of X by SVD
    out <- Ksvd.update(D,Y,X.pursuit)
    
    X.svd <- out$X
    D <- out$D
    
    #choose X based on greatest reduction in MSE
    use.svd <- norm(Y - D%*%X.svd, "F") <= norm(Y - D%*%X.pursuit, "F")
    if(use.svd){X <- X.svd}else{X <- X.pursuit}
    
    #check convergence
    if(norm(Y - D%*%X, type = "F") < 1e-6 || abs(norm(Y - D%*%X, type = "F") - norm.prev) < 1e-10  || k >= max.iters){
      iterating <- FALSE
    }
    
    #save norm and update iterator
    norm.prev <- norm(Y - D%*%X, type = "F")
    
    #save norms for plot proving MSE is reducing
    norm.save[k] <- norm(Y - D%*%X, type = "F")
    
    k <- k + 1
  }
  
  return(list(D = D, X = X, norms = norm.save[1:(k-1)]))
}

whichmax <- function(x){
  return(which(max(x)==x,arr.ind = T))
}

get.norms <- function(M){
  return(apply(M,2,norm,type = "2"))
}

