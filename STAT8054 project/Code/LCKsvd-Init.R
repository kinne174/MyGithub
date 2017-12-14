#STAT 8054
#initializaing parameters for LCKSVD 1&2

#to initialize dictionary first pool all common Ys and permute and then make the dictionary for that class, repeat for all
#classes and bind dictionarys and X matrices together
#to initialize Q, a K x N matrix, match the class of Ys (rows) to the dictionary columns (cols), should look like block matrices of 1's
#depending on how many Ys are in each class, not necessarily a square matrix though, # of dictionary columns 
#corresponds to # of Ys in class
#H is a m x N matrix where m is the number of classes, is the true labels for training set, H[i,j] is 1 is the jth sample
#of Y is a member of class i

#if number of samples from Y in class i is smaller than D.size/m then return error


#sourced functions
source("C:/Users/Mitch/Documents/UofM/Spring 2017/Stat 8054/Project/Code/MatchingPursuit.R")
source("C:/Users/Mitch/Documents/UofM/Spring 2017/Stat 8054/Project/Code/Ksvd.R")

# #trying it out
# set.seed(8054)
# n <- 20; N <- 500; D.size <- 50; m <- 5; Tol <- 3
# out <- createResponse(n,N,m)
# Y <- out$Y
# H <- out$H
# 
# lambda1 <- 2; lambda2 <- 1.5
# out.init <- LCKsvd.init(Y, H, D.size, Tol, lambda1, lambda2)
# system.time(LCKsvd.init(Y, H, D.size, Tol, lambda1, lambda2))

createResponse <- function(n, N, m){
  #function to create Y and H to be used in testing, pretty simple response since Y is just iid normal and it has
  #m different means, m is as evenly distributed as possible
  #Input:
  # n: length of observations
  # N: number of observations total
  # m: number of distinct classes in samples
  #Output:
  # Y: a n x N matrix of observations
  # H: a m x N matrix that tracks class of Y, H[i,j] is 1 if jth sample of Y is in class i
  
  #initialize empty Y.mat and H matrices
  Y.mat <- matrix(NA, ncol = N, nrow = n)
  H <- matrix(0, nrow = m, ncol = N)
  
  #determine how many extra samples of Y is needed
  leftover <- N - m*floor(N/m)
  
  #for loop to set columns of Y for each class and update H with which columns of Y are in the class
  for(i in 1:m){
    Y.mat[,(1 + (i-1)*floor(N/m)):(i*floor(N/m))] <- 
                                      matrix(rnorm(n*floor(N/m), mean = seq(1,3*m,length.out = m)[i]),nrow = n)
    H[i,(1 + (i-1)*floor(N/m)):(i*floor(N/m))] <- 1
  }
  
  #if the samples cannot be evenly distributed add on some extras starting with class 1
  if(leftover > 0){
    for(j in 1:leftover){
      Y.mat[,j + m*floor(N/m)] <- rnorm(n,mean = c(1:m)[j])
      H[j,j + m*floor(N/m)] <- 1
    }
  }
  
  return(list(Y = Y.mat, H = H))
}

LCKsvd.init <- function(Y, H, D.size, Tol, lambda1, lambda2){
  #function to initialize the dictionary, A & W matrices and also the Q matrix, uses random fuzzed samples of Y from each 
  #class to build a class dictionary and then column binds them all together to form the initial dictionary
  #Input:
  #       Y: n x N matrix of responses
  #       H: m x N true class label matrix of response Y
  #  D.size: preference for K the number of columns in the dictionary
  #     Tol: tolerance or max number of non zero elements allowed in each x vector
  # lambda1: penalty for W matrix in ridge regression style initiaion
  # lambda2: penalty for A matrix in ridge regression style initiaion
  #Output:
  # D.init: n x K matrix that is the dictionary initialization
  # A.init: K x K matrix that is the A matrix initialization
  # W.init: m x K matrix that is the W matrix initialization
  #      Q: K x N matrix that is the matrix to know what columns of Y contributed to creating which columns of D.init
  
  #dictionary columns will come evenly as possible from the classes of Y
  #if unable to evenly do this will have leftover > 0
  dict.elements <- floor(D.size/nrow(H))
  leftover <- D.size - nrow(H)*dict.elements
  
  #initialize D and Q
  D.init <- matrix(ncol = 0, nrow = nrow(Y))
  Q <- matrix(nrow = 0, ncol = ncol(Y))
  
  for(i in 1:nrow(H)){
    #make sure there are enough samples in each class to draw from, shouldn't be a problem as N >> K (D.size)
    if(length(which(H[i,] == 1, arr.ind = T)) < dict.elements){
      return(cat("There was an error in class ",i,". Not enough samples. Need atleast ", dict.elements, sep = ""))
    }
    #if this is a class which gets an extra column in D because of unable to evenly distribute enter if,
    #otherwise if evenly distributed go to else
    if(i <= leftover){
      #only notable differnce between if and else is the "dict.elements + 1" since some classes will contribute to 
      #one more column in D than others
      
      #randomly get columns of Y in class i to fuzz and use in dictionary
      indices <- sample(which(H[i,] == 1, arr.ind = T), dict.elements + 1)
      D.part <- Y[,indices] + matrix(rnorm((dict.elements+1)*nrow(Y)), nrow = nrow(Y))
      D.part <- t(t(D.part)/apply(D.part,2,norm,type = "2"))
      
      #need to keep track of which columns of Y are contributing to which columns of D
      Q.start <- matrix(0,nrow = dict.elements + 1, ncol = ncol(Y))
      Q.start[,which(H[i,] == 1,arr.ind = T)] <- 1
      
    }else{
      #randomly get columns of Y in class i to fuzz and use in dictionary
      indices <- sample(which(H[i,] == 1, arr.ind = T), dict.elements)
      D.part <- Y[,indices] + matrix(rnorm(dict.elements*nrow(Y)), nrow = nrow(Y))
      D.part <- t(t(D.part)/apply(D.part,2,norm,type = "2"))
      
      #need to keep track of which columns of Y are contributing to which columns of D
      Q.start <- matrix(0,nrow = dict.elements, ncol = ncol(Y))
      Q.start[,which(H[i,] == 1,arr.ind = T)] <- 1
    }
    
    #update D.init, first need to normalize the chosen columns of Y
    input.Ys <- Y[,which(H[i,] == 1, arr.ind = T)]
    input.Ys <- t(t(input.Ys)/apply(input.Ys,2,norm,type = "2"))
    
    out.ksvd <- My.Ksvd(input.Ys,D.part, pursuit = "OMP",Tol, max.iters = 5)
    D.init <- cbind(D.init, out.ksvd$D)
    
    #update Q
    Q <- rbind(Q, Q.start)
  }
  
  #Get sparse coding matrix X
  X <- My.matching.pursuit(D.init, Y, Tol, orthogonal = T)
  
  #Initialize A and W matrices based on Ridge Regression
  A.init <- t(qr.solve(crossprod(t(X)) + lambda2*diag(ncol(D.init)),X%*%t(Q)))
  
  W.init <- t(qr.solve(crossprod(t(X)) + lambda1*diag(ncol(D.init)),X%*%t(H)))
  
  return(list(D.init = D.init,A.init = A.init, W.init = W.init, Q = Q))
}


