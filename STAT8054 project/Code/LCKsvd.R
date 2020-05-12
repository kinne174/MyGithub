#STAT 8054 Project
#implementation of LC.Ksvd2, paper by Jiang et al

#sourced functions
source("C:/Users/Mitch/Documents/UofM/Spring 2017/Stat 8054/Project/Code/LCKsvd-init.R")
source("C:/Users/Mitch/Documents/UofM/Spring 2017/Stat 8054/Project/Code/Ksvd.R")

# #libraries
# library(beepr)
# 
# #trying it out
# set.seed(8054)
# n <- 7; N <- 50; m <- 3
# D.size <- 11; Tol <- 4
# out.response <- createResponse(n, N, m)
# Y <- out.response$Y
# H <- out.response$H
# 
# alpha <- 2; beta <- 1.5
# 
# a <- My.LCKsvd(Y, H, D.size, Tol, alpha, beta, "OMP", 1e2, TRUE)
# sum(apply(H,2,whichmax)==apply(a$W%*%a$X,2,whichmax))/N


My.LCKsvd <- function(Y, H, D.size, Tol, alpha, beta, pursuit, ksvd.iters, classification){
  #function to implement label consistent KSVD from paper
  #Input:
  #              Y: n x N matrix, response matrix
  #              H: m x N true class label matrix of response Y
  #         D.size: preference for K the number of columns in the dictionary
  #            Tol: tolerance or max number of non zero elements allowed in each x vector
  #          alpha: penalty term for dictionary/response class consistency
  #           beta: penalty term for classification consistency
  #        pursuit: which pursuit algorithm to use in updating X, currently can choose "MP", "OMP", or "F"
  #     ksvd.iters: how many times to iterate through ksvd
  # classification: return D,A and W to be ready for classification (according to paper)
  #Output:
  # D: n x K matrix, dictionary
  # A: K x K matrix, linear transformation of X for dictionary/response consistency
  # W: m x K matrix, classifier matrix
  # X: K x N matrix, sparse linear combination matrix
  # Q: K x N matrix, dictionary/responses classes
  
  #initialize dimensions
  K <- D.size
  n <- nrow(Y)
  N <- ncol(Y)
  m <- nrow(H)
  
  #get initialization of dictionary, A and W
  out.init <- LCKsvd.init(Y, H, D.size, Tol, lambda1 = 1, lambda2 = 1)
  D.init <- out.init$D.init
  A.init <- out.init$A.init
  W.init <- out.init$W.init
  Q <- out.init$Q
  
  #Definie Y.new and D.new according to the paper
  Y.new <- rbind(Y,sqrt(alpha)*Q,sqrt(beta)*H)
  D.new <- rbind(D.init, sqrt(alpha)*A.init, sqrt(beta)*W.init)
  
  #nomalize columns of D.new
  D.new <- t(t(D.new)/apply(D.new,2,norm,type = "2"))
  
  #run ksvd on Y.new and D.new
  out.ksvd <- My.Ksvd(Y.new, D.new, pursuit, Tol, max.iters = ksvd.iters)
  
  #reassign matrices
  D <- out.ksvd$D[1:n,]
  A <- out.ksvd$D[(n+1):(n + K),]
  W <- out.ksvd$D[(n + K + 1):(n + K + m),]
  
  #assign x matrix
  X <- out.ksvd$X
  
  #for classification purposes un normalize matrices according to paper
  if(classification){
    D.norms <- get.norms(D)
    D <- D*matrix(D.norms^-1, nrow = nrow(D), ncol = ncol(D), byrow = T)
    A <- A*matrix(D.norms^-1, nrow = nrow(A), ncol = ncol(A), byrow = T)
    W <- W*matrix(D.norms^-1, nrow = nrow(W), ncol = ncol(W), byrow = T)
  }
  
  return(list(D = D, A = A, W = W, X = X, Q = Q))
  
  # #for testing purposes
  # D.new <- rbind(D, A, W)
  # 
  # norm(Y - D%*%X, "F")
  # norm(Q - A%*%X, "F")
  # norm(H - W%*%X, "F")
  # sum(apply(H,2,whichmax) == apply(W%*%X,2,whichmax))
}
