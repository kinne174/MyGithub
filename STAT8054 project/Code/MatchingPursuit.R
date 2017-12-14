#STAT 8054 project
#Matching pursuit/ orthogonal matching pursuit

#Make sure when supplying D to My.matching.pursuit its columns are L2 normalized

# #trying it out
# set.seed(8054)
# n <- 20; N <- 500; K <- 50
# #randomly set Y values, should be a n x N matrix
# Y <- matrix(rnorm(n*N), nrow = n)
# #initialize D, CDshould be a n x K matrix
# D <- matrix(rnorm(n*K), ncol = K, nrow = n)
# #L2 normalize columns of D
# D <- t(t(D)/apply(D,2,norm,type = "2"))
# 
# system.time(invisible(replicate(500,My.matching.pursuit(D,Y,Tol = 5,orthogonal = F))))
# system.time(invisible(replicate(500,get.linear.combo(D,Y[,1],orthogonal = T,Tol = 5))))
# 
# X <- My.matching.pursuit(D,Y,Tol = 5,orthogonal = F)
# norm(Y - D%*%X,"F")

#takes abs inner product of a vector
inner.product.abs <- function(x){
  return(abs(sum(x)))
}

#function to find which column of D maximizes the abs inner product with the current residual
find.atom <- function(resid, D){
  inner.prods <- apply(D*resid,2,inner.product.abs)
  return(which(max(inner.prods)==inner.prods,arr.ind = T)[1])
}

get.linear.combo <- function(D, y.now, orthogonal, Tol){
  #function to get the x vector that is a linear combination of the columns of D for column Y[,y]
  #Input:
  #          D: dictionary
  #      y.now: current column of Y
  # orthogonal: T/F if should use orthogonal pursuit mapping
  #Output:
  # x.vec: vector of length K that goes in X matrix
  
  #max iterations
  max.it <- 1e3
  k <- 0
  
  #initialize residual and approximation vector to enter while loop
  a.next <- rep(0,length(y.now))
  r.next <- y.now
  a.now <- rep(100,length(y.now))
  
  #initialize zero x column to return for MP
  x <- rep(0, ncol(D))
  
  #initialize empty lambda set
  lambda.set <- c()
  
  #while loop to get best x vector
  #exit when residual is small or not changing or if its looped too long
  while(norm(r.next, "2") > 1e-6 && norm(a.now - a.next, "2") > 1e-10 && k < max.it){
    #&& norm(r.now - r.next, "2") > 1e-10
    
    #update residual and appoximation
    r.now <- r.next
    a.now <- a.next
    
    #find.atom returns column number of D that maximizes the absolute innerproduct of r.now and the current col of D
    if(length(lambda.set) < Tol){
      lambda <- find.atom(r.now, D)
    }else{
      lambda <- find.atom(r.now,D[,sort(lambda.set)])
      lambda <- sort(lambda.set)[2]
    }
    #lambda <- find.atom(r.now,D)
    
    #if lambda is not in lambda set then add it
    if(!(lambda %in% lambda.set)){
      lambda.set <- c(lambda.set,lambda)
    }
    
    if(orthogonal){
      #get least squares of estimator for next approximation
      beta <- qr.solve(crossprod(D[,sort(lambda.set)]),t(D[,sort(lambda.set)])%*%cbind(y.now))
      a.next <- as.vector(D[,sort(lambda.set)]%*%beta)
      
      x <- beta
    }else{
      #regular mapping pursuit based off of p.7 in Tropp
      a.next <- a.now + sum(r.now*D[,lambda])*D[,lambda]
      
      #update x[lambda] with how much the vector contributes
      x[lambda] <- x[lambda] + sum(r.now*D[,lambda])
    }
    #get next residual term
    r.next <- y.now - a.next
    
    #increment k
    k <- k + 1
    
    #check to see if norm is being reduced every cycle
    #print(norm(r.now, "2"))
  }
  return(list(x = x, lambda.set = lambda.set))
}


My.matching.pursuit <- function(D, Y, Tol, orthogonal = T){
  #function that will update X matrix to minimize difference between Y - DX
  #must make sure that columns of D are L2 normalized
  #Input:
  #          D: dictionary, a matrix of dim n x K
  #          Y: response matrix of dim n * N
  #        Tol: what is the T_0 that is the maximum number of columns that D can contribute
  # orthogonal: should orthogonal mapping pursuit be used
  #Output
  # X.mat: matrix of dim K x N of updated X matrix
  
  #initialize empty X.mat
  X.mat <- matrix(NA, nrow = ncol(D), ncol = ncol(Y))
  
  #for loop over the columns of Y since each column of X.mat correpsonds to a column in Y
  for(y in 1:ncol(Y)){
    #for testing
    #print(y)
    
    #function that gets the linear combination vector x for column y
    out.combo <- get.linear.combo(D, y.now = Y[,y], orthogonal, Tol)
    if(orthogonal){
      lambda.set <- out.combo$lambda.set
      x.out <- out.combo$x
      x.new <- rep(0,ncol(D))
      for(ii in 1:length(lambda.set)){
        x.new[sort(lambda.set)[ii]] <- x.out[ii]
      }
      x.vec <- x.new
    }else{
      x.vec <- out.combo$x
    }
    # #if the number of non zero elements in x.vec is greater than tol then have to change that
    # if(sum(x.vec != 0) > Tol){
    #   #see which columns of D contribute the most to mapping D to Y
    #   #only want Tol number of them too
    #   important.indices <- order(abs(x.vec))[(length(x.vec) - Tol + 1):length(x.vec)]
    # 
    #   #replace unimportant columns in dictionary with zeroes so they cannot be selected
    #   D.new <- D
    #   D.new[,-important.indices] <- 0
    # 
    #   #get new x.vec with limited dictionary to make sure the Tol is satisfied
    #   out.combo <- get.linear.combo(D.new, y.now = Y[,y], orthogonal, Tol)
    #   lambda.set <- out.combo$lambda.set
    #   x.out <- out.combo$x
    #   x.new <- rep(0,ncol(D))
    #   for(ii in 1:length(lambda.set)){
    #     x.new[sort(lambda.set)[ii]] <- x.out[ii]
    #   }
    #   x.vec <- x.new
    # 
    # #   #work around to make sure the x.vec is of length K but can still handle zeros in D in get.linear.combo
    # #   #since x.vec will be of length Tol instead of length K
    # #   if(orthogonal){
    # #     x.new <- rep(0,ncol(D))
    # #     for(ii in 1:length(important.indices)){
    # #       x.new[sort(important.indices)[ii]] <- x.vec[ii]
    # #     }
    # #     x.vec <- x.new
    # #   }
    # 
    #  }
    
    #assign the column
    X.mat[,y] <- x.vec
  }
  
  return(X.mat)
}

