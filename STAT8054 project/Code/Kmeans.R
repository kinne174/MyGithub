#STAT 8054

#Practicing writing the codebook p. 4316 fig. 1

#y is in R^n, D is in R^(n x K), x in R^K, there are N samples of y
#for each column in C need to see which y is most similar then take an average of those ys as the new column in C
#X matrix will comprise of columns of trivial vectors that help translate the columns of y to columns of C
#note that K << N
set.seed(8053)
n <- 10; N <- 500; K <- 50
#randomly set Y values, should be a n x N matrix
#Y <- matrix(c(rnorm(n*N/4),rnorm(mean = 3,n*N/4),rnorm(mean = -3,n*N/4),rnorm(mean = 10,n*N/4)),nrow = n, byrow = F)
#Y <- matrix(rnorm(n*N), nrow = n)
#initialize C, C should be a n x K matrix
C.true <- matrix(rnorm(n*K), ncol = K, nrow = n)

X.true <- matrix(0,nrow = K,ncol = N)
for(i in 1:N){
  indice <- sample(1:K,1)
  X.true[indice,i] <- 1
}

Y <- C.true%*%X.true + rnorm(N*n, sd = sqrt(.1))

C <- matrix(rnorm(n*K),ncol = K)

#reps <- 50
k <- 0
norm.new <- Inf
error <- 1e-10

while(k < 1 || abs(norm.new - norm.prev) > error){
#for(i in 1:reps){

  norm.prev <- norm.new
  k <- k + 1
  
  #R matrix so the rows represent the N samples in Y and the K represents the K columns in C, R[i,j] is 1 if the ith sample
  #of Y is most similar to the jth column in C
  R.part <- matrix(0,nrow = N, ncol = K)
  
  #double for loop to see for each sample of Y which column C closest to it, 
  for(y in 1:ncol(Y)){
    #assign y.now and set the min to be infinity to start comparison
    y.now <- Y[,y]
    min <- Inf
    for(c in 1:ncol(C)){
      #get L2 distance of y.now and each column of C
      diff.now <- sum((y.now - C[,c])^2)
      #if this differnce is less than current min distance re assign min column c and new min
      if(diff.now < min){
        min.c <- c
        min <- diff.now
      }
    }
    #document which column in C for the sample from Y
    R.part[y,min.c] <- 1
  }
  #update C matrix to have the columns of the mean of the Ys determined to "be together"
  for(c in 1:ncol(C)){
    if(sum(R.part[,c]) > 0){
      if(sum(R.part[,c]) > 1){
        C[,c] <- rowMeans(Y[,which(R.part[,c] == 1, arr.ind = T)])
      }else{
        C[,c] <- Y[,which(R.part[,c] == 1, arr.ind = T)]
      }
    }
  }
  #update norm to use in stop rule
  norm.new <- norm(Y - C%*%t(R.part),type = "F")
}

norm(Y - C.true%*%X.true, type = "F")
norm(Y - C%*%t(R.part), type = "F")
sum(apply(t(R.part),1,nonzero.rows))
sum(apply(X.true,1,nonzero.rows))

nonzero.rows <- function(vec){
  if(sum(vec) > 0) return(1)
  else return(0)
}
  
result.mat <- matrix(0,nrow = 3,ncol = K)
for(i in 1:K){
  max.val <- -Inf
  for(j in 1:K){
    proposal.val <- norm(crossprod(C[,i],C.true[,j]),"2")
    if(abs(proposal.val) > max.val){
      max.val <- proposal.val
      indice <- j
    }
  }
  result.mat[1,i] <- indice
  result.mat[2,i] <- max.val
  result.mat[3,i] <- crossprod(C.true[,indice])
}

#do classification by getting the difference between 
new <- rnorm(mean = 3,10)
diff <- c()
for(c in 1:ncol(C)){
  diff[c] <- sum((new - C[,c])^2)
}
which(diff == min(diff), arr.ind = T)
