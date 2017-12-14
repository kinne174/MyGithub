#STAT 8054 Simulations

#libraries
library(beepr)

#sourced functions
source("C:/Users/Mitch/Documents/UofM/Spring 2017/Stat 8054/Project/Code/LCKsvd.R")


#Seeing how well KSVD can recover a randomized dictionary
n <- 25; N <- 500; D.size <- 50; Tol <- 10
set.seed(8053)
D.true <- matrix(runif(n*D.size,-1,1),nrow = n)
D.true <- t(t(D.true)/apply(D.true,2,norm,type = "2"))
X.true <- matrix(0,nrow = D.size,ncol = N)
for(i in 1:N){
  indices <- sample(1:D.size,Tol,replace = F)
  X.true[indices,i] <- 1
}

Y <- D.true%*%X.true + matrix(rnorm(n*N, sd = sqrt(.5)),ncol = N)
init.time <- system.time(out.init <- LCKsvd.init(Y,matrix(1,nrow = 1,ncol = N),D.size,Tol,1,1))
D.init <- out.init$D.init
ksvd.time <- system.time(out.ksvd <- My.Ksvd(Y,D.init,"OMP",Tol,80))
init.time + ksvd.time
D.ksvd <- out.ksvd$D
X.ksvd <- out.ksvd$X

norm(Y - D.true%*%X.true, "F")
norm(Y - D.ksvd%*%X.ksvd, "F")

result.mat <- matrix(0,nrow = 2,ncol = D.size)
for(i in 1:D.size){
  max.val <- -Inf
  for(j in 1:D.size){
    proposal.val <- norm(crossprod(D.ksvd[,i],D.true[,j]),"2")
    if(abs(proposal.val) > max.val){
      max.val <- proposal.val
      indice <- j
    }
  }
  result.mat[1,i] <- indice
  result.mat[2,i] <- max.val
}
result.mat
length(unique(result.mat[1,]))


beep(8)


#LC KSVD
n <- 5; N <- 500; D.size <- 50; Tol <- 5; m <- 5
set.seed(8053)
out.train <- createResponse(n, N, m)
Y.train <- out.train$Y
H.train <- out.train$H
alpha <- 1; beta <- 1
system.time(out.lcksvd <- My.LCKsvd(Y.train, H.train, D.size, Tol, alpha, beta, "OMP", 80, TRUE))
W.lcksvd <- out.lcksvd$W
D.lcksvd <- out.lcksvd$D

out.test <- createResponse(n, N, m)
Y.test <- out.test$Y
H.test <- out.test$H
X.test <- My.matching.pursuit(D.lcksvd,Y.test,Tol,orthogonal = T)

sum(apply(H.test,2,whichmax)==apply(W.lcksvd%*%X.test,2,whichmax))/N
table(apply(H.test,2,whichmax), apply(W.lcksvd%*%X.test,2,whichmax))/N 

table(apply(H.test,2,whichmax), apply(W.lcksvd%*%out.lcksvd$X,2,whichmax))/N 
sum(apply(H.train,2,whichmax) == apply(W.lcksvd%*%out.lcksvd$X,2,whichmax))/N


y <- rnorm(5)
x <- My.matching.pursuit(D.lcksvd,cbind(y),5)
W.lcksvd%*%x
W.lcksvd[,1:5]

#real data example
load("C:/Users/Mitch/Documents/UofM/Spring 2017/Pubh 8442/Homework/Project/TCGA_Breast_Data.Rdata")
breast <- list(Emat = Expression.matrix, labs = Subtype)
save(breast, file = "C:/Users/Mitch/Documents/breast.RData")
#loads in Expression.matrix, Subtype and GeneNames, disregard Genenames
Emat <- Expression.matrix
labs <- ifelse(Subtype == "Basal",T,F)
H <- rbind(ifelse(Subtype == "Basal",1,0),ifelse(Subtype == "Basal",0,1))
set.seed(8054)
training.indices <- sample(1:348,floor(.6*348))
n.test <- 348 - floor(.6*348)
Y.train <- Emat[,training.indices]
Y.test <- Emat[,-training.indices]
H.train <- H[,training.indices]
H.test <- H[,-training.indices]
D.size <- 30
Tol <- 3
alpha <- 1; beta <- 2
system.time(out.real <- My.LCKsvd(Y.train,H.train,D.size,Tol,alpha,beta,"OMP",80,TRUE))
X.test <- My.matching.pursuit(out.real$D,Y.test,Tol,T)
table(apply(H.test,2,whichmax),apply(out.real$W%*%X.test,2,whichmax))
table(apply(H.test,2,whichmax),apply(H.test,2,whichmax))
sum(diag(table(apply(H.test,2,whichmax),apply(out.real$W%*%X.test,2,whichmax))/n.test))

labels <- read.table(file = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data", header = F)
attr <- read.table(file = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data", header = F)
attr <- t(attr)
sum(apply(attr,2,is.nan))
for(i in 1:nrow(attr)){
  for(j in 1:ncol(attr)){
    if(is.nan(attr[i,j])){
      attr[i,j] <- mean(attr[i,], na.rm = T )
    }
  }
}
H <- rbind(ifelse(labels[,1] == 1,1,0),ifelse(labels[,1] == 1,0,1))
set.seed(8054)
train.indices <- sample(1:ncol(attr), floor(.6*ncol(attr)))
Y.train <- attr[,train.indices]
Y.test <- attr[,-train.indices]
H.train <- H[,train.indices]
H.test <- H[,-train.indices]
D.size <- 50; Tol <- 5
alpha <- 1; beta <- 1
system.time(out.attr <- My.LCKsvd(Y.train, H.train, D.size, Tol, alpha, beta, "OMP", 80, TRUE))
X.test <- My.matching.pursuit(out.attr$D,Y.test,Tol,T)
table(apply(H.test,2,whichmax),apply(out.attr$W%*%X.test,2,whichmax))
table(apply(H.test,2,whichmax),apply(H.test,2,whichmax))
