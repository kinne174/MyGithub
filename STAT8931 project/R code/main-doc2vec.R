setwd("C:/Users/Mitch/Documents/UofM/Fall 2017/STAT 8931/Project")

source("discriminant-analysis-functions.r")
source("ridge-functions.r")
source("fold-matrix.r")

models_list = c("DM", "BOW")
doc_size_list = c("250", "500", "1000")
num_samples_list = c("25000", "50000", "100000")

folds = 5
lam.vec=10^seq(from=-12, to=-8, by=0.25)

set.seed(8931)

for(m in models_list){
  for(d in doc_size_list){
    for(num in num_samples_list){
      
      savename_LDA = paste("Output/LDA",num, "_",m, "_", d,".rdata", sep = "")
      savename_QDA = paste("Output/QDA",num, "_",m, "_", d,".rdata", sep = "")
      
      file_name = paste("Data/doc2vec_vectors_",num,"_", m, "_", d, ".csv",sep = "")
      dat = read.csv(file_name, header = F)
      
      y=as.numeric(dat[,as.numeric(d) + 1])
      ## 2=positive, 1=negative
      x=as.matrix(dat[,-1*(as.numeric(d) + 1)])
      
      err.MLE_L=numeric(folds)
      err.diagonal_L=numeric(folds)
      err.ridge_L=numeric(folds)
      
      picked.ridge_L=numeric(folds)
      
      err.MLE_Q=numeric(folds)
      err.diagonal_Q=numeric(folds)
      err.ridge_Q=numeric(folds)
      
      picked.ridge_Q=matrix(NA, nrow=folds, ncol=2)
      
      n=dim(x)[1]
      p=dim(x)[2]
      
      fold.mat <- fold.matrix(folds, n)
      
      for(k in 1:folds){
        
        x.tr=x[-fold.mat[,k],]
        y.tr=y[-fold.mat[,k]]
        x.te=x[fold.mat[,k],,drop=FALSE]
        y.te=y[fold.mat[,k]]
        
        
        ## regular maximum likelihood
        fit.MLE=fit.lda(X=x.tr, y=y.tr, method="MLE")
        predicted.responses=classify.qda(fit=fit.MLE, Xtest=x.te)
        err.MLE_L[k]=sum( predicted.responses != y.te )
        
        rm(fit.MLE)
        
        ## using the diagonal inverse covariance estimator
        fit.diagonal=fit.lda(X=x.tr, y=y.tr, method="diagonal")
        predicted.responses=classify.qda(fit=fit.diagonal, Xtest=x.te)
        err.diagonal_L[k]=sum( predicted.responses != y.te )
        
        rm(fit.diagonal)
        
        ## using the ridge-penalized likelihood inverse covariance estimator
        fit.ridge=fit.lda(X=x.tr, y=y.tr, method="ridge", lam.vec=lam.vec)
        predicted.responses=classify.qda(fit=fit.ridge, Xtest=x.te)
        err.ridge_L[k]=sum( predicted.responses != y.te )
        
        picked.ridge_L[k]=fit.ridge$picked.ridge
        
        rm(fit.ridge)
        
        ## regular maximum likelihood
        fit.MLE=fit.qda(X=x.tr, y=y.tr, method="MLE")
        predicted.responses=classify.qda(fit=fit.MLE, Xtest=x.te)
        err.MLE_Q[k]=sum( predicted.responses != y.te )
        
        rm(fit.MLE)
        
        ## using the diagonal inverse covariance estimator
        fit.diagonal=fit.qda(X=x.tr, y=y.tr, method="diagonal")
        predicted.responses=classify.qda(fit=fit.diagonal, Xtest=x.te)
        err.diagonal_Q[k]=sum( predicted.responses != y.te )
        
        rm(fit.diagonal)
        
        ## using the ridge-penalized likelihood inverse covariance estimator
        fit.ridge=fit.qda(X=x.tr, y=y.tr, method="ridge", lam.vec=lam.vec)
        predicted.responses=classify.qda(fit=fit.ridge, Xtest=x.te)
        err.ridge_Q[k]=sum( predicted.responses != y.te )
        
        picked.ridge_Q[k,]=fit.ridge$picked.ridge
        
        rm(fit.ridge)
        
        cat("finished fold k=", k, "for model", m, "using doc size",d,"and num samples",num, "\n")
      }
      save(err.MLE_L, err.diagonal_L, err.ridge_L, picked.ridge_L, lam.vec, file=savename_LDA)
      save(err.MLE_Q, err.diagonal_Q, err.ridge_Q, picked.ridge_Q, lam.vec, file=savename_QDA)
      
      
    }
  }
}









