fold.matrix <- function(folds, num.indices){
  if (folds > num.indices) return("ERROR: folds is greater than number of indices")
  num.leaveOut <- floor(num.indices/folds)
  perm.indices <- sample(num.indices)
  indices.matrix <- matrix(perm.indices[1:(num.leaveOut*folds)], nrow =num.leaveOut,ncol = folds )
  return(indices.matrix)
}