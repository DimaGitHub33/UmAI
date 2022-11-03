

######################################################################
######################################################################
############################## PRIDIT ################################
######################################################################
######################################################################



######################################################################
####################### Function ridit.calc ##########################
######################################################################



ridit.calc <- function(t) {
  
  p.vector.list <- lapply(t, function(x) prop.table(table(x)))  
  names(p.vector.list) <- names(t)  
  
  s.vector.list <- vector('list', length(p.vector.list))
  for (j in 1:length(p.vector.list)){
    x <- p.vector.list[[j]]
    s.vector.list[[j]] <- numeric(length(x))
    s.vector.list[[j]][1] <- -sum(x[2:length(x)])
    for (i in 2:length(x)){
      s.vector.list[[j]][i] <- sum(x[1:(i-1)]) - (1-sum(x[1:i]))
    }
    names(s.vector.list[[j]]) <- names(p.vector.list[[j]])
  }
  names(s.vector.list) <- names(t)
  
  
  t2 <- data.frame(matrix(nr=nrow(t), nc=ncol(t)))
  names(t2) <- names(t)
  for (i in 1:ncol(t)){
    for (j in 1:nrow(t)){
      t2[j, i] <- s.vector.list[[i]][names(s.vector.list[[i]]) == t[j, i]]
      assign('F',t2, envir=globalenv())
    }
  }  
}





######################################################################
####################### Function iteracions ##########################
######################################################################


iteracions <- function(t){
  
  #transposta de F
  
  Ft <- t(F)
  
  
  #creem el vector W0
  
  W0 <- as.matrix(numeric(length(F)))
  
  for(j in 1:length(F)){
    
    W0[j] <- 1
  }
  
  
  
  S0 <- data.frame(as.matrix(F)%*%W0)
  
  W1a <- as.matrix(Ft)%*%as.matrix(S0)
  
  W1b <- norm(W1a, type="2")
  
  W1 <- W1a/W1b
  
  S1 <- data.frame(as.matrix(F)%*%W1)
  
  condp1=1
  
  iter <- 1

  
  while (condp1 > 0){
    
    iter <- iter+1
    
    Wa <- as.matrix(Ft)%*%as.matrix(S1) 
    
    Wb <- norm(Wa, type="2")
    
    W2 <- Wa/Wb
    
    S2 <- data.frame(as.matrix(F)%*%W2)
    
    W0 <- W1
    
    W1 <- W2
    
    S1 <- S2
    
    
    if (W0[1,1] == W1[1,1] && W0[2,1] == W1[2,1])  {
      
      condp1 =0
      
    } else {
      
      condp1=condp1+1
    }

  }
  
  cat("\nIteracions= ", iter, "\n")
  return(W2)

  
}





######################################################################
###################### Principal Components ##########################
######################################################################


### We check the interations with Principal Components

#ridit.calc(t)

CompPrinc <- function(t){

  PCA <- princomp(F)
  load <- as.vector(PCA$loadings[,1])
  return(load)

}


##########################################################
# We do not do this by now

  PCA <- princomp(F)
  load <- as.vector(PCA$loadings[,1])
  scores <- data.frame(as.matrix(F)%*%load)
  row.names(scores) <- row.names(F)
  colnames(scores) <- 'Score'
  assign('ridit.finalscore', cbind(F,scores), envir=globalenv())
  assign('originaldata.finalscore', cbind(t,scores), envir=globalenv())

##########################################################





######################################################################
######################## Matrix B, Vector C ##########################
######################################################################




matBvectC <- function(t){
  
  p.vector.list <- lapply(t, function(x) prop.table(table(x)))
  
  b.vector.list <- vector('list', length(p.vector.list))
  
  for(j in 1:length(p.vector.list)){
    
    b.vector.list[[j]][1] <- -p.vector.list[[j]][2]
    b.vector.list[[j]][2] <- p.vector.list[[j]][1]
    
  }
  
  names(b.vector.list) <- names(p.vector.list)
  
  
  matB <- diag(nrow=ncol(t)*2, ncol=ncol(t)*2)
  
  auxcol <- b.vector.list[[1]]	
  cont <- 1
  for(j in 1:length(auxcol)){
    for(i in 1:length(b.vector.list)){
      matB[cont,cont] <- b.vector.list[[i]][j] 
      cont <- cont +1	
      
    }
  }
  
  
  matB
  Winf

  cat("\nMatriu B\n"); print(matB)
  cat("\nVector Winf\n"); print(Winf)
  
  Winf2 <- as.matrix(numeric(length(Winf)*2))
  
  for(i in 1:length(Winf)){
    Winf2[i, 1] <- Winf[i]
    Winf2[i+length(Winf), 1] <- Winf[i]
  }
  
  ######################################################
  #   multiplying matrix B times Winf we obtain vector C
  
  (vectorC <- matB%*%Winf2)
  
  #cat("\nVector C\n"); print(vectorC)

  return (vectorC)
  
}



  
  ######################################################################
  ############################# Examples ###############################
  ######################################################################
  
  
  ######################################################################
  ########################## Example 1 #################################
  ######################################################################
  
  
  A <- rep(c("Si", "No"), c(2,3))
  B <- rep(c("Si", "No", "Si"), c(1,1,3))
  C <- rep(c("S�","No", "S�", "No"), c(1,1,2,1))
  D <- rep(c("S�", "No", "S�"), c(2, 1, 2))
  (x <- data.frame(A, B, C, D))
  

  ridit.calc(x)
  (F)
  (Winf <- iteracions(x))
  (Compr_PCA <- CompPrinc(x))
  vectC <- matBvectC(x)
  cat("\nVector C\n"); print(vectC)
  
  
  ######################################################################
  ########################## Example 2 #################################
  ######################################################################
  
  A1 <- rep(c("Si", "No"), c(2,3))
  B1 <- rep(c("Si", "No", "Si"), c(1,1,3))
  (x1 <- data.frame(A1,B1))
  
  ridit.calc(x1)
  (F)
  (Winf <- iteracions(x1))
  (Compr_PCA <- CompPrinc(x1))
  vectC <- matBvectC(x1)
  cat("\nVector C\n"); print(vectC)
  
  
  
  
  # STEPS: 
  
  # Step 1: We "execute" the functions (ridit.calc, iteracions, ComprPrinc and matBvecC)
  # Step 2: From function ridit.calc we obtain F, in the iterations function
  #         we obtain Winf, from function Compr_PCA we obtain the PCA (which must  
  #         coincide with Winf) and from function matBvectC we obtain the diagonal matrix B
  #         and vector C.
  # Step 3: We run the examples
  # Step 4: We run the functions
  

  
getwd()
setwd("UmAI/")
Fullcov = read.csv("x3.csv",header=TRUE)
ridit.calc(Fullcov)
(F)
(Winf <- iteracions(Fullcov))
(Compr_PCA <- CompPrinc(Fullcov))
vectC <- matBvectC(Fullcov)
cat("\nVector C\n"); print(vectC)
yhat=(as.matrix(F))%*%as.matrix(Winf)
yobs = read.csv("Fullcoverage.csv",header=TRUE)
all=data.frame(yhat, yobs)
r=subset.data.frame(all, y==0)
r1=subset.data.frame(all, y==1)
hist(r$as.matrix.F......W0)
hist(r1$as.matrix.F......W0)
vectC*1000
 
  
  
  