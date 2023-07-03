###########################
## Preliminary functions ##
###########################

# To find parents of node j in dag

pa = function(j, dag){
  ifelse(all(dag[,j] == 0),
         return(NULL),
         return(as.numeric(which(dag[,j] != 0))))
}

fa = function(j, dag){
  return(as.numeric(c(j, which(dag[,j] != 0))))
}

lGamma = function(s,a){
  
  # Compute the logarithm of the multivariate gamma function Gamma_s(a)
  
  if(s == 0){
    return(0)
  } else{
    
    return((0.25*s*(s - 1))*log(pi) + sum(lgamma(a + 0.5*(1 - 1:s))))
  }
  
}

#######################################################
## Main functions for marginal likelihood evaluation ##
#######################################################

c_norm_const = function(a, U){
  
  # Compute the log-normalizing constant of a Wishart_q(a,U) (prior or posterior)
  
  0.5*a*log(det(U)) - lGamma(ncol(U), a/2)

}

marg_B = function(S, n, B, U, a, q){
  
  # Compute the log-marginal likelihood of a multivariate Gaussian model restricted to variables in the set B
  
  # The marginal likelihood is computed as the ratio of prior and posterior normalizing constants
  
  ###########
  ## INPUT ##
  ###########
  
  # S    : sample covariance matrix
  # n    : number of observations (rows of data matrix)
  # B    : subset of {1,...,q} (variables the marginal likelihood refer to)
  # a, U : hyperparameters of the Wishart prior on Omega
  # q    : number of variables (columns of the data matrix)
  
  ############
  ## OUTPUT ##
  ############
  
  # m : log-marginal likelihood m(X_B)
  
  if(length(B) == 0){
    
    m = 0
    
  }else{
    
    m = -0.5*n*length(B)*log(pi) + c_norm_const(a - (q - length(B)), as.matrix(U[B,B])) -
                                     c_norm_const(a - (q - length(B)) + n, as.matrix(U[B,B] + S[B,B]))
    
  }
  
  return(m)
  
}

find_Aj = function(D, j){
  
  # To find observational measurements relative to node j
  
  Aj = which(D[,j] == 0)
  
  return(Aj)
  
}

marg_like_node_j = function(j, Aj_all, Sj_all, U, a, q, DAG){
  
  # Compute the log-marginal likelihood relative to node j in the DAG
  
  Aj = Aj_all[[j]]
  Sj = Sj_all[[j]]
  
  marg_B(S = Sj, n = length(Aj), B = fa(j, DAG), U = U, a = q, q = q) -
    marg_B(S = Sj, n = length(Aj), B = pa(j, DAG), U = U, a = q, q = q)
  
}

marg_like_dag = function(DAG, Aj_all, Sj_all, a, U, q){
  
  # Compute the DAG log-marginal likelihood (sum over nodes j = 1,...,q)
  
  sum(sapply(1:q, function(j) marg_like_node_j(j, Aj_all, Sj_all, U, a, q, DAG)))
  
}