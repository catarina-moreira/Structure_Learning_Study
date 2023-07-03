get_summaries_from_posterior = function(Graphs, X, D, a, U, a_pi = NULL, b_pi = NULL){
  
  ###########
  ## INPUT ##
  ###########
  
  # Graphs : (q,q,T) array collecting the T = S - burn adjacency matrices of DAGs visited by the MCMC
  # X      : input (n,q) data matrix
  # D      : (n,q) matrix linking observations to intervention targets
  
  # a, U       : hyperparameters of the Wishart prior
  # a_pi, b_pi : hyperparameters of the Beta prior on the probabilities of edge inclusion
  
  ############
  ## OUTPUT ##
  ############
  
  # MAP : (q,q) adjacency matrix of the Maximum A Posteriori DAG model estimate
  # PPI : (q,q) matrix collecting posterior probabilities of edge inclusion for each directed edge (u,v)
  
  q = ncol(X)
  
  if(is.null(a_pi)){a_pi = 1}
  if(is.null(b_pi)){b_pi = q}
  
  # Graphs is the array containing the S adjacency matrices of the DAGs visited by the MCMC
  
  library(BCDAG)
  
  bd_encode <- function(matrix, separator = ";") {
    paste(matrix, collapse = separator)
  }
  
  
  bd_decode <- function(string, separator = ";") 
  {
    vec4mat <- as.numeric(strsplit(string, separator)[[1]])
    q <- length(vec4mat)
    matrix(vec4mat, ncol = sqrt(q))
  }
  
  
  dag_code = apply(Graphs, 3, bd_encode, separator = "")
  q        = dim(Graphs)[1]
  
  uniq_dag = unique(dag_code)
  
  ##################################################################################
  ## Find Maximum A Posteriori (MAP) DAG using re-normalized marginal likelihoods ##
  ##################################################################################
  
  Aj_all = lapply(1:q, find_Aj, D = D)
  Sj_all = lapply(1:q, function(j) t(X[Aj_all[[j]],])%*%X[Aj_all[[j]],]) 
  
  margs  = c()
  priors = c()
  
  probs_post = c()
  
  K = length(uniq_dag)
  
  for(k in 1:K){
    
    margs[k] = marg_like_dag(DAG = bd_decode(uniq_dag[k], separator = ""), Aj_all = Aj_all, Sj_all = Sj_all, a = a, U = U, q = q)
    
    priors[k] = lgamma(sum(bd_decode(uniq_dag[k], separator = "")) + a_pi) +
      lgamma(q*(q-1)/2 - sum(bd_decode(uniq_dag[k], separator = "")) + b_pi - 1)
    
  }
  
  for(k in 1:K){
    
    probs_post[k] = (1 + sum(exp(margs[-k] - margs[k])*(exp(priors[-k] - priors[k]))))^(-1)
    
  }
  
  MAP = bd_decode(uniq_dag[which.max(probs_post)], separator = "")
  colnames(MAP) = rownames(MAP) = colnames(X)
  
  #############################################################
  ## Compute Posterior Probabilities of edge Inclusion (PPI) ##
  #############################################################
  
  PPI = matrix(0, q, q)
  colnames(PPI) = rownames(PPI) = colnames(X)
  
  for(k in 1:K){
    
    PPI = PPI + bd_decode(uniq_dag[k], separator = "")*probs_post[k]
    
  }
  
  return(list(MAP = MAP,
              PPI = PPI))
  
}
