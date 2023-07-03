mcmc_dag_int = function(X, S, burn, a, U, D, a_pi = NULL, b_pi = NULL){
  
  ###########
  ## INPUT ##
  ###########
  
  # X : (n,q) data matrix
  # D : (n,q) matrix linking observations to intervention targets (D[i,j] = i if observation i was produced under intervention on j)
  
  # S    : the (integer) number of MCMC iterations
  # burn : the (integer) burn-in period
  
  # a, U : hyperparameters of the Wishart prior on Omega
  # a_pi : first shape hyperparameter of the Beta prior for the probability of edge inclusion
  # b_pi : second shape hyperparameter of the Beta prior for the probability of edge inclusion
  
  ############
  ## OUTPUT ##
  ############
  
  # X : the input (n,q) data matrix
  # D : the input (n,q) matrix
  
  # DAG_post : a (q,q,T) array collecting the T = S - burn (q,q) adjacency matrices of the DAGs visited by the MCMC
  
  source("move_dag.R")
  source("marg_like_gaussian.R")
  
  q = dim(X)[2]
  n = dim(X)[1]
  
  if(is.null(a_pi)){a_pi = 1}
  if(is.null(b_pi)){b_pi = q}
  
  Aj_all = lapply(1:q, find_Aj, D = D)
  Sj_all = lapply(1:q, function(j) t(X[Aj_all[[j]],])%*%X[Aj_all[[j]],]) 
  
  DAG_post = array(NA, c(q, q, S))
  
  # Initialize the chain
  
  DAG = matrix(0, q, q)
  
  DAG_post[,,1] = DAG
  
  pb = txtProgressBar(min = 2, max = S, style = 3)
  
  for(s in 1:S){
    
    ## Update the graph
    
    DAG_move = move(A = DAG)
    
    DAG_prop   = DAG_move$A_new
    nodes_prop = DAG_move$nodes
    
    type.operator = DAG_move$type.operator
    
    
    logprior.new = lgamma(sum(DAG_prop) + a_pi) + 
      lgamma(q*(q-1)/2 - sum(DAG_prop) + b_pi - 1)
    
    logprior.old = lgamma(sum(DAG) + a_pi) + 
      lgamma(q*(q-1)/2 - sum(DAG) + b_pi - 1)
    
    logprior = logprior.new - logprior.old
    
    
    # logprior = sum(DAG_prop)*log(w) + (0.5*q*(q-1) - sum(DAG_prop))*(1 - w) - sum(DAG)*log(w) - (0.5*q*(q-1) - sum(DAG))*(1 - w)
    
    u = nodes_prop[1]
    v = nodes_prop[2]
    
    Au = Aj_all[[u]]
    Av = Aj_all[[v]]
    
    Su = Sj_all[[u]]
    Sv = Sj_all[[v]]
    
    marg_prop = marg_B(S = Su, n = length(Au), B = fa(u, DAG_prop), U = U, a = q, q = q) -
                marg_B(S = Su, n = length(Au), B = pa(u, DAG_prop), U = U, a = q, q = q) +
                marg_B(S = Sv, n = length(Av), B = fa(v, DAG_prop), U = U, a = q, q = q) -
                marg_B(S = Sv, n = length(Av), B = pa(v, DAG_prop), U = U, a = q, q = q)
    
    marg = marg_B(S = Su, n = length(Au), B = fa(u, DAG), U = U, a = q, q = q) -
           marg_B(S = Su, n = length(Au), B = pa(u, DAG), U = U, a = q, q = q) +
           marg_B(S = Sv, n = length(Av), B = fa(v, DAG), U = U, a = q, q = q) -
           marg_B(S = Sv, n = length(Av), B = pa(v, DAG), U = U, a = q, q = q)
        
    
    # acceptance ratio
    
    ratio_D = min(0, marg_prop - marg + logprior)
    
    # accept move
    
    if(log(runif(1)) < ratio_D){
      
      DAG = DAG_prop
      
    }
    
    DAG_post[,,s] = DAG
    
    setTxtProgressBar(pb, s)
    close(pb)
    
  }
  
  return(out = list(X = X,
                    D = D,
                    DAG_post = DAG_post[,,(burn + 1):S]))
  
}

