library(gRbase)

###########################
## Preliminary functions ##
###########################

# Transitions between DAGs are obtained through three types of actions applied to the input DAG

# A     : (q,q) adjacency matrix of the input DAG
# nodes : (2,1) vector with numerical labels of the two nodes on which the action is applied

id = function(A, nodes){ # Insert directed edge x -> y (type 1)
  x = nodes[1]
  y = nodes[2]
  A[x,y] = 1
  return(A)
}

dd = function(A, nodes){ # Delete directed edge x -> y (type 2)
  x = nodes[1]
  y = nodes[2]
  A[x,y] = 0
  return(A)
}

rd = function(A, nodes){ # Reverse directed edge x -> y (type 3)
  x = nodes[1]
  y = nodes[2]
  A[x,y] = 0 
  A[y,x] = 1
  return(A)
}

# To compute the number of edges in the DAG

n.edge = function(A){
  sum(A)
}

names   = c("action","test","x","y")
actions = c("id","dd","rd")

###################
## Main function ##
##################

move = function(A, q = q){
  
  ###########
  ## INPUT ##
  ###########
  
  # A : (q,q) adjacency matrix of the initial DAG
  # q : number of vertices in the DAG

  ############
  ## Output ##
  ############
  
  # A_new         : (q,q) adjacency matrix of the modified (proposed) DAG
  # type.operator : type of the operator applied to A to obtain A_new (1: insert; 2: delete; 3: reverse edge)
  # nodes         : (2,1) vector with numerical labels of the two nodes involved in the move
  
  A_na = A
  diag(A_na) = NA
  
  id_set = c()
  dd_set = c()
  rd_set = c()
  
  # set of nodes for id
  
  set_id = which(A_na == 0, TRUE)
  
  if(length(set_id) != 0){
    id_set = cbind(1, rbind(set_id))
  }
  
  # set of nodes for dd
  
  set_dd = which(A_na == 1, TRUE)
  
  if(length(set_dd != 0)){
    dd_set = cbind(2, set_dd)
  }
  
  # set of nodes for rd
  
  set_rd = which(A_na == 1, TRUE)
  
  if(length(set_rd != 0)){
    rd_set = cbind(3, set_rd)
  }
  
  O = rbind(id_set, dd_set, rd_set)

  repeat {
    
    i = sample(dim(O)[1],1)
    
      act_to_exe  = paste0(actions[O[i,1]],"(A=A,c(",as.vector(O[i,2]),",",as.vector(O[i,3]),"))")
      A_succ      = eval(parse(text = act_to_exe))
      act_to_eval = paste0("is.DAG(A_succ)")
      val = eval(parse(text = act_to_eval))
    
    if (val != 0){
      break
    }
  }
  
  A_new = A_succ
  
  return(list(A_new = A_new, type.operator = O[i,1], nodes = O[i,2:3]))
  
}

# 3 actions

# A     adjacency matrix of CPDAG C
# x,y   nodes involved in the action


id = function(A, nodes){
  x = nodes[1]
  y = nodes[2]
  A[x,y] = 1
  return(A)
}

dd = function(A, nodes){
  x = nodes[1]
  y = nodes[2]
  A[x,y] = 0
  return(A)
}

rd = function(A, nodes){ # reverse D x -> y
  x = nodes[1]
  y = nodes[2]
  A[x,y] = 0 
  A[y,x] = 1
  return(A)
}
