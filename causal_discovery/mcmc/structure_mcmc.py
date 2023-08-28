import pandas as pd
import numpy as np
import networkx as nx
import random

from mcmc.utils import initialize_structures, update_structures
from mcmc.graph_utils import total_possible_add_edges, total_possible_remove_edges, total_possible_reverse_edges, generate_random_dag, propose_new_graph

from mcmc.scores import LogMarginalLikelihood, MarginalLikelihood, BGeScore, BICScore, AICScore


# STRUCTURE MCMC
##########################################
def structured_MCMC(data : pd.DataFrame, initial_graph : nx.DiGraph, iterations : int, score_function : str,  random_restarts = False, restart_freq = 100, is_proposal_symmetric : bool = False):
    """
    Perform the structured Markov Chain Monte Carlo (MCMC) for Bayesian Network structure learning.
    
    This function carries out the MCMC procedure to explore the space of possible Bayesian Network structures 
    given the data. It proposes new graph structures and decides whether to accept or reject them based on 
    a scoring function, such as the marginal likelihood.
    
    Parameters:
    - data (pd.DataFrame): The dataset for which the Bayesian Network structure is to be learned.
    - initial_graph (nx.DiGraph): The initial directed acyclic graph from which the MCMC procedure starts.
    - iterations (int): Number of MCMC iterations.
    - score_function (str): The name of the score function used to evaluate the graphs.
    - random_restarts (bool, optional): Whether to perform random restarts during MCMC. Default is False.
    - restart_freq (int, optional): Frequency of random restarts if random_restarts is True. Default is 100.
    - is_proposal_symmetric (bool, optional): If the proposal distribution is symmetric or not. Default is False.
    
    Returns:
    - dict: A dictionary containing:
        * marginal_likelihood (list): A list of marginal likelihoods (or scores) for each iteration.
        * acceptance_rate (float): Proportion of proposed graphs that were accepted.
        * accepted_iterations (list): List of iteration numbers where a new graph was accepted.
    - graph_candidates (list): A list of nx.DiGraph objects, one for each iteration.
    - ID_TO_FREQ (dict): A dictionary mapping each graph ID to its frequency of occurrence.
    - ID_TO_GRAPH (dict): A dictionary mapping each graph ID to the actual graph structure.
    - ID_TO_MARGINAL (dict): A dictionary mapping each graph ID to its computed marginal likelihood.
    - OPERATIONS (dict): A dictionary capturing the operations performed to move from one graph to another.
    - HASH_TO_ID (dict): A dictionary mapping hash values of graphs to their IDs.
    - PARAMS (dict): A dictionary mapping each graph ID to its learned parameters.
    
    Note:
    - This function assumes the availability of helper functions like `propose_new_graph`, `compute_score_function`,
    `generate_random_dag`, and others in the environment.
    - The function is stochastic and may produce different results on different runs due to its inherent randomness.
    """
    
    i = 0
    HASH_TO_ID = {}
    ACCEPT = 0
    ACCEPT_INDX = []

    nodes = list(initial_graph.nodes())
    
    posterior_candidates = []
    graph_candidates = [initial_graph]
    
    # Initialize the graph
    G_current = initial_graph  
    posterior_G_current, params_curr = compute_score_function( data, G_current, score_type = score_function )
    posterior_candidates = [posterior_G_current]
    
    PARAMS = {}
    PARAMS[ "G_0" ] = params_curr
    
    # initialize data structures with the initial graph and its posterior
    ID_TO_GRAPH, HASH_TO_ID, ID_TO_FREQ, OPERATIONS, ID_TO_MARGINAL, COUNT = initialize_structures(G_current, posterior_G_current)

    # start the MCMC loop
    for i in range(iterations):
        
        # Propose a new graph by applying an operation to the current graph: add an edge, delete an edge, or reverse an edge
        G_proposed, operation = propose_new_graph(G_current)
        
        # check if G_proposed is a directed acyclic graph, if not, reject it
        if not nx.is_directed_acyclic_graph(G_proposed):
            continue
        
        # if the proposed graph is a DAG, compute the posterior
        posterior_G_proposed, params = compute_score_function( data, G_proposed, score_type = score_function )
        
        # Calculate the acceptance probability
        if "Log" in score_function:
            A = compute_log_acceptance_ratio(G_current, posterior_G_current, G_proposed, posterior_G_proposed, operation, is_proposal_symmetric)
            
            u = np.log(np.random.uniform(0, 1)) # Draw a random number in log space
        else:
            A = compute_acceptance_ratio(G_current, posterior_G_current, G_proposed, posterior_G_proposed, operation, is_proposal_symmetric)

            u = np.random.uniform(0, 1) # Generate a random number
            
        if u < A: 
            ACCEPT = ACCEPT + 1
            ACCEPT_INDX.append(i)
                
            # update the current graph and its posterior
            G_current = G_proposed.copy()
            posterior_G_current = posterior_G_proposed
                
            # add information to the data structures
            ID_TO_GRAPH, HASH_TO_ID, ID_TO_FREQ, OPERATIONS, ID_TO_MARGINAL, COUNT = update_structures(G_current, posterior_G_current, operation, ID_TO_GRAPH, HASH_TO_ID, ID_TO_FREQ, OPERATIONS, ID_TO_MARGINAL, COUNT, posterior_candidates)

        if random_restarts:
            if i % restart_freq == 0: # force the chain to jump
                G_current = generate_random_dag(nodes)  
                posterior_G_current, params = compute_score_function( data, G_current, score_type = score_function )

        posterior_candidates.append(posterior_G_current)
        graph_candidates.append(G_current)
        PARAMS["G_" + str(i)] = params
        i = i + 1        
        
    return { "marginal_likelihood" : posterior_candidates, "acceptance_rate" : ACCEPT / iterations, "accepted_iterations" : ACCEPT_INDX}, graph_candidates, ID_TO_FREQ, ID_TO_GRAPH, ID_TO_MARGINAL, OPERATIONS, HASH_TO_ID, PARAMS



# PROPOSAL DISTRIBUTIONS AND ACCEPTANCE RATIOS
#######################################################################
def Q_G_to_G_prime(G, operation):
    """
    Compute the transition probability for the proposed graph structure operation in the MCMC chain.
    
    This function calculates the probability of transitioning from the current graph structure `G` 
    to a new proposed structure `G'` using the specified operation. This probability is used in 
    the acceptance criterion of the MCMC algorithm.
    
    Parameters:
    - G (nx.DiGraph): The current graph structure in the MCMC chain.
    - operation (str): The operation proposed to move from graph `G` to the new structure `G'`.
                    Allowed values are "add_edge", "delete_edge", and "reverse_edge".
    
    Returns:
    - float: Transition probability of the specified operation, given the current graph structure.
            This is the ratio of the number of possible operations of the specified type to the 
            total number of possible operations.
    - str: A message indicating that the operation is not found, if the operation is not one of 
        the allowed values. This is an error message.
    """
    possible_adds = total_possible_add_edges(G);
    possible_removes = total_possible_remove_edges(G);
    possible_reverses = total_possible_reverse_edges(G);
    
    nodes = G.nodes()
    total_possible_operations = len(nodes) * (len(nodes) - 1) #possible_adds + possible_removes + possible_reverses
    
    # if G -> G' is an add edge operation, then count how many add edges are possible
    if operation == "add_edge":
        #print(f"Possible adds: {possible_adds}\t Total possible operations: {total_possible_operations}")
        return possible_adds / total_possible_operations
    
    # if G -> G' is a remove edge operation, then count how many remove edges are possible
    if operation == "delete_edge":
        #print(f"Possible deletes: {possible_removes}\t Total possible operations: {total_possible_operations}")
        return possible_removes / total_possible_operations
    
    # if G -> G' is a reverse edge operation, then count how many reverse edges are possible
    if operation == "reverse_edge":
        #print(f"Possible reverses: {possible_reverses}\t Total possible operations: {total_possible_operations}")
        return possible_reverses / total_possible_operations
    
    return "Operation not found! You can only add_egde, remove_edge, or reverse_edge"


def Q_G_prime_to_G(G_prime, operation):
    """
    Compute the reverse transition probability for the graph structure operation in the MCMC chain.
    
    Given a proposed graph structure `G'`, this function calculates the probability of transitioning 
    back to the original graph structure `G` using the inverse of the specified operation. This reverse 
    transition probability is used in the acceptance criterion of the Metropolis-Hastings MCMC algorithm 
    when the proposal distribution is not symmetric.
    
    Parameters:
    - G_prime (nx.DiGraph): The proposed graph structure in the MCMC chain.
    - operation (str): The operation that was used to move from the original graph structure `G` to 
                    the proposed structure `G'`. Allowed values are "add_edge", "delete_edge", and 
                    "reverse_edge".
    
    Returns:
    - float: Reverse transition probability of the specified operation, given the proposed graph structure.
            This is the ratio of the number of possible inverse operations of the specified type to the 
            total number of possible operations.
    - str: A message indicating that the operation is not found, if the operation is not one of 
        the allowed values. This is an error message.
    """
    possible_adds = total_possible_add_edges(G_prime);
    possible_removes = total_possible_remove_edges(G_prime);
    possible_reverses = total_possible_reverse_edges(G_prime);
    
    nodes = G_prime.nodes()
    total_possible_operations = len(nodes) * (len(nodes) - 1) #possible_adds + possible_removes + possible_reverses
    
    # if G -> G' is an add edge operation, then count how many remove edges are possible
    if operation == "add_edge":
        #print(f"Possible removes: {possible_removes}\t Total possible operations: {total_possible_operations}")
        return possible_removes / total_possible_operations
    
    # if G -> G' is a remove edge operation, then count how many remove edges are possible
    if operation == "delete_edge":
        #print(f"Possible adds: {possible_adds}\t Total possible operations: {total_possible_operations}")
        return possible_adds / total_possible_operations
    
    # if G -> G' is a reverse edge operation, then count how many reverse edges are possible
    if operation == "reverse_edge":
        #print(f"Possible reverses: {possible_reverses}\t Total possible operations: {total_possible_operations}")
        return possible_reverses / total_possible_operations
    
    return "Operation not found! You can only add_egde, remove_edge, or reverse_edge"
    
def compute_non_symmetric_acceptance_ratio(G_current, posterior_G_current, G_proposed, posterior_G_proposed, operation_G_to_G_prime ):
    """
    Compute the acceptance ratio for the Metropolis-Hastings MCMC algorithm when the proposal distribution is not symmetric.
    
    This function calculates the acceptance ratio for a proposed move in the MCMC chain using the 
    Metropolis-Hastings criterion when transitioning from a current graph structure `G` to a proposed 
    graph structure `G'` through a specified operation. The acceptance ratio accounts for 
    the non-symmetry in the proposal distribution.
    
    Parameters:
    - G_current (nx.DiGraph): The current graph structure in the MCMC chain.
    - posterior_G_current (float): The posterior probability (or any score proportional to it) 
                                of the current graph structure `G`.
    - G_proposed (nx.DiGraph): The proposed graph structure in the MCMC chain.
    - posterior_G_proposed (float): The posterior probability (or any score proportional to it) 
                                of the proposed graph structure `G'`.
    - operation_G_to_G_prime (str): The operation that was used to move from the current graph 
                                structure `G` to the proposed structure `G'`. Allowed values are 
                                "add_edge", "delete_edge", and "reverse_edge".
    
    Returns:
    - float: Acceptance ratio for the proposed move. This value is used to determine whether to accept 
            or reject the proposed move in the MCMC chain.
    """
    # Compute the proposal distributions at the current and proposed graphs
    Q_G_proposed_given_G = Q_G_to_G_prime(G_current, operation_G_to_G_prime)
    Q_G_given_G_proposed = Q_G_prime_to_G( G_proposed, operation_G_to_G_prime )

    return min(1, (posterior_G_proposed * Q_G_given_G_proposed) / (posterior_G_current * Q_G_proposed_given_G))


def compute_non_log_symmetric_acceptance_ratio(G_current, log_marginal_likelihood_G_current, G_proposed, log_marginal_likelihood_G_proposed, operation_G_to_G_prime ):
    """
    Compute the log of the acceptance ratio for the Metropolis-Hastings MCMC algorithm when the proposal distribution is not symmetric.
    
    This function calculates the log of the acceptance ratio for a proposed move in the MCMC chain using the 
    Metropolis-Hastings criterion when transitioning from a current graph structure `G` to a proposed 
    graph structure `G'` through a specified operation. The acceptance ratio accounts for 
    the non-symmetry in the proposal distribution and operates in the log space for numerical stability.
    
    Parameters:
    - G_current (nx.DiGraph): The current graph structure in the MCMC chain.
    - log_marginal_likelihood_G_current (float): The log of the marginal likelihood of the current graph structure `G`.
    - G_proposed (nx.DiGraph): The proposed graph structure in the MCMC chain.
    - log_marginal_likelihood_G_proposed (float): The log of the marginal likelihood of the proposed graph structure `G'`.
    - operation_G_to_G_prime (str): The operation that was used to move from the current graph 
                                structure `G` to the proposed structure `G'`. Allowed values are 
                                "add_edge", "delete_edge", and "reverse_edge".
    
    Returns:
    - float: Log of the acceptance ratio for the proposed move. This value is used to determine whether to accept 
            or reject the proposed move in the MCMC chain. If the returned value is negative, it represents the actual 
            log acceptance ratio. If zero, it means the acceptance probability is 1
    """
    Q_G_proposed_given_G = Q_G_to_G_prime( G_current, operation_G_to_G_prime )
    Q_G_given_G_proposed = Q_G_prime_to_G( G_proposed, operation_G_to_G_prime )
    
    log_numerator = log_marginal_likelihood_G_proposed + np.log(Q_G_given_G_proposed)
    log_denominator = log_marginal_likelihood_G_current + np.log(Q_G_proposed_given_G)

    log_alpha = log_numerator - log_denominator
    return min(0, log_alpha)

def compute_symmetric_acceptance_ratio(posterior_G_current, posterior_G_proposed):
    """
    Compute the symmetric acceptance ratio for Metropolis-Hastings in Markov Chain Monte Carlo (MCMC) sampling.

    This function calculates the acceptance ratio for a proposed move in an MCMC sampling procedure when 
    the proposal distribution is symmetric. It is used to decide whether to accept or reject the proposed move 
    based on the posterior probabilities of the current and proposed states.

    Parameters:
    - posterior_G_current (float): The posterior probability of the current state (or graph) in the MCMC chain.
    - posterior_G_proposed (float): The posterior probability of the proposed state (or graph) in the MCMC chain.

    Returns:
    - float: The acceptance ratio, which is a value between 0 and 1. If this value is higher than a uniformly 
            drawn random number between 0 and 1, the proposed move is accepted; otherwise, it's rejected.
    """
    return min(1, posterior_G_proposed / posterior_G_current)


def compute_log_symmetric_acceptance_ratio(log_marginal_likelihood_G_current, log_marginal_likelihood_G_proposed):
    """
    Compute the symmetric acceptance ratio in logarithmic scale for Metropolis-Hastings in Markov Chain Monte Carlo (MCMC) sampling.

    This function calculates the acceptance ratio for a proposed move in an MCMC sampling procedure when 
    the proposal distribution is symmetric and both the current and proposed likelihoods are provided in logarithmic scale.
    The logarithmic scale is often used to avoid underflow issues when working with very small likelihood values.
    It is used to decide whether to accept or reject the proposed move based on the log likelihoods of the current and proposed states.

    Parameters:
    - log_marginal_likelihood_G_current (float): The logarithm of the marginal likelihood (or posterior probability) 
                                                of the current state (or graph) in the MCMC chain.
    - log_marginal_likelihood_G_proposed (float): The logarithm of the marginal likelihood (or posterior probability) 
                                                of the proposed state (or graph) in the MCMC chain.

    Returns:
    - float: The log acceptance ratio. If this value is higher than the logarithm of a uniformly drawn random number 
            between 0 and 1, the proposed move is accepted; otherwise, it's rejected.
    """
    return min(0, log_marginal_likelihood_G_proposed - log_marginal_likelihood_G_current)

def compute_acceptance_ratio(G_current, posterior_G_current, G_proposed, posterior_G_proposed, operation, is_proposal_symmetric):
    """
    Compute the acceptance ratio for a proposed move in a Markov Chain Monte Carlo (MCMC) sampling procedure.

    This function calculates the acceptance ratio based on the current and proposed states (graphs), 
    their respective posterior probabilities, and the nature of the proposal distribution. It is used to 
    decide whether to accept or reject the proposed move.

    Parameters:
    - G_current (nx.DiGraph): The current state (graph) in the MCMC chain.
    - posterior_G_current (float): The posterior probability of the current state `G_current`.
    - G_proposed (nx.DiGraph): The proposed state (graph) for the next step in the MCMC chain.
    - posterior_G_proposed (float): The posterior probability of the proposed state `G_proposed`.
    - operation (str): The operation that was applied to transition from `G_current` to `G_proposed`. 
                    Typically one of: "add_edge", "delete_edge", or "reverse_edge".
    - is_proposal_symmetric (bool): A flag indicating whether the proposal distribution is symmetric. 
                                If `True`, the proposal distribution Q(G' | G) is the same as Q(G | G').

    Returns:
    - float: The acceptance ratio. If this value is greater than a uniformly drawn random number between 
            0 and 1, the proposed move is accepted; otherwise, it's rejected.
    """
    if is_proposal_symmetric:
        # A(G, G') = min(1, P(G' | D) / P(G | D))
        A = compute_symmetric_acceptance_ratio(posterior_G_current, G_proposed, posterior_G_proposed)
    else:
        #  A(G, G') = min(1, [P(G' | D) * Q(G | G')] / [P(G | D) * Q(G' | G)])
        A =  compute_non_symmetric_acceptance_ratio(G_current, posterior_G_current, G_proposed, posterior_G_proposed, operation)
    return A


def compute_log_acceptance_ratio(G_current, log_marginal_likelihood_G_current, G_proposed, log_marginal_likelihood_G_proposed, operation, is_proposal_symmetric):
    """
    Compute the logarithm of the acceptance ratio for a proposed move in a Markov Chain Monte Carlo (MCMC) sampling procedure.

    This function calculates the logarithm of the acceptance ratio based on the current and proposed states (graphs),
    their respective log marginal likelihoods, and the nature of the proposal distribution. It is used to decide whether
    to accept or reject the proposed move, especially when working in log space avoids numerical underflow issues that
    can arise with very small probabilities.

    Parameters:
    - G_current (nx.DiGraph): The current state (graph) in the MCMC chain.
    - log_marginal_likelihood_G_current (float): The logarithm of the marginal likelihood of the current state `G_current`.
    - G_proposed (nx.DiGraph): The proposed state (graph) for the next step in the MCMC chain.
    - log_marginal_likelihood_G_proposed (float): The logarithm of the marginal likelihood of the proposed state `G_proposed`.
    - operation (str): The operation that was applied to transition from `G_current` to `G_proposed`. Typically one of: "add_edge", "delete_edge", or "reverse_edge".
    - is_proposal_symmetric (bool): A flag indicating whether the proposal distribution is symmetric. If `True`, the proposal distribution Q(G' | G) is the same as Q(G | G').

    Returns:
    - float: The logarithm of the acceptance ratio. If this value is greater than the logarithm of a uniformly drawn random number between 0 and 1, the proposed move is accepted; otherwise, it's rejected.
    """
    if is_proposal_symmetric:
        # A(G, G') = min(0, logP(G' | D) / P(G | D))
        A = compute_log_symmetric_acceptance_ratio(log_marginal_likelihood_G_current, G_proposed, log_marginal_likelihood_G_proposed)
    else:
        #  A(G, G') = min(0, log[P(G' | D) * Q(G | G')] / [P(G | D) * Q(G' | G)])
        A =  compute_non_log_symmetric_acceptance_ratio(G_current, log_marginal_likelihood_G_current, G_proposed, log_marginal_likelihood_G_proposed, operation)
    return A

# SCORE FUNCTIONS CALL
###############################################################################################################
def compute_score_function( data : pd.DataFrame, G : nx.DiGraph, score_type : str = "Log_Marginal_Likelihood" ):
    """
    Compute the specified score for a given Bayesian network structure and dataset.

    This function calculates a specified score (like BIC, AIC, BGe, etc.) for a given graph structure `G` and dataset.
    The score reflects the quality or fit of the Bayesian network to the data. The function supports a variety of 
    scoring methods, each of which has its own assumptions and properties.

    Parameters:
    - data (pd.DataFrame): The dataset for which the score is to be computed. Each column corresponds to a variable 
                        and each row to a data point.
    - G (nx.DiGraph): The graph structure (Bayesian network) for which the score is to be computed.
    - score_type (str, optional): The type of score to be computed. Allowed values are "BIC", "AIC", "BGe", 
                                "Log_Marginal_Likelihood", and "Marginal_Likelihood". Default is "Marginal".

    Returns:
    - float: The computed score for the Bayesian network given the data. Higher values indicate a better fit 
            (for some scores; for others like BIC and AIC, lower is better).
    - dict: A dictionary containing the parameters (e.g., conditional probabilities or regression coefficients) 
            estimated during the computation of the score.

    Raises:
    - OverflowError: If the computation of the Marginal Likelihood results in a value too large to represent.
    """  
    if score_type == "BIC":
        score = BICScore(data, G)
        res, params = score.compute()
        return res, params
    
    if score_type == "AIC":
        score = AICScore(data, G)
        res, params = score.compute()
        return res, params
    
    if score_type == "BGe":
        score = BGeScore(data, G)
        res, params = score.compute()
        return res, params
    
    if score_type == "Log_Marginal_Likelihood":
        score = LogMarginalLikelihood(data, G)
        res, params = score.compute()
        return res, params
    
    if score_type == "Marginal_Likelihood":
        try:
            score = MarginalLikelihood(data, G)
            res, params = score.compute()
        except OverflowError: # when the marginal is too big, let's reject the graph
            return -1, {}
        
    print("[Error] Score function not found!")
    return res, {}
