import pandas as pd
import numpy as np
import networkx as nx
import random

from mcmc.utils import initialize_structures, update_structures
from mcmc.graph_utils import total_possible_add_edges, total_possible_remove_edges, total_possible_reverse_edges, generate_random_dag, propose_new_graph
from mcmc.mcmc_abstract import MCMC_abstract
from mcmc.scores import LogMarginalLikelihood, MarginalLikelihood, BGeScore, BICScore, AICScore


class StructureMCMC( MCMC_abstract ):

    def __init__(self, data: pd.DataFrame,  iterations : int, G : nx.DiGraph, score_function : str = "Log_Marginal_Likelihood"):
            super().__init__(data, iterations, G )
            self.score_function = score_function
            
    def getScoreFunction(self):
        return self.score_function
    
    def run(self, random_restarts = False, restart_freq = 100):
        """
        Perform the structured Markov Chain Monte Carlo (MCMC) for Bayesian Network structure learning.
        
        This function carries out the MCMC procedure to explore the space of possible Bayesian Network structures 
        given the data. It proposes new graph structures and decides whether to accept or reject them based on 
        a scoring function, such as the marginal likelihood.
        """
        
        iter = 0
        accept = 0
        accept_indx = []
        
        candidate_scores = []
        candidate_graphs = [self.G_initial]
        
        # Use initialized graph. If the user did not provide a graph, PC Algo will take place and will generate a graph
        G_current = self.G_initial.copy()
        score_G_current, params_curr = self.compute_score_function( G_current )
        candidate_scores = [score_G_current]
        
        candidate_params = {}
        candidate_params[ iter ] = params_curr

        # start the MCMC loop
        for i in range(self.iterations):
            
            # Propose a new graph by applying an operation to the current graph: add an edge, delete an edge, or reverse an edge
            G_proposed, operation = propose_new_graph( G_current )
            
            # check if G_proposed is a directed acyclic graph, if not, reject it
            if not nx.is_directed_acyclic_graph( G_proposed ):
                continue
            
            # if the proposed graph is a DAG, compute the posterior
            score_G_proposed, params = self.compute_score_function( G_proposed )
            
            # Calculate the acceptance probability
            if "Log" in self.score_function:
                A = self.compute_log_acceptance_ratio(G_current, score_G_current, G_proposed, score_G_proposed, operation)
                u = np.log(np.random.uniform(0, 1)) # Draw a random number in log space
            else:
                A = self.compute_acceptance_ratio(G_current, score_G_current, G_proposed, score_G_proposed, operation)
                u = np.random.uniform(0, 1) # Generate a random number
                
            # metropolis condition    
            if u < A: 
                accept = accept + 1
                accept_indx.append(i)
                    
                # update the current graph and its posterior
                G_current = G_proposed.copy()
                score_G_current = score_G_proposed
            
            if random_restarts:
                if i % restart_freq == 0: # force the chain to jump
                    G_current = generate_random_dag( self.nodes )   # generate a random graph with 50% prob of adding an edge
                    score_G_current, params = self.compute_score_function( G_current )

            candidate_scores.append(score_G_current)
            candidate_graphs.append(G_current)
            candidate_params[iter] = params
            i = i + 1        
            
        return {"scores" : candidate_scores, "acceptance_rate" : accept / self.iterations, "accepted_iterations" : accept_indx, "graphs" : candidate_graphs, "params" : candidate_params} 

    
    def compute_acceptance_ratio(self, G_current, score_G_current, G_proposed, score_G_proposed, operation):
        """
        Compute the acceptance ratio for a proposed move in a Markov Chain Monte Carlo (MCMC) sampling procedure.

        This function calculates the acceptance ratio based on the current and proposed states (graphs), 
        their respective posterior probabilities, and the nature of the proposal distribution. It is used to 
        decide whether to accept or reject the proposed move.

        Returns:
        - float: The acceptance ratio. If this value is greater than a uniformly drawn random number between 
                0 and 1, the proposed move is accepted; otherwise, it's rejected.
        """
        # in structure MCMC, a proposal is only symmetric if an edge reverse operation takes place
        if operation == "reverse_edge":
            # A(G, G') = min(1, P(G' | D) / P(G | D))
            A = self.compute_symmetric_acceptance_ratio(score_G_current, score_G_proposed)
        else:
            #  A(G, G') = min(1, [P(G' | D) * Q(G | G')] / [P(G | D) * Q(G' | G)])
            A =  self.compute_non_symmetric_acceptance_ratio(G_current, score_G_current, G_proposed, score_G_proposed, operation)
        return A

    def compute_log_acceptance_ratio(self, G_current, score_G_current, G_proposed, score_G_proposed, operation):
        """
        Compute the logarithm of the acceptance ratio for a proposed move in a Markov Chain Monte Carlo (MCMC) sampling procedure.

        This function calculates the logarithm of the acceptance ratio based on the current and proposed states (graphs),
        their respective log marginal likelihoods, and the nature of the proposal distribution. It is used to decide whether
        to accept or reject the proposed move, especially when working in log space avoids numerical underflow issues that
        can arise with very small probabilities.

        Returns:
        - float: The logarithm of the acceptance ratio. If this value is greater than the logarithm of a uniformly drawn random number between 0 and 1, the proposed move is accepted; otherwise, it's rejected.
        """
        
        # in structure MCMC, a proposal is only symmetric if an edge reverse operation takes place
        if operation == "reverse_edge":
            # A(G, G') = min(0, logP(G' | D) / P(G | D))
            A = self.compute_log_symmetric_acceptance_ratio(score_G_current, score_G_proposed)
        else:
            #  A(G, G') = min(0, log[P(G' | D) * Q(G | G')] / [P(G | D) * Q(G' | G)])
            A =  self.compute_non_log_symmetric_acceptance_ratio(G_current, score_G_current, G_proposed, score_G_proposed, operation)
        return A

    # PROPOSAL DISTRIBUTIONS AND ACCEPTANCE RATIOS
    #######################################################################
    def Q_G_to_G_prime(self, G, operation):
        """
        Compute the transition probability for the proposed graph structure operation in the MCMC chain.
        
        This function calculates the probability of transitioning from the current graph structure `G` 
        to a new proposed structure `G'` using the specified operation. This probability is used in 
        the acceptance criterion of the MCMC algorithm.
        """
        possible_adds = total_possible_add_edges(G);
        possible_removes = total_possible_remove_edges(G);
        
        nodes = G.nodes()
        total_possible_operations = len(nodes) * (len(nodes) - 1) #possible_adds + possible_removes + possible_reverses
        
        # if G -> G' is an add edge operation, 
        # then count how many add edges are possible
        if operation == "add_edge":
            return possible_adds / total_possible_operations
        
        # if G -> G' is a remove edge operation, 
        # then count how many remove edges are possible
        if operation == "delete_edge":
            return possible_removes / total_possible_operations
        
        return "Operation not found! You can only add_egde, remove_edge, or reverse_edge"


    def Q_G_prime_to_G(self, G_prime, operation):
        """
        Compute the reverse transition probability for the graph structure operation in the MCMC chain.
        
        Given a proposed graph structure `G'`, this function calculates the probability of transitioning 
        back to the original graph structure `G` using the inverse of the specified operation. This reverse 
        transition probability is used in the acceptance criterion of the Metropolis-Hastings MCMC algorithm 
        when the proposal distribution is not symmetric.
        """
        possible_adds = total_possible_add_edges(G_prime);
        possible_removes = total_possible_remove_edges(G_prime);
        
        nodes = G_prime.nodes()
        total_possible_operations = len(nodes) * (len(nodes) - 1) #possible_adds + possible_removes + possible_reverses
        
        # if G -> G' is an add edge operation, then G' -> G is a remove
        # count how many edges are possible to remove
        if operation == "add_edge":
            return possible_removes / total_possible_operations
        
        # if G -> G' is a delete edge operation, then G' -> G is an add
        # count how many edges are possible to add
        if operation == "delete_edge":
            #print(f"Possible adds: {possible_adds}\t Total possible operations: {total_possible_operations}")
            return possible_adds / total_possible_operations
        
        return "Operation not found! You can only add_egde, remove_edge, or reverse_edge"
        
    def compute_non_symmetric_acceptance_ratio(self, G_current, score_G_current, G_proposed, score_G_proposed, operation_G_to_G_prime ):
        """
        Compute the acceptance ratio for the Metropolis-Hastings MCMC algorithm when the proposal distribution is not symmetric.
        
        This function calculates the acceptance ratio for a proposed move in the MCMC chain using the 
        Metropolis-Hastings criterion when transitioning from a current graph structure `G` to a proposed 
        graph structure `G'` through a specified operation. The acceptance ratio accounts for 
        the non-symmetry in the proposal distribution.
        """
        # Compute the proposal distributions at the current and proposed graphs
        Q_G_proposed_given_G = self.Q_G_to_G_prime(G_current, operation_G_to_G_prime)
        Q_G_given_G_proposed = self.Q_G_prime_to_G( G_proposed, operation_G_to_G_prime )

        return min(1, (score_G_proposed * Q_G_given_G_proposed) / (score_G_current * Q_G_proposed_given_G))


    def compute_non_log_symmetric_acceptance_ratio(self, G_current, score_G_current, G_proposed, score_G_proposed, operation_G_to_G_prime ):
        """
        Compute the log of the acceptance ratio for the Metropolis-Hastings MCMC algorithm when the proposal distribution is not symmetric.
        
        This function calculates the log of the acceptance ratio for a proposed move in the MCMC chain using the 
        Metropolis-Hastings criterion when transitioning from a current graph structure `G` to a proposed 
        graph structure `G'` through a specified operation. The acceptance ratio accounts for 
        the non-symmetry in the proposal distribution and operates in the log space for numerical stability.
        """
        Q_G_proposed_given_G = self.Q_G_to_G_prime( G_current, operation_G_to_G_prime )
        Q_G_given_G_proposed = self.Q_G_prime_to_G( G_proposed, operation_G_to_G_prime )
        
        log_numerator = score_G_proposed + np.log(Q_G_given_G_proposed)
        log_denominator = score_G_current + np.log(Q_G_proposed_given_G)

        log_alpha = log_numerator - log_denominator
        return min(0, log_alpha)

    def compute_symmetric_acceptance_ratio(self, score_G_current, score_G_proposed):
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
        return min(1, score_G_proposed / score_G_current)


    def compute_log_symmetric_acceptance_ratio(self, score_G_current, score_G_proposed):
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
        return min(0, score_G_proposed - score_G_current)

    # SCORE FUNCTIONS CALL
    ###############################################################################################################
    def compute_score_function(self, G : nx.DiGraph):
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
        
        if self.score_function == "BIC":
            score = BICScore(self.data, G)
            res, params = score.compute()
            return res, params
        
        if self.score_function == "AIC":
            score = AICScore(self.data, G)
            res, params = score.compute()
            return res, params
        
        if self.score_function == "BGe":
            score = BGeScore(self.data, G)
            res, params = score.compute()
            return res, params
        
        if self.score_function == "Log_Marginal_Likelihood":
            score = LogMarginalLikelihood(self.data, G)
            res, params = score.compute()
            return res, params
        
        if self.score_function == "Marginal_Likelihood":
            try:
                score = MarginalLikelihood(self.data, G)
                res, params = score.compute()
            except OverflowError: # when the marginal is too big, let's reject the graph
                return -1, {}
            
        print("[Error] Score function not found!")
        return res, {}
