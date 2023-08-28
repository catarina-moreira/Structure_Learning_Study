import pandas as pd
import numpy as np
import networkx as nx

from scipy.special import gammaln

from abc import ABC, abstractmethod

class Score(ABC):
    """
    Base class for scoring Bayesian networks. This class provides basic functionality 
    for storing and accessing data, graph structure, and various graph attributes.

    Attributes:
    - df (pd.DataFrame): The dataset used for computing the score.
    - G (nx.DiGraph): The graph structure of the Bayesian network.
    - nodes (list): A list of nodes present in the graph.
    - n (int): The number of samples (rows) in the dataset.

    Methods:
    - compute(): Placeholder method for computing the score. Raises a NotImplementedError 
                indicating that this method should be implemented in derived classes.
    - getData(): Returns the dataset.
    - getGraph(): Returns the graph structure.
    - getNodes(): Returns the list of nodes in the graph.
    - getSampleSize(): Returns the number of samples in the dataset.
    """
    
    def __init__(self, df: pd.DataFrame, G : nx.DiGraph):
        """
        Initializes the Score class with a dataset and a graph structure.

        Parameters:
        - df (pd.DataFrame): The dataset.
        - G (nx.DiGraph): The graph structure of the Bayesian network.
        """
        self.df = df
        self.G = G
        self.nodes = list(G.nodes())
        self.n, _ = self.df.shape
    
    @abstractmethod
    def compute(self):
        """
        Placeholder method for computing the score. Must be implemented in derived classes.

        Raises:
        - NotImplementedError: Indicates that this method should be implemented in derived classes.
        """    
        raise NotImplementedError("The method compute() must be implemented in derived classes.")

    def getData(self):
        """
        Returns the dataset.

        Returns:
        - pd.DataFrame: The dataset.
        """
        return self.df
    
    def getGraph(self):
        """
        Returns the graph structure of the Bayesian network.

        Returns:
        - nx.DiGraph: The graph structure.
        """
        return self.G
    
    def getNodes(self):
        """
        Returns the list of nodes present in the graph.

        Returns:
        - list: The list of nodes.
        """
        return self.nodes
    
    def getSampleSize(self):
        """
        Returns the number of samples (rows) in the dataset.

        Returns:
        - int: The number of samples.
        """
        return self.n

# LOG MARGINAL LIKELIHOOD
#################################################################################
class LogMarginalLikelihood(Score):
    """
    Computes the log marginal likelihood of a Gaussian Bayesian network given 
    a dataset and a graph structure. This class extends the `Score` base class.

    Attributes:
    - a0 (float): The prior hyperparameter for the inverse gamma distribution.
    - b0 (float): The prior hyperparameter for the inverse gamma distribution.
    - Lambda0_scaler (float): A scaler for the prior precision matrix of the Gaussian distribution.

    Methods:
    - compute(): Computes the log marginal likelihood for the given graph structure and dataset.
    """
    
    def __init__(self, df: pd.DataFrame, G : nx.DiGraph, a0=1, b0=1, Lambda0_scaler=0.1):
        """
        Initializes the LogMarginalLikelihood class with a dataset, a graph structure, 
        and optional hyperparameters.

        Parameters:
        - df (pd.DataFrame): The dataset.
        - G (nx.DiGraph): The graph structure of the Bayesian network.
        - a0 (float, optional): Prior hyperparameter for the inverse gamma distribution. Default is 1.
        - b0 (float, optional): Prior hyperparameter for the inverse gamma distribution. Default is 1.
        - Lambda0_scaler (float, optional): Scaler for the prior precision matrix. Default is 0.1.
        """
        super().__init__(df, G)
        self.a0 = a0
        self.b0 = b0
        self.Lambda0_scaler = Lambda0_scaler
    
    
    def compute(self):
        """
        Computes the log marginal likelihood for the Bayesian network given the dataset 
        and graph structure.

        Returns:
        - float: The log marginal likelihood.
        - dict: Parameters for each node, including posterior precision matrix ('Lambda_n'), 
                posterior mean ('m_n'), and hyperparameters ('a_n' and 'b_n') for the inverse gamma distribution.
        """
        total_log_ml = 0
        parameters = {}              # Dictionary to store the parameters for each node
        
        N = self.getSampleSize()     # Number of data points

        # For each node in the graph
        for node in self.nodes:
            y = self.df[node].values    # Extract the data for the target node y
            
            # If the node has parents
            if self.G.in_degree(node) > 0:
                # Extract the data for the node and its parents
                X = self.df[list(self.G.predecessors(node))].values
            else:
                # For root nodes, X is just an intercept term
                X = np.ones((len(y), 1))
            
            # If the node is not a root node, add the intercept term
            if self.G.in_degree(node) > 0:
                X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
                
            p_node = X.shape[1]     # Number of predictors for this node + intercept
            
            # Setting up the Priors for beta
            Lambda0 = np.eye(p_node)*self.Lambda0_scaler   # Prior precision matrix
            m0 = np.zeros(p_node)                          # Prior mean vector
            
            # Bayesian Linear Regression
            # Compute the posterior precision matrix Lambda_n for beta
            Lambda_n = Lambda0 + X.T @ X

            # Compute the posterior mean m_n for beta
            beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
            
            m_n = np.linalg.inv(Lambda_n) @ (X.T @ X @ beta_hat + Lambda0 @ m0)

            # Compute a_n and b_n for sigma^2
            a_n = self.a0 + N/2
            b_n = self.b0 + 0.5 * (y.T @ y + m0.T @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n)
            
            # Save the parameters for the node
            parameters[node] = {
                'Lambda_n': Lambda_n,
                'm_n': m_n,
                'a_n': a_n,
                'b_n': b_n
            }

            # Compute the Marginal Likelihood for this node and add to total
            log_ml_node = ( - (len(y)/2) * np.log(2*np.pi) 
                            + 0.5 * (np.linalg.slogdet(Lambda0)[1] - np.linalg.slogdet(Lambda_n)[1]) 
                            + self.a0 * np.log(self.b0) - a_n * np.log(b_n) + gammaln(a_n) - gammaln(self.a0) )
            total_log_ml += log_ml_node
        
        return total_log_ml, parameters


# MARGINAL LIKELIHOOD
#################################################################################
class MarginalLikelihood(Score):
    """
    Computes the marginal likelihood of a Gaussian Bayesian network given 
    a dataset and a graph structure. This class extends the `Score` base class.

    The marginal likelihood is computed from the log marginal likelihood by taking its exponentiation.

    Attributes:
    - a0 (float): The prior hyperparameter for the inverse gamma distribution.
    - b0 (float): The prior hyperparameter for the inverse gamma distribution.
    - Lambda0_scaler (float): A scaler for the prior precision matrix of the Gaussian distribution.

    Methods:
    - compute(): Computes the marginal likelihood for the given graph structure and dataset.
    """
    def __init__(self, df: pd.DataFrame, G : nx.DiGraph, a0=1, b0=1, Lambda0_scaler=0.1):
        super().__init__(df, G)
        self.a0 = a0
        self.b0 = b0
        self.Lambda0_scaler = Lambda0_scaler
        
    def compute(self):
        lorMargScore = LogMarginalLikelihood(self.df, self.G, self.a0, self.b0, self.Lambda0_scaler)
        score, params = lorMargScore.compute()
        return np.exp(score), params

# LOG LIKELIHOOD
#################################################################################
class LogLikelihood(Score):
    pass

# LIKELIHOOD
#################################################################################
class Likelihood(Score):
    pass

# BGESCORE
#################################################################################
class BGeScore(Score):
    pass

# BDEU SCORE
#################################################################################
class BDeuScore(Score):
    pass

# BIC SCORE
#################################################################################
class BICScore(Score):
    pass

# AIC SCORE
#################################################################################
class AICScore(Score):
    pass
