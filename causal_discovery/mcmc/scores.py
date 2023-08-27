import pandas as pd
import numpy as np
import networkx as nx

from scipy.special import gammaln


def compute_log_marginal_likelihood(df: pd.DataFrame, G: nx.DiGraph, a0 = 1, b0 = 1, Lambda0_scaler = 0.1):
    n, _ = df.shape  

    total_log_ml = 0
    parameters = {}  # Dictionary to store the parameters for each node

    # Loop through each node in the graph
    for node in G.nodes():
        # Extract the data for the node
        y = df[node].values
        
        # If the node has parents
        if G.in_degree(node) > 0:
            # Extract the data for the node and its parents
            X = df[list(G.predecessors(node))].values
        else:
            # For root nodes, X is just an intercept term
            X = np.ones((len(y), 1))
        
        # If the node is not a root node, add the intercept term
        if G.in_degree(node) > 0:
            X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
            
        p_node = X.shape[1]     # Number of predictors for this node + intercept
        # Setting up the Priors for beta
        Lambda0 = np.eye(p_node)*Lambda0_scaler   # Prior precision matrix
        m0 = np.zeros(p_node)                     # Prior mean vector
        
        # Bayesian Linear Regression
        # Compute the posterior precision matrix Lambda_n for beta
        Lambda_n = Lambda0 + X.T @ X

        # Compute the posterior mean m_n for beta
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        
        m_n = np.linalg.inv(Lambda_n) @ (X.T @ X @ beta_hat + Lambda0 @ m0)

        # Compute a_n and b_n for sigma^2
        a_n = a0 + n/2
        b_n = b0 + 0.5 * (y.T @ y + m0.T @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n)
        
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
                        + a0 * np.log(b0) - a_n * np.log(b_n) + gammaln(a_n) - gammaln(a0) )
        total_log_ml += log_ml_node
    
    return total_log_ml, parameters

def compute_marginal_likelihood(df: pd.DataFrame, G: nx.DiGraph):
    marg, params = compute_log_marginal_likelihood(df, G)
    return np.exp(marg), params

def BGeScore():
    pass

def BIC_score(df: pd.DataFrame, G: nx.DiGraph):
    n, _ = df.shape
    
    # Compute the log-likelihood using the provided log-likelihood function
    log_likelihood = log_likelihood_bn(df, G)
    
    # Count the number of parameters in the model
    k = 0
    for node in G.nodes():
        num_parents = G.in_degree(node)
        
        # Add the number of regression coefficients for this node (including intercept)
        k += num_parents + 1
        
        # Add one for the error variance of this node
        k += 1

    # Compute the BIC score
    BIC = -2 * log_likelihood + k * np.log(n)
    
    return BIC
