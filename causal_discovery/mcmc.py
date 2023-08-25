import pandas as pd
import numpy as np
import networkx as nx
import arviz as az

import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.linear_model import LinearRegression

import itertools
import random

import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

from scipy.special import gamma
from scipy.special import gammaln

import xarray as xr


from scipy.linalg import block_diag

import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.estimators import ParameterEstimator, BicScore

from scipy.special import gammaln, multigammaln
from numpy.linalg import det, inv

from itertools import chain, combinations

from scipy.stats import invgamma, multivariate_normal

def plot_graph_distribution( freq_dict, figsize = (15,7) ):
    
    max_labels = []
    
    # assume freq_dict is the dictionary with the frequencies
    labels = list(freq_dict.keys())
    values = list(freq_dict.values())

    # find the maxumum frequency and get the label
    max_freq = max(values)
    max_label = labels[values.index(max_freq)]
    
    plt.figure(figsize=figsize)
    
    # create a bar plot
    plt.bar(labels, values)
    
    # if there are other bars with the same frequency, change their color
    for i in range(len(labels)):
        if values[i] == max_freq:
            plt.bar(labels[i], values[i], color='red')
            max_labels.append(labels[i])

    # set the title and axis labels
    plt.title("Graph Distribution. MAX = " + str(max_label) + " with prob " + str(max_freq))
    plt.xlabel("Graph")
    plt.ylabel("Probability")
    
    # rotate x labels
    plt.xticks(rotation=90)

    # no grid
    plt.grid(False)

    # show the plot
    plt.tight_layout()
    plt.show()
    
    return max_labels, max_freq
    

# normalise the frequencies of Gs    
def normalise_Gs(Gs):
    Gs_copy = Gs.copy()
    
    total = np.sum(list(Gs_copy.values()))
    for i in Gs_copy.keys():
        Gs_copy[i] = Gs_copy[i] / total
    return Gs_copy

# add noise to data
def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def plot_edges_trace(graphs, figsize=(15, 5)):
    # Compute the number of edges in each graph
    num_edges = [G.number_of_edges() for G in graphs]

    plt.figure(figsize=figsize)
    plt.plot(num_edges)

    plt.title("Trace plot of the number of edges")
    plt.xlabel("Step")
    plt.ylabel("Number of edges")
    plt.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    
def plot_trace(posteriors, burnit, figsize=(15, 5)):
    fig, ax = plt.subplots()
    plt.figure(figsize=figsize)
    
    data = posteriors["marginal_likelihood"]
    ax.plot(data[burnit:])

    #for i in posteriors["accepted_iterations"]:
    #    # plot the accepted iterations in red
    #    ax.plot(i, posteriors["marginal_likelihood"][i], "ro")
        
    ax.set_title("Trace plot of the log marginal likelihood")
    ax.set_xlabel("Step")
    ax.set_ylabel("Log marginal likelihood")
    plt.tight_layout()
    plt.show()


def plot_graph(G, title="Graph", node_size = 2000):
    nx.draw(G, with_labels=True, arrowsize=20, node_size=node_size, node_color="skyblue", pos=nx.spring_layout(G))
    ax = plt.gca()
    ax.margins(0.20)
    ax.set_title(title)
    plt.axis("off")
    plt.show()
    
def get_max_score_key(scores_dict):
    """Return the key corresponding to the maximum score."""
    return max(scores_dict, key=scores_dict.get)




def plot_graph_transition(G1, G2, iteration):
    operation = None
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    G_out = G2.copy()
    
    # Draw nodes
    pos = nx.spring_layout(G_out)
    nx.draw_networkx_nodes(G_out, pos, ax=ax, node_size=1000, node_color='lightblue', alpha=0.8, linewidths=0.5)
    nx.draw_networkx_labels(G_out, pos, ax=ax, font_size=15)
    
    # Identify removed edges
    removed_edges = set(G1.edges()) - set(G2.edges())
    if len(removed_edges) > 0:
        operation = "Removing an edge"
        # add the removed edges to the graph
        for edge in removed_edges:
            G_out.add_edge(edge[0], edge[1])
            nx.draw_networkx_edges(G_out, pos, ax=ax, edgelist=removed_edges, width=2.5, edge_color='red', style='dashed', arrowsize=20)

    # Identify added edges
    added_edges = set(G2.edges()) - set(G1.edges())
    if len(added_edges) > 0:
        operation = "Adding an edge"
        nx.draw_networkx_edges(G_out, pos, ax=ax, edgelist=added_edges, width=2.5, edge_color='green', arrowsize=20)
    
    # Identify reversed edges
    reversed_edges = {(v, u) for u, v in removed_edges if (v, u) in added_edges}
    if len(reversed_edges) > 0:
        operation = "Adding an edge"
        nx.draw_networkx_edges(G_out, pos, ax=ax, edgelist=reversed_edges, width=2.5, edge_color='yellow', style='dashed', arrowsize=20)

    nx.draw_networkx_edges(G_out, pos, ax=ax, edgelist=list(set(G_out.edges()) - added_edges - reversed_edges), width=1.5, edge_color='gray', arrowsize=20)
    
    # Set title and axis labels
    ax.set_title(f"G_{iteration - 1} -> G_{iteration} by {operation}", fontsize=20)
    plt.grid(False)
    ax.margins(0.20)
    plt.show()
    
    return G_out

def generate_all_dags(df: pd.DataFrame):
    
    N = df.shape[1]
    nodes = list(df.columns)
    
    # Generate all possible directed edges among the nodes
    all_possible_edges = list(itertools.permutations(nodes, 2))

    # Dictionary to store all unique DAGs
    all_dags = {}
    indx = 0
    
    total_score = 0
    
    # Iterate over the subsets of all possible edges to form directed graphs
    for r in range(len(all_possible_edges)+1):
        for subset in itertools.combinations(all_possible_edges, r):
            G = nx.DiGraph(list(subset))
            if nx.is_directed_acyclic_graph(G):
                if indx == 0:
                    # create a graoh with N nodes
                    [ G.add_node(nodes[i]) for i in range(N) ]
                
                # check if G has N nodes, if not, add the remaining missing nodes
                #if len( G.nodes() ) < N:
                #    for n in nodes:
                #        if not G.has_node( n ):
                #            G.add_node( n )
                    
                all_dags[str(indx)] = {}
                all_dags[str(indx)]["DAG"] = G
                log_score, params = compute_log_marginal_likelihood(df, G) 
                all_dags[str(indx)]["log_score"] = log_score
                all_dags[str(indx)]["params"] = params
                indx += 1
                
    # get the max score
    min_score = min([ all_dags[id]["log_score"] for id in all_dags.keys() ])
    max_score = max([ all_dags[id]["log_score"] for id in all_dags.keys() ])
    
    print(f"min score {min_score}")
    print(f"max score {max_score}")
    
    # since the marginal likelihood grows very fast, we need to normalise the scores
    # we will subtract the max score from all scores and then divide by the total score
    for id in all_dags.keys():
        all_dags[id]["log_score"] = all_dags[id]["log_score"]  - max_score
        all_dags[id]["score"] = np.exp( all_dags[id]["log_score"]  )
        total_score = total_score + all_dags[id]["score"]
    
    # iterate of the dags and normalise the scores
    for dag in all_dags.keys():
        all_dags[dag]["score_normalised"] = all_dags[dag]["score"] / total_score
        
    # sanity check: check if the normalised score sums to 1
    total_score_normalised = 0
    for dag in all_dags.keys():
        total_score_normalised = total_score_normalised + all_dags[dag]["score_normalised"]
    print(f"total_score_normalised = {total_score_normalised}")
        
    return all_dags, total_score



def compute_log_marginal_likelihood(df: pd.DataFrame, G: nx.DiGraph):
    n, _ = df.shape  

    # For sigma^2
    a0 = 1
    b0 = 1

    total_log_ml = 0
    parameters = {}  # Dictionary to store the parameters for each node

    N = len(df) 
    
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
        Lambda0 = np.eye(p_node)*0.1   # Prior precision matrix
        m0 = np.zeros(p_node)     # Prior mean vector
        
        # Bayesian Linear Regression
        # Compute the posterior precision matrix Lambda_n for beta
        Lambda_n = Lambda0 + X.T @ X

        # Compute the posterior mean m_n for beta
        m_n = np.linalg.inv(Lambda_n) @ (Lambda0 @ m0 + X.T @ y)

        # Compute a_n and b_n for sigma^2
        a_n = a0 + N/2
        b_n = b0 + 0.5 * ( y.T @ y + m0 @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n  )
        
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
    return np.exp(compute_log_marginal_likelihood(df, G))



def sample_parameters(parameters):
    sampled_values = {}
    for node, param in parameters.items():
        # Sample sigma^2 from the inverse gamma distribution
        sigma2 = invgamma.rvs(a=param['a_n'], scale=param['b_n'])

        # Sample beta from the multivariate normal distribution
        cov_matrix = sigma2 * np.linalg.inv(param['Lambda_n'])
        beta = multivariate_normal.rvs(mean=param['m_n'], cov=cov_matrix)

        sampled_values[node] = {
            'beta': beta,
            'sigma2': sigma2
        }

    return sampled_values


def log_likelihood_bn(df, G):
    total_log_likelihood = 0.0
    
    # For each node in the graph
    for node in G.nodes():
        # Extract the data for the node
        y = df[node].values
        
        # If the node has parents (is not a root)
        if G.in_degree(node) > 0:
            # Extract the data for the node's parents
            X = df[list(G.predecessors(node))].values
            
            # Fit a linear regression model
            reg = LinearRegression().fit(X, y)
            
            # Compute the predicted values
            y_pred = reg.predict(X)
            
            # Compute the residuals
            residuals = y - y_pred
            
            # Compute the variance of residuals (our estimated sigma^2)
            sigma2 = np.var(residuals)
            
            # Compute the log likelihood for this node
            node_log_likelihood = (-0.5 * np.log(2 * np.pi * sigma2)) - (residuals**2 / (2 * sigma2))
            
            # Add to the total log likelihood
            total_log_likelihood += np.sum(node_log_likelihood)
            
        # If the node is a root, we assume a simple Gaussian likelihood
        else:
            mean = np.mean(y)
            sigma2 = np.var(y)
            
            root_log_likelihood = (-0.5 * np.log(2 * np.pi * sigma2)) - ((y - mean)**2 / (2 * sigma2))
            total_log_likelihood += np.sum(root_log_likelihood)
    
    return total_log_likelihood


def compute_parent_dict(graph):
    parent_dict = {}
    
    for node in graph.nodes():
        parent_dict[node] = list(graph.predecessors(node))
    return parent_dict


def propose_new_graph(G: nx.DiGraph):
    # Create a deep copy of the graph to modify and return
    proposed_graph = G.copy()
    
    plot_graph(proposed_graph)
    
    # List all the nodes
    nodes = list(G.nodes())
    
    
    # List all possible edges that could be added to the graph
    possible_edges = [(i, j) for i in nodes for j in nodes if i != j and not G.has_edge(i, j)]
    
    # Select an operation: add, remove, or reverse
    operations = ["add_edge", "delete_edge", "reverse_edge"]
    
    if len(G.edges()) == 0:  # If no edges, the only valid operation is "add"
        operation = "add_edge"
    elif len(possible_edges) == 0:  # If graph is fully connected, the only valid operation is "remove"
        operation = "delete_edge"
    else:
        operation = random.choice(operations)
    
    if operation == "add_edge":
        edge_to_add = random.choice(possible_edges)
        proposed_graph.add_edge(*edge_to_add)
    elif operation == "delete_edge":
        edge_to_remove = random.choice(list(G.edges()))
        proposed_graph.remove_edge(*edge_to_remove)
    else:  # reverse
        edge_to_reverse = random.choice(list(G.edges()))
        proposed_graph.remove_edge(*edge_to_reverse)
        proposed_graph.add_edge(edge_to_reverse[1], edge_to_reverse[0])
        
        # Check for cycles and revert the operation if a cycle is introduced
        if not nx.is_directed_acyclic_graph(proposed_graph):
            proposed_graph.remove_edge(edge_to_reverse[1], edge_to_reverse[0])
            proposed_graph.add_edge(*edge_to_reverse)
    
    # check if proposed graph is a dag
    if not nx.is_directed_acyclic_graph(proposed_graph):
        propose_new_graph(G) # if it is not a dag, start over
    
    plot_graph(proposed_graph)
    
    return proposed_graph, operation

def Q_G_to_G_proposed(G: nx.DiGraph, operation: str):
    """
    Compute the forward proposal probability Q(G_prime | G) based on the transition from G to G_prime and the operation.
    Returns the forward proposal probability.
    """
    
    # print("Q(G -> G')")
    nodes = list(G.nodes())
    max_possible_edges = len(nodes) * (len(nodes) - 1)
    
    len_edges = len(G.edges())
    if len_edges == 0:
        return 1
    
    if operation == 'add_edge':
        possible_edges_to_add_G = max_possible_edges - len(G.edges())
        forward_prob = 1 / possible_edges_to_add_G
        
    elif operation == 'delete_edge':
        forward_prob = 1 / len(G.edges())
    
    elif operation == 'reverse_edge':
        forward_prob = 1 / len(G.edges())

    return forward_prob

def Q_G_proposed_to_G(G_prime: nx.DiGraph, operation: str):
    """
    Compute the backward proposal probability Q(G | G_prime) based on the transition from G_prime to G and the operation.
    Returns the backward proposal probability.
    """
    nodes = list(G_prime.nodes())
    max_possible_edges = len(nodes) * (len(nodes) - 1)
    
    len_edges = len(G_prime.edges())
    if len_edges == 0:
        return 1
    
    if operation == 'add_edge':
        backward_prob = 1 / len(G_prime.edges())
        
    elif operation == 'delete_edge':
        possible_edges_to_add_G_prime = max_possible_edges - len(G_prime.edges())
        backward_prob = 1 / possible_edges_to_add_G_prime
        
    elif operation == 'reverse_edge':
        backward_prob = 1 / len(G_prime.edges())

    return backward_prob

def compute_non_symmetric_acceptance_ratio(G_current, posterior_G_current, G_proposed, posterior_G_proposed, operation_G_G_prime ):

    # Compute the proposal distributions at the current and proposed graphs
    #print("Q(G|G')")
    Q_G_proposed_given_G = Q_G_to_G_proposed( G_current, operation_G_G_prime )
    Q_G_given_G_proposed = Q_G_proposed_to_G( G_proposed, operation_G_G_prime )

    return min(1, (posterior_G_proposed * Q_G_given_G_proposed) / (posterior_G_current * Q_G_proposed_given_G))


def compute_non_log_symmetric_acceptance_ratio(G_current, log_marginal_likelihood_G_current, G_proposed, log_marginal_likelihood_G_proposed, operation_G_G_prime ):
    
    Q_G_proposed_given_G = Q_G_to_G_proposed( G_current, operation_G_G_prime )
    Q_G_given_G_proposed = Q_G_proposed_to_G( G_proposed, operation_G_G_prime )

    log_numerator = log_marginal_likelihood_G_proposed + np.log(Q_G_given_G_proposed)
    log_denominator = log_marginal_likelihood_G_current + np.log(Q_G_proposed_given_G)

    log_alpha = log_numerator - log_denominator
    
    return min(0, log_alpha)

def compute_symmetric_acceptance_ratio(posterior_G_current, posterior_G_proposed):
    return min(1, posterior_G_proposed / posterior_G_current)

def compute_acceptance_ratio(G_current, posterior_G_current, G_proposed, posterior_G_proposed, operation, is_proposal_symmetric):
    
    if is_proposal_symmetric:
        # A(G, G') = min(1, P(G' | D) / P(G | D))
        A = compute_symmetric_acceptance_ratio(posterior_G_current, G_proposed, posterior_G_proposed)
    else:
        #  A(G, G') = min(1, [P(G' | D) * Q(G | G')] / [P(G | D) * Q(G' | G)])
        A =  compute_non_symmetric_acceptance_ratio(G_current, posterior_G_current, G_proposed, posterior_G_proposed, operation)
    return A


def compute_log_acceptance_ratio(G_current, log_marginal_likelihood_G_current, G_proposed, log_marginal_likelihood_G_proposed, operation, is_proposal_symmetric):
    
    if is_proposal_symmetric:
        # A(G, G') = min(0, logP(G' | D) / P(G | D))
        A = compute_log_symmetric_acceptance_ratio(log_marginal_likelihood_G_current, G_proposed, log_marginal_likelihood_G_proposed)
    else:
        #  A(G, G') = min(0, log[P(G' | D) * Q(G | G')] / [P(G | D) * Q(G' | G)])
        A =  compute_non_log_symmetric_acceptance_ratio(G_current, log_marginal_likelihood_G_current, G_proposed, log_marginal_likelihood_G_proposed, operation)
    return A


def compute_score_function( data : pd.DataFrame, G : nx.DiGraph, score_type : str = "Marginal" ):
    
    if score_type == "BIC":
        return BIC_score(data, G)
    
    if score_type == "AIC":
        return AIC_score(data, G)
    
    if score_type == "BGe":
        return BGe_score(data, G)
    
    if score_type == "Log_Marginal_Likelihood":
        return compute_log_marginal_likelihood(data, G)
    
    res = -1
    try:
        res = compute_marginal_likelihood(data, G)
    except OverflowError: # when the marginal is too big, let's reject the graph
        res = -1
    return res

def graph_hash( G : nx.DiGraph ):
    matrix = nx.adjacency_matrix(G).todense()
    return hash(matrix.tobytes())
    
def initialize_structures(G_current, posterior_G_current):
    HASH_TO_ID = {graph_hash(G_current): "G_0"}
    ID_TO_GRAPH = {"G_0": G_current}
    ID_TO_FREQ = {"G_0": 1}
    ID_TO_MARGINAL = {"G_0": [posterior_G_current]}
    OPERATIONS = ["None"]
    COUNT = 1
    return ID_TO_GRAPH, HASH_TO_ID, ID_TO_FREQ, OPERATIONS, ID_TO_MARGINAL, COUNT

def update_structures(G_current, posterior_G_current, operation, ID_TO_GRAPH, HASH_TO_ID, ID_TO_FREQ, OPERATIONS, ID_TO_MARGINAL, COUNT, posterior_candidates):
    
    OPERATIONS.append(operation)
            
    # check if graph is already in the dictionary
    # if yes, increase the frequency by 1
    if graph_hash(G_current) in HASH_TO_ID.keys():
        ID = HASH_TO_ID[ graph_hash(G_current) ]
        val = ID_TO_FREQ[ ID ] 
        ID_TO_FREQ[ ID ] = val + 1
    else: # if not, add it to the dictionary
        HASH_TO_ID[ graph_hash(G_current) ] = "G_" + str(COUNT)
        ID_TO_GRAPH[ "G_" + str(COUNT)] = G_current
        ID_TO_FREQ[ "G_" + str(COUNT)] = 1
        ID_TO_MARGINAL["G_" + str(COUNT)] = [posterior_G_current] 
        COUNT = COUNT + 1

    return ID_TO_GRAPH, HASH_TO_ID, ID_TO_FREQ, OPERATIONS, ID_TO_MARGINAL, COUNT


### MCMC
##############################
def structured_MCMC(data : pd.DataFrame, initial_graph : nx.DiGraph, iterations : int, score_function : str,  restart_freq = 10, is_proposal_symmetric : bool = False):
    i = 0
    ACCEPT = 0
    ACCEPT_INDX = []
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
        # G_proposed = G_current.copy() # forcing the graph to stay in the same place
        # operation = "reverse_edge"
        G_proposed, operation = propose_new_graph(G_current)
        
        # check if G_proposed is a directed acyclic graph, if not, reject it
        if not nx.is_directed_acyclic_graph(G_proposed):
            continue
        
        # if the proposed graph is a DAG, compute the posterior
        posterior_G_proposed, params = compute_score_function( data, G_proposed, score_type = score_function )
        
        # Calculate the acceptance probability
        if "Log" in score_function:
            
            A = compute_log_acceptance_ratio(G_current, posterior_G_current, G_proposed, posterior_G_proposed, operation, is_proposal_symmetric)
            
            # Draw a random number in log space
            u = np.log(np.random.uniform(0, 1))
            
        else:
            A = compute_acceptance_ratio(G_current, posterior_G_current, G_proposed, posterior_G_proposed, operation, is_proposal_symmetric)

            # Generate a random number
            u = np.random.uniform(0, 1)
            
        # Accept if u is less than A
        # print(f"A = {np.round(A,4)} \t u = {np.round(u,4)} \t u < A = {u < A}")
        if u < A: 
            ACCEPT = ACCEPT + 1
            ACCEPT_INDX.append(i)
                
            # update the current graph and its posterior
            G_current = G_proposed.copy()
            posterior_G_current = posterior_G_proposed
                
            # add information to the data structures
            ID_TO_GRAPH, HASH_TO_ID, ID_TO_FREQ, OPERATIONS, ID_TO_MARGINAL, COUNT = update_structures(G_current, posterior_G_proposed, operation, ID_TO_GRAPH, HASH_TO_ID, ID_TO_FREQ, OPERATIONS, ID_TO_MARGINAL, COUNT, posterior_candidates)

        #if i % restart_freq == 0: # force the chain to jump
        #    index = np.random.randint(0, len(graph_candidates))
        #    G_current = graph_candidates[index]
        
        posterior_candidates.append(posterior_G_current)
        graph_candidates.append(G_current)
        PARAMS["G_" + str(i)] = params
        i = i + 1        
        
    return { "marginal_likelihood" : posterior_candidates, "acceptance_rate" : ACCEPT / iterations, "accepted_iterations" : ACCEPT_INDX}, graph_candidates, ID_TO_FREQ, ID_TO_GRAPH, ID_TO_MARGINAL, OPERATIONS

def generate_data(n_samples: int = 1000, intercept = 1, sigma = 0.2):
    # Generate independent variables
    
    # sample X2 from a normal distribution with center 1 and std 2

    X4 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    X3 = np.random.randn(n_samples)
    
    # Generate X1 as a linear combination of X2 and X3 plus some Gaussian noise
    beta2, beta3 = 0.8, 0.9  # Arbitrarily chosen coefficients
    noise = np.random.randn(n_samples)*sigma # Gaussian noise with standard deviation 0.2
    X1 = intercept + beta2 * X2 + beta3 * X3 + noise
    
    # Combine into a dataframe
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})
    
    return df

data2 = generate_data(n_samples = 2000, intercept=1, sigma=0.05)
data2.head()

DATA = data2

# create a dense directed acyclic graph
initial_graph = nx.DiGraph()
initial_graph.add_nodes_from(DATA.columns)
for i in range(0, len(DATA.columns)):
    for j in range(i+1, len(DATA.columns)):
        initial_graph.add_edge(DATA.columns[i], DATA.columns[j])
        
NUM_ITERATIONS = 10

print("Starting MCMC...")
posteriors, graph_candidates, Gs, graphs, marginals, operations = structured_MCMC(DATA, initial_graph, NUM_ITERATIONS, score_function = "Log_Marginal_Likelihood", restart_freq=100)
print("Finished MCMC")

print()
print("Average Log Marginal Likelihood: ", np.mean(posteriors["marginal_likelihood"]))
print("Acceptance rate: ", posteriors['acceptance_rate'])

