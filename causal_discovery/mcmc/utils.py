from scipy.stats import invgamma, multivariate_normal
import networkx as nx
import numpy as np

def sample_parameters(parameters):
    """
    Samples parameters for each node in a Bayesian network from their respective 
    posterior distributions.

    Given the posterior parameters for each node in a Bayesian network, this function 
    samples the regression coefficients (beta) from a multivariate normal distribution 
    and the error variance (sigma^2) from an inverse gamma distribution.

    Parameters:
    - parameters (dict): A dictionary containing the posterior parameters for each node.
                    Each entry is of the form:
                            node: {
                                'Lambda_n': [np.array] Posterior precision matrix,
                                'm_n': [np.array] Posterior mean vector,
                                'a_n': [float] Posterior shape parameter for sigma^2,
                                'b_n': [float] Posterior scale parameter for sigma^2
                            }

    Returns:
    - dict: A dictionary containing the sampled values for each node. Each entry is of the form:
                node: {
                    'beta': [np.array] Sampled regression coefficients,
                    'sigma2': [float] Sampled error variance
                } 
    """
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

def compute_parent_dict(graph):
    parent_dict = {}
    
    for node in graph.nodes():
        parent_dict[node] = list(graph.predecessors(node))
    return parent_dict

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

def get_max_score_key(scores_dict):
    """Return the key corresponding to the maximum score."""
    return max(scores_dict, key=scores_dict.get)