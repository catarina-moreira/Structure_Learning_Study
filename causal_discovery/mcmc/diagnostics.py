import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

def trace_plot( scores, burnit, withAcceptedGraphs = False, figsize=(15, 5)):
    fig, ax = plt.subplots()
    plt.figure(figsize=figsize)
    
    data = scores["marginal_likelihood"]
    ax.plot(data[burnit:])

    if withAcceptedGraphs:
        for i in scores["accepted_iterations"]:
            #plot the accepted iterations in red
            ax.plot(i, scores["marginal_likelihood"][i], "ro")
        
    ax.set_title("Trace plot of the log marginal likelihood")
    ax.set_xlabel("Step")
    ax.set_ylabel("Log marginal likelihood")
    plt.tight_layout()
    plt.show()
    
    
def edges_trace_plot( graphs, figsize=(15, 5)):
    # Compute the number of edges in each graph
    num_edges = [G.number_of_edges() for G in graphs]

    plt.figure(figsize=figsize)
    plt.plot(num_edges, linewidth=0.5 )

    plt.title("Trace plot of the number of edges")
    plt.xlabel("Step")
    plt.ylabel("Number of edges")
    plt.grid(False)
    
    plt.tight_layout()
    plt.show()
    
def mcmc_graph_distribution_plot( freq_dict, figsize = (15,7) ):
    
    max_labels = []
    
    # assume freq_dict is the dictionary with the frequencies
    labels = list(freq_dict.keys())
    values = list(freq_dict.values())

    # find the maximum frequency and get the label
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

def mcmc_edge_frequency_heatmap(dags : list, figsize= (8,8)):
    
    nodes = list( dags[0].nodes())
    
    num_nodes = len(nodes)
    frequency_matrix = np.zeros((num_nodes, num_nodes))
    
    for G in dags:
        for edge in G.edges():
            source, target = edge
            frequency_matrix[nodes.index(source), nodes.index(target)] += 1
    
    # Normalize by the number of samples to get frequencies
    frequency_matrix /= len(dags)
    
    # Visualize as heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(edge_frequencies, annot=True, cmap="YlGnBu", xticklabels=nodes_list, yticklabels=nodes_list)
    plt.title("Edge Frequencies from Sampled DAGs")
    plt.show()
    
    
    
    
def edge_probability_over_iterations( dags, edge, figsize = (10,6)):
    """
    Compute the probability of a specific edge over the MCMC iterations.
    
    Parameters:
    - dags: List of nx.DiGraphs representing the sampled graphs over MCMC iterations.
    - edge: Tuple representing the edge of interest.
    
    Returns:
    - List of edge probabilities over the MCMC iterations.
    """
    edge_probs = [1 if G.has_edge(*edge) else 0 for G in dags]
    final_eddge_prob = np.cumsum(edge_probs) / np.arange(1, len(dags) + 1)

    edge_probs_over_iterations = edge_probability_over_iterations(sampled_dags, edge)

    plt.figure(figsize=figsize)
    plt.plot(edge_probs_over_iterations, color='blue', alpha=0.7)
    plt.xlabel("MCMC Iteration")
    plt.ylabel(f"Probability of edge {edge}")
    plt.title(f"Evolution of Probability for Edge {edge} over MCMC Iterations")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
def gelman_rubin_diagnostic( chains):
    """
    Compute the Gelman-Rubin Diagnostic R-hat statistic for MCMC chains.
    
    Parameters:
    - chains: List of chains, where each chain is a list of parameter values over iterations.
    
    Returns:
    - R-hat statistic.
    """
    # Number of chains and iterations per chain
    m = len(chains)
    n = len(chains[0])
    
    # Chain means
    chain_means = np.mean(chains, axis=1)
    
    # Overall mean
    overall_mean = np.mean(chain_means)
    
    # Between-chain variance B
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    
    # Within-chain variance W
    W = 1 / m * np.sum(np.var(chains, axis=1, ddof=1))
    
    # Estimated variance of the target distribution
    V_hat = (1 - 1/n) * W + 1/n * B
    
    # R-hat statistic
    R_hat = np.sqrt(V_hat / W)
    
    return R_hat

