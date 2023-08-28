import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

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
    

def plot_edges_trace(graphs, figsize=(15, 5)):
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



