import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()



def plot_graph(G, title="Graph", node_size = 2000, figsize=(4,4)):
    
    plt.figure(figsize=figsize)
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

def plot_histogram(samples, title = "Histogram and Density Plot", xlabel = "x-axis", figsize=(10,6)):
    
    plt.figure(figsize=figsize)

    sns.histplot(samples, kde=True, bins=30, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    
def log_marginal_likelihood_plot( all_dags : dict):
    
    # Extracting keys and scores
    keys = list(all_dags.keys())
    scores = [all_dags[key]["log_score"] for key in keys]

    # Plotting
    plt.figure(figsize=(10,6))
    plt.bar(keys, scores, color='skyblue')
    plt.xlabel('Graph Index')
    plt.ylabel('marginal loglikelihood scaled and normalised')
    plt.title('Scores of different DAGs')
    plt.grid(False)
    plt.show()


def marginal_likelihood_plot( all_dags : dict, figsize=(10,6)):
    # Extracting keys and scores
    keys = list(all_dags.keys())
    scores = [all_dags[key]["score_normalised"] for key in keys]

    # Plotting
    plt.figure(figsize=figsize)
    plt.bar(keys, scores, color='skyblue')
    plt.xlabel('Graph Index')
    plt.ylabel('marginal likelihood scaled & normalised')
    plt.title('Scores of different DAGs')
    plt.grid(False)
    plt.show()
    
