import networkx as nx
import random

import numpy as np
import pandas as pd

import itertools

from mcmc.scores import LogMarginalLikelihood

def possible_edges_for_dag(G : nx.DiGraph):
    """
    Find all possible edges that can be added to a directed acyclic graph (DAG)
    without creating a cycle.

    Parameters:
    - G (nx.DiGraph): A directed acyclic graph.

    Returns:
    - List[Tuple]: A list of possible edges (as tuples) that can be added 
                to the DAG without creating a cycle.
    """
    possible_edges = []

    # Check all possible edges
    for u in G.nodes():
        for v in G.nodes():
            if u != v and not G.has_edge(u, v):
                # Temporarily add the edge
                G.add_edge(u, v)
                
                # Check for cycles using simple_cycles
                if any(nx.simple_cycles(G)):
                    G.remove_edge(u, v)  # Remove the edge if it causes a cycle
                else:
                    possible_edges.append((u, v))
                    G.remove_edge(u, v)  # Remove the edge to continue checking other possibilities
    return possible_edges

def total_possible_add_edges(G : nx.DiGraph ):
    """
    Computes the number of edges that can be added to the graph without introducing a cycle.

    The function iterates through all pairs of nodes in the graph, 
    and for each pair that doesn't already have an edge, it temporarily adds an edge 
    and checks if the graph remains acyclic. If the graph remains a DAG, the count of 
    possible additions is incremented.

    Parameters:
    - G (nx.DiGraph): The input directed graph.

    Returns:
    - int: The number of possible edge additions that would not introduce a cycle in the graph.
    """
    # Compute possible adds (all non-existing edges minus those that create cycles)
    possible_adds = 0
    for u in G.nodes():
        for v in G.nodes():
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                if not nx.is_directed_acyclic_graph(G):
                    G.remove_edge(u, v)
                else:
                    possible_adds += 1
                    G.remove_edge(u, v)
    return possible_adds

def total_possible_remove_edges(G : nx.DiGraph ):
    # Compute possible removes (simply all existing edges)
    """
    Computes the number of edges that can be removed from the graph.

    Since any edge in the graph can be removed without restrictions, 
    this function simply returns the current number of edges in the graph.

    Parameters:
    - G (nx.DiGraph): The input directed graph.

    Returns:
    - int: The number of edges in the graph.
    """
    # Compute possible removes (simply all existing edges)
    return len(G.edges())
    


def total_possible_reverse_edges(G : nx.DiGraph ):
    """
    Computes the number of edges in the graph that can be reversed without introducing a cycle.

    The function iterates through all the edges in the graph. 
    For each edge, it temporarily reverses the edge and checks if the resulting graph 
    remains acyclic. If the graph remains a DAG after the reversal, the count of possible 
    reversals is incremented.

    Parameters:
    - G (nx.DiGraph): The input directed graph.

    Returns:
    - int: The number of edges that can be reversed without introducing a cycle in the graph.
    """
    # Compute possible reverses (all edges whose reversal doesn't introduce a cycle)
    possible_reverses = 0
    edges_list = list(G.edges())
    for (u, v) in edges_list:
        G.remove_edge(u, v)
        G.add_edge(v, u)
        if nx.is_directed_acyclic_graph(G):
            possible_reverses += 1
        G.remove_edge(v, u)
        G.add_edge(u, v)
    return possible_reverses

def total_possible_reverse_edges(G : nx.DiGraph ):
    """
    Computes the number of edges in the graph that can be reversed without introducing a cycle.

    The function iterates through all the edges in the graph. 
    For each edge, it temporarily reverses the edge and checks if the resulting graph 
    remains acyclic. If the graph remains a DAG after the reversal, the count of possible 
    reversals is incremented.

    Parameters:
    - G (nx.DiGraph): The input directed graph.

    Returns:
    - int: The number of edges that can be reversed without introducing a cycle in the graph.
    """
    # Compute possible reverses (all edges whose reversal doesn't introduce a cycle)
    possible_reverses = 0
    edges_list = list(G.edges())
    for (u, v) in edges_list:
        G.remove_edge(u, v)
        G.add_edge(v, u)
        if nx.is_directed_acyclic_graph(G):
            possible_reverses += 1
        G.remove_edge(v, u)
        G.add_edge(u, v)
        
    return possible_reverses
    
    
def propose_new_graph(G: nx.DiGraph):
    """
    Propose a new graph structure based on a given directed acyclic graph (DAG) 
    by performing one of three operations: adding an edge, deleting an edge, 
    or reversing an existing edge.

    Args:
        G (nx.DiGraph): A directed acyclic graph.

    Returns:
        - nx.DiGraph: The proposed graph after performing one of the operations.
        - str: The operation performed ("add_edge", "delete_edge", or "reverse_edge").

    """
    proposed_graph = G.copy()
    
    # List all possible edges that could be added to the graph without creating a cycle
    possible_edges = possible_edges_for_dag(proposed_graph)
    
    # Select an operation: add, remove, or reverse
    operations = ["add_edge", "delete_edge", "reverse_edge"]
    
    # If no edges, the only valid operation is "add"
    if len(G.edges()) == 0:  
        operation = "add_edge"
        
        # If graph is fully connected, the only valid operation is "delete"
    elif len(possible_edges) == 0:  
        operation = "delete_edge"
    else:
        # otherwise, randomly select an operation
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
    
    # todo: call the input function with the edges that led to a cycle and do not consider them in the future
    # check if proposed graph is a dag
    if not nx.is_directed_acyclic_graph(proposed_graph):
        propose_new_graph(G) # if it is not a dag, start over

    return proposed_graph, operation


def generate_random_dag(node_labels: list):
    """
    Generates a random Directed Acyclic Graph (DAG) using the given node labels.

    The function works by iterating over all pairs of nodes, and with a 50\% chance, 
    adds a directed edge from the earlier node to the later node in the list. 
    This ensures that the resulting graph is acyclic since edges always move in one direction 
    based on the ordering of nodes in the input list.

    Parameters:
    - node_labels (list): A list of node labels that will be used for the nodes of the graph.

    Returns:
    - nx.DiGraph: A randomly generated Directed Acyclic Graph with nodes labeled as per the input list.

    Note:
    - The resulting graph is not guaranteed to be connected. Some nodes might be isolated or disconnected from others.
    - The randomness comes from the 50\% chance of adding an edge between any pair of nodes in the specified order.
    """
    G = nx.DiGraph()
    num_nodes = len(node_labels)
    
    # Add nodes with specified labels
    for label in node_labels:
        G.add_node(label)
    
    # Randomly add edges
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):     # Ensure that we only move forward to prevent cycles
            if random.random() > 0.5:       # 50% chance to add an edge
                G.add_edge(node_labels[i], node_labels[j])
    
    return G


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
                if len( G.nodes() ) < N:
                    for n in nodes:
                        if not G.has_node( n ):
                            G.add_node( n )
                    
                all_dags[str(indx)] = {}
                all_dags[str(indx)]["DAG"] = G
                score = LogMarginalLikelihood(df, G)
                log_score, params = score.compute()
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
        all_dags[dag]["score_normalised"] = np.round(all_dags[dag]["score"] / total_score, 6)
        
    # sanity check: check if the normalised score sums to 1
    total_score_normalised = 0
    for dag in all_dags.keys():
        total_score_normalised = total_score_normalised + all_dags[dag]["score_normalised"]
    print(f"total_score_normalised = {np.round(total_score_normalised, 6)}")
        
    return all_dags, total_score

def generate_graph_distribution(graph_list : list, data : pd.DataFrame):
    graph_to_id = {}
    id_to_freq = {}
    id_to_graph = {}
    id_to_score = {}
    current_id = 0

    for graph in graph_list:
        score = LogMarginalLikelihood( data, graph)
        graph_hash = hash(tuple(sorted(graph.edges())))
        
        # If this graph has not been seen before, assign a new ID
        if graph_hash not in graph_to_id:
            graph_to_id[graph_hash] = f"G_{current_id}"
            id_to_freq[f"G_{current_id}"] = 1
            id_to_graph[f"G_{current_id}"] = graph
            id_to_score[f"G_{current_id}"] = score.compute()
            current_id += 1
        else:
            id_to_freq[graph_to_id[graph_hash]] += 1
            
    return id_to_freq, id_to_graph, id_to_score

