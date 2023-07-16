import itertools

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

def remove_vars_from_list(list_nodes, nodes_to_remove):
    """This function takes two lists as input parameters. It returns a new list containing all nodes from list_nodes that are not in nodes_to_remove.
    
    Args:
        list_nodes (list): A list of nodes (strings) to be filtered.
        nodes_to_remove (list):  A list of nodes (strings) to be removed from list_nodes.
    
    Returns:
        list : A new list of strings containing all nodes from list_nodes that are not in nodes_to_remove.
    """
    return [node for node in list_nodes if node not in nodes_to_remove]

def generate_subsets( list_of_variables ):
    """This function takes a list of variables as input. It generates all possible subsets of the input list using the combinations() function from the itertools module. The function returns a list of tuples, where each tuple represents a subset of the input list.
    
    Args:
        list_of_variables (list): A list of variables for which subsets are to be generated.

    Returns:
        list: A list of tuples, where each tuple represents a subset of the input list.
    """
    combinations = []
    for i in range( 1, len( list_of_variables ) + 1 ):
        combinations += list( itertools.combinations( list_of_variables, i ) )
    return combinations

def plot_skeleton( graph : nx.classes.graph.Graph, label ):
    """Plots a graph with the title label.

    Args:
        graph (nx.classes.graph.Graph): A NetworkX graph object representing the skeleton to be plotted.
        label (str): A string representing the title of the plot.
    """
    fig, ax = plt.subplots()
    nx.draw(graph, with_labels=True, ax=ax)
    ax.set_title(label)
    plt.show()

