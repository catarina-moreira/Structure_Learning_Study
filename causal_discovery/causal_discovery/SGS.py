from src.StructureLearning import StructureLearningAlgorithm

import networkx as nx
import pandas as pd
import numpy as np

from src.util import generate_subsets, remove_vars_from_list
from src.statistical_tests import test_conditional_independence

class SGS(StructureLearningAlgorithm):
    """
    Spirtes-Glymour-Scheines (SGS) algorithm for learning causal structures.
    """

    def fit(self, data):
        
        # init - start with a fully connected graph
        self.graph = nx.complete_graph( data.columns )
        
        # Skeleton Identification
        self.skeleton_phase(data)

        # V-structure Orientation
        self.v_structure_phase(data)

        # Remaining Edge Orientation
        self.orientation_phase()

        return self
    
    def skeleton_phase(self, data : pd.DataFrame, verbose=False ):
        
        [self.dag.add_node(node) for node in  self.graph.nodes()]
        self.init_debug( "skeleton_phase" )
        
        #For each pair of variables X and Y in the graph
        for node1 in self.graph.nodes:
            for node2 in self.graph.nodes:
                
                if node1 == node2:
                    continue
                        
                # Check conditional independence
                # If X and Y are conditionally independent given some subset of the remaining variables, 
                # then remove the edge between X and Y 
                subsets = generate_subsets( remove_vars_from_list(list(self.graph.nodes()), [node1, node2]) )
                for node3 in subsets:
                    
                    independent = test_conditional_independence(data, node1, node2, list(node3), verbose=verbose)
                    [print(f"\tPr({node1} || {node2}) | {list(node3)}: {independent}") if independent and verbose else None]
                    [print(f"----------------------------------------") if independent and verbose else None]
                    
                    if independent:
                        if self.graph.has_edge(node1, node2):
                            self.graph.remove_edge(node1, node2)
                            self.debug_graph["skeleton_phase"].append(self.graph.copy())
    
        return self

    def v_structure_phase(self, data : pd.DataFrame, verbose=False ):
        
        [print("Entered v-structure phase") if verbose else None]
        
        self.init_debug( "v-structure_phase" )

        # Identify all triples (X, Y, Z) that form a V-structure X -> Y <- Z
        for X in self.graph.nodes():
            for Y in self.graph.nodes():
                for Z in self.graph.nodes():
                    
                    if X == Y or X == Z or Y == Z:
                        continue
                    
                    if self.graph.has_edge(X, Z) or test_conditional_independence(data, X, Z, [Y], verbose=verbose):
                        continue
                    
                    if not self.graph.has_edge(X, Y):
                        self.graph.add_edge(X, Y, arrowhead='v-struct')
                        self.debug_graph["v-structure_phase"].append(self.graph.copy())
                    
                    if not self.dag.has_edge(X, Y):
                        self.dag.add_edge(X, Y)
                        self.debug_dag["v-structure_phase"].append(self.dag.copy()) 
                    
                    if not self.graph.has_edge(Z, Y):
                        self.graph.add_edge(Z, Y, arrowhead='v-struct')
                        self.debug_graph["v-structure_phase"].append(self.graph.copy())
                    
                    if not self.dag.has_edge(Z, Y):
                        self.dag.add_edge( Z, Y)
                        self.debug_dag["v-structure_phase"].append(self.dag.copy())     
        return self

    def orientation_phase(self, verbose=False ):
        
        self.init_debug( "orientation_phase" )
        while True:
            
            change = False
            for edge in list(self.graph.edges):
                
                node1, node2 = edge
                
                # if this edge is already oriented, skip it
                if 'arrowhead' in self.graph[node1][node2]:
                    continue
                
                # for all nodes connected to node1 that are not node2
                for node3 in self.graph.neighbors(node1):
                    if node3 == node2 or not self.graph.has_edge(node1, node3):
                        continue
                    
                    # check if there is a v-structure connecting node1 and node3
                    # then add an edge node1 -> node2
                    if 'arrowhead' in self.graph[node1][node3] and self.graph[node1][node3]['arrowhead'] == 'v-struct':
                        self.graph.add_edge(node1, node2, arrowhead='v-struct')
                        self.debug_graph["orientation_phase"].append(self.graph.copy())
                        self.dag.add_edge(node1, node2)
                        self.debug_dag["orientation_phase"].append(self.dag.copy()) 
                        change = True    
                        break
                    
                if change:
                    continue
                
                # for all nodes connected to node1 that are not node2
                for node3 in self.graph.neighbors(node2):
                    if node3 == node1 or not self.graph.has_edge(node2, node3):
                        continue
                    
                    if 'arrowhead' in self.graph[node2][node3] and self.graph[node2][node3]['arrowhead'] == 'v-struct':
                        self.graph.add_edge(node2, node1, arrowhead='v-struct')
                        self.debug_graph["orientation_phase"].append(self.graph.copy())
                        self.dag.add_edge(node2, node1)
                        self.debug_dag["orientation_phase"].append(self.dag.copy()) 
                        change = True
                        break
                    
            # convergence reached
            if not change:
                break
    
        return self

    def get_structure(self):
        return self.dag
    
    def get_debug_dag(self):
        return self.debug_dag
    
    def get_debug_graph(self):
        return self.debug_graph
    
    def init_debug(self, phase):
        self.debug_graph[phase] = list()
        self.debug_graph[phase].append(self.graph.copy())
        
        self.debug_dag[phase] = list()
        self.debug_dag[phase].append(self.dag.copy())
        