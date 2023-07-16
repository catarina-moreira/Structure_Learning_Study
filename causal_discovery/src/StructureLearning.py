from abc import ABC, abstractmethod

import networkx as nx


class StructureLearningAlgorithm(ABC):
    """
    Abstract base class for structure learning algorithms.
    """

    # define constructor
    def __init__(self):
        self.graph = nx.classes.graph.Graph()
        self.dag = nx.DiGraph()

    @abstractmethod
    def fit(self, data):
        """
        Fit the structure learning algorithm to the data.

        Parameters:
        data : array-like
            The data to fit the algorithm to.

        Returns:
        self : object
            Returns the instance itself.
        """
        pass

    @abstractmethod
    def get_structure(self):
        """
        Get the learned structure.

        Returns:
        structure : object
            The learned structure.
        """
        pass

