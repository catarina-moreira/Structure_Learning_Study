from abc import ABC, abstractmethod

from causal_discovery.SGS import SGS

# create a MCMC abtract class
class MCMC_abstract(ABC):
    
    # the class constructor takes a dataset and a number of iterations and optionally a graph
    def __init__(self, df, iterations, G = None):
        self.df = df
        self.iterations = iterations
        
        if G is None:
            # if no graph is provided, initialize a  graph using the PC Algorithm
            SGS_model = SGS()
            SGS_model.fit(data)
            self.G = SGS_model.get_structure()
        else:
            # otherwise use the provided graph
            self.G = G
    
    @abstractmethod
    def compute_log_acceptance_ratio(self, score_old, score_new):
        pass
    
    @abstractmethod
    def compute_acceptance_ratio(self, score_old, score_new):
        pass
    
    @abstractmethod
    def run_mcmc():
        pass  
    
    # the getGraph method returns the graph structure
    def getGraph(self):
        return self.G

    # the getNodes method returns the list of nodes present in the graph
    def getNodes(self):
        return list(self.G.nodes())
    
    # the getSampleSize method returns the number of samples in the dataset
    def getSampleSize(self):
        return self.df.shape[0]
    
    # the getData method returns the dataset
    def getData(self):
        return self.df
    
    # the getIterations method returns the number of iterations
    def getIterations(self):
        return self.iterations
    