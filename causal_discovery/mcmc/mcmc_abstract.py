from abc import ABC, abstractmethod

from causal_discovery.SGS import SGS

# create a MCMC abtract class
class MCMC_abstract(ABC):
    
    # the class constructor takes a dataset and a number of iterations and optionally a graph
    def __init__(self, df, iterations, G_initial = None):
        self.df = df
        self.iterations = iterations
        
        if G_initial is None:
            # if no graph is provided, initialize a  graph using the PC Algorithm
            SGS_model = SGS()
            SGS_model.fit(data)
            self.G_initial = SGS_model.get_structure()
        else:
            self.G_initial = G_initial # otherwise use the provided graph
            
        self.nodes = list(G_initial.nodes())
    
    @abstractmethod
    def compute_log_acceptance_ratio(self, score_old, score_new):
        pass
    
    @abstractmethod
    def compute_acceptance_ratio(self, score_old, score_new):
        pass
    
    @abstractmethod
    def run():
        pass  
    
    # the getGraph method returns the graph structure
    def getInitialGraph(self):
        return self.G_initial

    # the getNodes method returns the list of nodes present in the graph
    def getNodes(self):
        return list(self.G_initial.nodes())
    
    # the getSampleSize method returns the number of samples in the dataset
    def getSampleSize(self):
        return self.df.shape[0]
    
    # the getData method returns the dataset
    def getData(self):
        return self.df
    
    # the getIterations method returns the number of iterations
    def getIterations(self):
        return self.iterations
    