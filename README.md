# Structure_Learning_Study

https://pyagrum.readthedocs.io/en/1.7.1/BNLearning.html#pyAgrum.BNLearner.useMIIC

Learning methods:
- useMIIC() 
    - Multivariate Information based Inductive Causation (MIIC)
- useGreedyHillClimbing()
- useLocalSearchWithTabuList() 
    - implements a greedy search in which we allow applying at most N consecutive graph changes that decrease the score. To prevent infinite loops, when using local search, you should use a structural constraint that includes a tabu list of at least N elements.
- useK2() 
    - needs total ordering of the variables (https://github.com/danieleninni/bayesian-network-k2-algorithm)
- use3off2() 
    - https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0856-x
- useEM()

Score methods:
- useScoreAIC()
- useScoreBD()
- useScoreBDeu() - 
- useScoreBIC()
- useScoreK2()
- useSoreLog2Likelihood()

Priors:
- useSmoothingPrior(weight=1)
- useDirichletPrior()
- useBDeuPrior()
- useAprioriSmoothing()
- useAprioriDirichlet()
- useAprioriBDeu()

Correction:
- useMDLCorrection()
- useNMLCorrection()