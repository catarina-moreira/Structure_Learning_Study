These R codes implement our Bayesian method for learning directed networks from interventional experimental data

Specifically:

mcmc_dag.R : contains the main MCMC algorithm for posterior inference on DAGs from interventional data

move_dag.R : performs one move from a DAG to an adjacent DAG (implements the proposal distribution over the space of DAGs)

marg_like_gaussian.R : contains functions to compute the marginal likelihood of Gaussian DAG models from a mixture of (observational and) interventional data

posterior_summaries.R : produce posterior summaries, such as Maximum A Posteriori (MAP) DAG estimate and Posterior Probabilities of edge Inclusion (PPI), starting from the output of mcmc_dag.R

gmInt.R : implements mcmc_dag.R on the gmInt dataset of Kalish et al. (2012)