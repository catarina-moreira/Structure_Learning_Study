################
## gmInt data ##
################

# load required libraries

library(pcalg)

# Load the data

data(gmInt)

X = gmInt$x
n = dim(X)[1]
q = dim(X)[2]

# Build the (n,q) matrix D linking observations to intervention targets

D = matrix(0, n, q)

D[which(gmInt$target.index == 1), gmInt$targets[[1]]] = 1
D[which(gmInt$target.index == 2), gmInt$targets[[2]]] = 1
D[which(gmInt$target.index == 3), gmInt$targets[[3]]] = 1

# Fix prior hyperparameters (of Wishart prior on Omega)

a = q
U = diag(1, q)

# Fix number of MCMC iterations and burn in and run the MCMC algorithm

S    = 12000
burn = 2000

source("mcmc_dag.R")

out = mcmc_dag_int(X = X, S = S, burn = burn, a = a, U = U, D = D)

#########################
## Posterior summaries ##
#########################

# Recover Maximum A Posterior DAG (MAP) estimate and Posterior Probabilities of edge Inclusion (PPI)

source("posterior_summaries.R")

str(out)

out_summary = get_summaries_from_posterior(Graphs = out$DAG_post, X = out$X, D = out$D, a = a, U = U)

MAP = out_summary$MAP
PPI = out_summary$PPI

# Plot true and estimated I-EG

par(mfrow = c(1,2))

plot(dag2essgraph(gmInt$g, targets = gmInt$targets))
plot(dag2essgraph(as(MAP, "graphNEL"), targets = gmInt$targets))

# Plot PPI

library(fields)

colori = colorRampPalette(c('white','black'))

par(mar = c(1,1,1,1), oma = c(1,1,0.5,0.5), cex = 1, mgp = c(4,1,0), mfrow = c(1,1), mai = c(1,1,0.1,0.4))

labs = colnames(PPI)
image.plot(t(PPI[q:1,1:q]), col = colori(100), zlim = c(0,1), cex.sub = 1, xlab = "v", ylab = "u", axes = F, horizontal = F, legend.shrink = 1)
axis(1, at = seq(0, 1, l = q), lab = labs, las = 2, srt = 35, cex = 1.2)
axis(2, at = seq(0, 1, l = q), lab = labs[8:1], las = 2, srt = 35, cex = 1.2)
