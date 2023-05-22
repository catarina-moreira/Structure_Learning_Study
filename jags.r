install.packages('rjags')
install.packages('coda')
install.packages('devtools')

# specify the model
library('jags')

mod_string = "model{ 
  for (i in 1:n){
    y[i] ~ dnorm(mu, 1.0/sig2) 
  }
  mu ~ dt(0.0, 1.0/1.0, 1)
  sig2 = 1.0
  }"

# 2. Set up the model
set.seed(50)

y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3,1.9)
n = length(y)

data_jags = list(y=y, n=n)
params = c('mu')

inits = function(){
  inits = list("mu" = 0.0)
}

mod = jags.model(textConnection(mod_string), data=data_jags, inits=inits)

# run the MCMC sampler
update(mod, 500)

mod_sim = coda.samples(model=mod, variable.names = params, n.iter = 1000)
  
# post processing
library('coda')

plot(mod_sim)

  
  
