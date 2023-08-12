def estimate_bn_params(data, parent_dict):
    
    # Initialize a dictionary to hold the parameters for each node
    params = {}

    # Loop over each node in the network
    for node, parents in parent_dict.items():
        if len(parents) == 0:
            # If the node has no parents, estimate the parameters of the marginal distribution
            mean = np.mean(data.loc[:, node])
            std = np.std(data.loc[:, node])
            params[node] = {'mean': mean, 'std': std}
        else:
            # If the node has parents, estimate the parameters of the conditional distribution
            # todo: one prior for the noise, one prior for the coefficients and one prior for the graph
            reg = LinearRegression().fit(data.loc[:, parents], data.loc[:, node])
            residuals = data.loc[:, node] - reg.predict(data.loc[:, parents])
            std_resid = np.std(residuals)
            params[node] = {'coefficients': reg.coef_, 'intercept': reg.intercept_, 'std_resid': std_resid}

    return params

# score
def compute_log_likelihood(data, params, parent_dict):
    log_likelihood = 0

    for node, node_params in params.items():
        if len(parent_dict[node]) == 0:
            # If the node has no parents, compute the log-likelihood of the data given the marginal distribution
            log_likelihood += norm.logpdf(data.loc[:, node], loc=node_params['mean'], scale=node_params['std'])
        else:
            # If the node has parents, compute the log-likelihood of the data given the conditional distribution
            predicted_values = node_params['intercept'] + np.dot(data.loc[:, parent_dict[node]], node_params['coefficients'])
            residuals = data.loc[:, node] - predicted_values
            log_likelihood += norm.logpdf(residuals.values, loc=0, scale=node_params['std_resid'])
        
    return log_likelihood

def uniform_prior():
    return 1.0

def compute_gaussian_posterior(graph :  nx.DiGraph, data : pd.DataFrame, my_prior):
    # Define the prior probability of the structure
    prior = my_prior  
    
    # Compute the parent dictionary from the graph
    parent_dict = compute_parent_dict(graph)

    # Estimate the parameters of the Gaussian distributions for each node
    params = estimate_bn_params(data, parent_dict)

    # Compute the log-likelihood of the data given the structure and parameters
    log_likelihood = compute_log_likelihood(data, params, parent_dict)

    # Compute the posterior probability
    posterior = np.exp(np.log(prior) + log_likelihood)

    return np.mean(posterior)


def f_X_G_unbounded(df: pd.DataFrame, G: nx.DiGraph):
    total_rss = 0.0
    
    # For each node in the graph
    for node in G.nodes():
        # If the node has parents (is not a root)
        if G.in_degree(node) > 0:
            # Extract the data for the node and its parents
            y = df[node].values
            X = df[list(G.predecessors(node))].values
            
            # Fit a linear regression model
            reg = LinearRegression().fit(X, y)
            
            # Compute the predicted values
            y_pred = reg.predict(X)
            
            # Compute the residual sum of squares for this node and add to total
            rss = np.sum((y - y_pred) ** 2)
            total_rss += rss
    
    print( total_rss)
    return total_rss

def sigmoid(x):
    """Sigmoid function to map any value between 0 and 1."""
    return 1 / (1 + np.exp(-x))

def f_X_G(df: pd.DataFrame, G: nx.DiGraph):
    total_rss = 0.0
    
    # For each node in the graph
    for node in G.nodes():
        # If the node has parents (is not a root)
        if G.in_degree(node) > 0:
            # Extract the data for the node and its parents
            y = df[node].values
            X = df[list(G.predecessors(node))].values
            
            # Fit a linear regression model
            reg = LinearRegression().fit(X, y)
            
            # Compute the predicted values
            y_pred = reg.predict(X)
            
            # Compute the residual sum of squares for this node and add to total
            rss = np.sum((y - y_pred) ** 2)
            total_rss += rss
    
    # Normalize the total RSS value using the sigmoid function
    # We use a scaling factor (e.g., 1e-5) to adjust the sensitivity of the sigmoid function
    scaling_factor = 1e-5
    constrained_value = sigmoid(-scaling_factor * total_rss)
    
    return constrained_value


def f_X_G_2(df: pd.DataFrame, G: nx.DiGraph):
    return 1.0


def log_marginal_likelihood(df: pd.DataFrame, G: nx.DiGraph, a: float, b: float):
    # Compute f(X, G)
    f_val = f_X_G(df, G)

    
    # Compute n (number of data points) and p (number of variables)
    n, p = df.shape
    
    # Compute a' and b' values
    a_prime = (n + p) / 2 + a
    b_prime = f_val / 2 + b
    
    # Compute the logarithm of the marginal likelihood using the provided formula
    log_ml = a_prime * np.log(b_prime) - gammaln(a_prime)

    return log_ml

def safe_exp(x: float, max_val: float = 50.0, min_val: float = -50.0):
    """Safely exponentiates a value by bounding it between min_val and max_val."""
    bounded_x = np.clip(x, min_val, max_val)
    return np.exp(bounded_x)


def marginal_likelihood(df: pd.DataFrame, G: nx.DiGraph, a: float, b: float):
    log_ml = log_marginal_likelihood(df, G, a, b)
    return safe_exp(log_ml)