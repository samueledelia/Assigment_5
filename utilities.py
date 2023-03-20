import numpy as np
from scipy.stats import norm

def analytical_normal_measures(alpha, weights, returns):
    x = np.log(returns)  # log return vector
    mu = x.mean()
    sigma = x.cov()
    norm_inv = norm.ppf(alpha)
    mu_port = np.dot(weights, mu)
    sigma_port = np.dot(weights, np.dot(sigma, weights))

    # Compute the Var and the ES
    var = mu_port + norm_inv * sigma_port
    es = 1/((1-alpha)*np.sqrt(2*np.pi))*np.exp(-(norm_inv**2)/2)
    return var, es